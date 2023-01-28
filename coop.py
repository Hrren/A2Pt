import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import random
import copy

_tokenizer = _Tokenizer()

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 args,
                 d_model=512,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.num = len(args.train_classes)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        #position encoding
        self.positional_embedding = nn.Parameter(torch.randn(self.num, d_model))
        self.positional_embedding_t = nn.Parameter(torch.randn(self.num, d_model))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    #def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
    def forward(self, vis, txt):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis = vis.unsqueeze(1).repeat(1, self.num, 1)
        vis = vis.permute(1, 0, 2)
        vis2 = self.norm1(vis)
        vis = vis.permute(1, 0, 2)
        vis = vis + self.positional_embedding.to(vis.dtype)
        vis = vis.permute(1, 0, 2)
        q = k = vis
        #q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        _, b, _ = vis.size()
        txt = txt.unsqueeze(1).repeat(1, b, 1)
        vis2 = self.multihead_attn(query=(vis2.permute(1,0,2) + self.positional_embedding.to(vis.dtype)).permute(1,0,2),
                                   key=(txt.permute(1,0,2) + self.positional_embedding_t.to(txt.dtype)).permute(1,0,2),
                                   value=vis
                                   )[0]
        #print('weights', weights[0, :, :], weights[0, :, :].size())

        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)

        vis = vis.permute(1, 0, 2)
   
        return vis


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        #print(x.size()) 
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
              
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.ctx_num
        ctx_init = cfg.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.image_size  # 224 for multi and cfg.image_size for others
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] # for better text add _not
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # for better text add _not
        #prompts = [prompt_prefix + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) #(20, 77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) #(20, 77, 512)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS (20, 1, 512)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS (20, 60, 512)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.ctp

    def forward(self):
        ctx = self.ctx
        #print('ctx', ctx)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        #print('prefix', prefix, prefix.size())
        #print('suffix', suffix, suffix.size())

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
          
        # for better text_prompt
        #if self.class_token_position == "end":
        #    prompts = []
        #    for i in range(self.n_cls):
        #        name_len = self.name_lens[i]
        #        prefix_i = prefix[i : i + 1, :, :]
        #        class_i = suffix[i : i + 1, :name_len, :]
        #        suffix_i = suffix[i : i + 1, self.n_ctx + name_len:, :]
        #        ctx_i_half1 = ctx[i : i + 1, :, :]
        #        ctx_i_half2 = ctx[i : i + 1, :, :]
        #        prompt = torch.cat(
        #            [
        #                prefix_i,     # (1, 1, dim)
        #                ctx_i_half1,  # (1, n_ctx//2, dim)
        #                class_i,      # (1, name_len, dim)
         #               ctx_i_half2,  # (1, n_ctx//2, dim)
         #               suffix_i,     # (1, *, dim)
         #           ],
         #           dim=1,
         #       )
         #       prompts.append(prompt)
         #   prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        
        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.multi_attention = TransformerDecoderLayer(cfg)
        self.ln_post = LayerNorm(512)
    def forward(self, image):
        #with torch.no_grad():
        image_features = self.image_encoder(image.type(self.dtype))
        #image_features = self.image_encoder(image.half())

        prompts = self.prompt_learner() #(20, 77, 512)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        vis = self.multi_attention(image_features, text_features)
        vis = self.ln_post(vis[:, 0, :])
        feature_back = image_features - vis
       
        simi_fb = torch.cosine_similarity(vis, feature_back, dim=1)
        simi_cf = torch.cosine_similarity(vis, image_features, dim=1)
        simi_cb = torch.cosine_similarity(feature_back, image_features, dim=1)
        vis = vis / vis.norm(dim=-1, keepdim=True)
        
        self.text_features = text_features
      
        
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        feature_back = feature_back / feature_back.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits_com = logit_scale * image_features @ text_features.t()
        logits_for = logit_scale * vis @ text_features.t()
        logits_back = logit_scale * feature_back @ text_features.t()
        
        return logits_for, logits_back, logits_com, simi_fb

