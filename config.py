# ----------------------
# PROJECT ROOT DIR
# ----------------------
project_root_dir = ''

# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
#exp_root = ''        # directory to store experiment output (checkpoints, logs, etc)
exp_root = ''
save_dir = ''    # Evaluation save dir

# evaluation model path (for openset_test.py and openset_test_fine_grained.py, {} reserved for different options)
root_model_path = ''
root_criterion_path = ''

# -----------------------
# DATASET ROOT DIRS
# -----------------------
cifar_10_root = ''                                          # CIFAR10
cifar_100_root = ''                                        # CIFAR100
cub_root = ''                                                   # CUB
aircraft_root = ''                      # FGVC-Aircraft
mnist_root = ''                                              # MNIST
pku_air_root = ''                                   # PKU-AIRCRAFT-300
car_root = ""                                 # Stanford Cars
meta_default_path = ""              # Stanford Cars Devkit
svhn_root = ''                   # SVHN
tin_train_root_dir = ''        # TinyImageNet Train
tin_val_root_dir = ''     # TinyImageNet Val
imagenet_root = ''              # ImageNet-1K
imagenet21k_root = ''                       # ImageNet-21K-P
imagenet_a_val_root_dir = ''        #Imagenet-a
imagenet_sketch_val = ''
imagenet_v2_val = ''

# ----------------------
# FGVC / IMAGENET OSR SPLITS
# ----------------------
osr_split_dir = ''


# ----------------------

# ----------------------
imagenet_moco_path = ''
places_moco_path = ''
places_supervised_path = ''
imagenet_supervised_path = ''
