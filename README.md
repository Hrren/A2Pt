# A<sup>2</sup>Pt: Anti-Associative Prompt Tuning for Open Set Visual Recognition

In this study, we unveiled that the unknown open set misclassified as a known close class is caused by the irrelevant co-occurrence association information.We accordingly proposed an anti-associative prompt tuning (A<sup>2</sup>Pt) approach to precisely suppress the associative influence.A<sup>2</sup>Pt leverages the cross-modal priors to generate the most class-related target feature and employ anti-association calibration with three loss functions to supervise the difference between target and association. Extensive experiments on CIFAR, TinyImageNet and ImageNet-21K-P benchmarks validated the effectiveness of the proposed approach, in striking contrast with the state-of-the-art.
The A<sup>2</sup>Pt approach provides fresh insight into the OSR problem.
<img width="773" alt="图片1" src="https://github.com/Hrren/A2Pt/assets/88883209/917d85fb-45b3-45f4-b43f-53edd57a1f53">

## <a name="ssb"/> :globe_with_meridians: The Semantic Shift Benchmark

Download instructions for the datasets in the SSB can be found at the links below. The folder `data/open_set_splits` contains pickle files with the class splits. For each dataset, `data` contains functions which return PyTorch datasets containing 'seen' and 'unseen' classes according to the SSB splits. 

* [ImageNet-21K-P](https://github.com/Alibaba-MIIL/ImageNet21K),

Links for the legacy open-set datasets are also available at the links below:
* [MNIST](https://pytorch.org/vision/stable/datasets.html),
[SVHN](https://pytorch.org/vision/stable/datasets.html),
[CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html),
[TinyImageNet](https://github.com/rmccorm4/Tiny-Imagenet-200)

For TinyImageNet, you also need to run `create_val_img_folder` in `data/tinyimagenet.py` to create
a directory with the test data.


## <a name="running"/> :running: Running
### Dependencies
```
pip install -r requirements.txt
```
### Config
---
Set paths to datasets  ```config.py```

Set ```SAVE_DIR``` (logfile destination) and ```PYTHON``` (path to python interpreter) in ```bash_scripts``` scripts.

### Scripts

**Train models**: To train models on all splits on a specified dataset, run:

```
bash osr_train._double.sh
```
## <a name="cite"/> :clipboard: Citation

If you use this code in your research, please consider citing our paper:
```
@artical{Ren:2023:A2pt,
author = {Hairui Ren, Fan Tang, Xingjia Pan, Juan Cao, Weiming Dong, Zhiwen Lin, Ke Yan, Changsheng Xu},
title = {Anti-Associative Prompt Tuning for Open Set Visual Recognition},
year = {2023},
booktitle = {IEEE TRANSACTIONS ON Multimedia},
}
```
