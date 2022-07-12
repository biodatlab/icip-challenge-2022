# Multitask learning via pseudo-label generation and ensemble prediction for parasitic egg cell detection: IEEE ICIP Challenge 2022

This repository hosts scripts for [ICIP 2022 challenge](https://2022.ieeeicip.org/challenges/)
for parasitic egg detection and classification in Microscopic images 
from [Biomedical and Data Lab at Mahidol University, Thailand](https://biodatlab.github.io/).
You can see the challenge website [here](https://icip2022challenge.piclab.ai/).
Our technique applies mulitask learning with pseudo mask generation which outperform single task model.
Moreover, we use continuous training using ensembling technique with pseudo label generation to achieve
the best-performing model (rank 3 on the test leaderboard). The details of the technique are discussed in our paper
_"Multitask learning via pseudo-label generation and ensemble prediction for parasitic egg cell detection: IEEE ICIP Challenge 2022"_
(to release at the [ICIP conference 2022](https://2022.ieeeicip.org/)).

![Proposed technique](/images/diagram.png)

Below you can see example bounding box and instance segmentation predictions of our final models on the given test set.

![Example predictions](/images/example_predictions.jpg)

## Setup Instructions for Linux

Assuming Pytorch with GPU support is installed.

```sh
pip install mmcv-full==1.15.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet==2.25.0
```

## Prediction

First download the trained models using the script,

```sh
cd models && bash get_model.sh
```

Then, run the following to get the ensembled predictions.

```sh
python predict_ensemble.py PATH_TO_IMAGES_FOLDER PATH_TO_MODEL_FOLDER --out SUBMISSION_JSON_FILE_NAME
# python predict_ensemble.py examples/ models/ --out pred_output.json
```

Downloading the pretrained models can take around 10 minutes from the repository. For prediction, it takes
around 20 minutes to predict and ensemble on the official test set of around 1650 images on a single NVIDIA RTX2080Ti.

## Results and Models
Individual models with their leaderboard scores, configs and checkpoints are shown in the table below. 
| Backbone          | Architecture | Epochs | mIoU (Leaderboard) | Config                                                                                               | Checkpoint                                                                                          |
|-------------------|--------------|--------|--------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| HRNet             | CascadeRCNN  | 10     | 0.927              | [config](https://f003.backblazeb2.com/file/icip-weights/cascade-rcnn-hrnetv2p-w32-10epoch.py)        | [ckpt](https://f003.backblazeb2.com/file/icip-weights/cascade-rcnn-hrnetv2p-w32-10epoch.pth)        |
| HRNet             | HTC          | 10     | 0.928              | [config](https://f003.backblazeb2.com/file/icip-weights/htc_hrnetv2p_w32_10epoch.py)                 | [ckpt](https://f003.backblazeb2.com/file/icip-weights/htc_hrnetv2p_w32_10epoch.pth)                 |
| X-101-32x4d-dcnv2 | HTC          | 10     | 0.928              | [config](https://f003.backblazeb2.com/file/icip-weights/htc_x101_64x4d_fpn_dconv_10epoch.py)         | [ckpt](https://f003.backblazeb2.com/file/icip-weights/htc_x101_64x4d_fpn_dconv_10epoch.pth)         |
| R-101-dcnv2       | GFL          | 10     | 0.923              | [config](https://f003.backblazeb2.com/file/icip-weights/gfl_r101_fpn_dconv_c3-c5_mstrain_10epoch.py) | [ckpt](https://f003.backblazeb2.com/file/icip-weights/gfl_r101_fpn_dconv_c3-c5_mstrain_10epoch.pth) |
| R-101-dcnv2       | TOOD         | 10     | 0.926              | [config](https://f003.backblazeb2.com/file/icip-weights/tood_r101_dconv_10epoch.py)                  | [ckpt](https://f003.backblazeb2.com/file/icip-weights/tood_r101_dconv_10epoch.pth)                  |


## Requirements

See requirements in `requirements.txt` including

- [mmdetection](https://github.com/open-mmlab/mmdetection)

## Authors

- Zaw Htet Aung, Department of Biomedical Engineering, Mahidol University, Nakhon Pathom, Thailand
- Kittinan Srithaworn, Looloo technology, Bangkok, Thailand
- Titipat Achakulvisut, Department of Biomedical Engineering, Mahidol University, Nakhon Pathom, Thailand
