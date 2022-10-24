# Multitask learning via pseudo-label generation and ensemble prediction for parasitic egg cell detection: IEEE ICIP Challenge 2022

This repository hosts scripts for [ICIP 2022 challenge](https://2022.ieeeicip.org/challenges/)
for parasitic egg detection and classification in Microscopic images solution
developed by [Biomedical and Data Lab at Mahidol University, Thailand](https://biodatlab.github.io/).
You can see the challenge website [here](https://icip2022challenge.piclab.ai/).

Our technique applies:

- **mulitask learning** using pseudo mask generated with DeepMAC which outperforms single task model.
- **ensemble prediction** using multiple detection models
- **pseudo-label generation** on test dataset to continue training the models

Our best-performing model got rank 3 on the test leaderboard. The details of the technique are discussed in our paper
_"Multitask learning via pseudo-label generation and ensemble prediction for parasitic egg cell detection: IEEE ICIP Challenge 2022"_
(available at [IEEE ICIP 2022 at https://ieeexplore.ieee.org/document/9897464](https://ieeexplore.ieee.org/document/9897464)).
You can see the diagram of the proposed techniques below.

![Proposed technique](/images/diagram.png)

For qualitative analysis, you can see example bounding box and instance segmentation predictions of our final models on the given test set.

![Example predictions](/images/example_predictions.jpg)

And the example of single model prediction and ensemble prediction.

![Single model predictions](/images/single_model_predictions.png)

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
| Backbone | Architecture | Epochs | mIoU (Leaderboard) | Config | Checkpoint |
|-------------------|--------------|--------|--------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| HRNet | CascadeRCNN | 10 | 0.927 | [config](https://github.com/biodatlab/icip-challenge-2022/blob/main/configs/cascade-rcnn-hrnetv2p-w32-10epoch.py) | [ckpt](https://f003.backblazeb2.com/file/icip-weights/cascade-rcnn-hrnetv2p-w32-10epoch.pth) |
| HRNet | HTC | 10 | 0.928 | [config](https://github.com/biodatlab/icip-challenge-2022/blob/main/configs/htc_hrnetv2p_w32_10epoch.py) | [ckpt](https://f003.backblazeb2.com/file/icip-weights/htc_hrnetv2p_w32_10epoch.pth) |
| X-101-32x4d-dcnv2 | HTC | 10 | 0.928 | [config](https://github.com/biodatlab/icip-challenge-2022/blob/main/configs/htc_x101_64x4d_fpn_dconv_10epoch.py) | [ckpt](https://f003.backblazeb2.com/file/icip-weights/htc_x101_64x4d_fpn_dconv_10epoch.pth) |
| R-101-dcnv2 | GFL | 10 | 0.923 | [config](https://github.com/biodatlab/icip-challenge-2022/blob/main/configs/gfl_r101_fpn_dconv_c3-c5_mstrain_10epoch.py) | [ckpt](https://f003.backblazeb2.com/file/icip-weights/gfl_r101_fpn_dconv_c3-c5_mstrain_10epoch.pth) |
| R-101-dcnv2 | TOOD | 10 | 0.926 | [config](https://github.com/biodatlab/icip-challenge-2022/blob/main/configs/tood_r101_dconv_10epoch.py) | [ckpt](https://f003.backblazeb2.com/file/icip-weights/tood_r101_dconv_10epoch.pth) |

## Requirements

See requirements in `requirements.txt` including

- [mmdetection](https://github.com/open-mmlab/mmdetection)

## Citation

Cite an article as

> Aung, Zaw Htet, Kittinan Srithaworn, and Titipat Achakulvisut. "Multitask learning via pseudo-label
generation and ensemble prediction for parasitic egg cell detection: IEEE ICIP Challenge 2022."
In 2022 IEEE International Conference on Image Processing (ICIP), pp. 4273-4277. IEEE, 2022.

Or using Bibtex:

```
@inproceedings{aung2022multitask,
  title={Multitask learning via pseudo-label generation and ensemble prediction for parasitic egg cell detection: IEEE ICIP Challenge 2022},
  author={Aung, Zaw Htet and Srithaworn, Kittinan and Achakulvisut, Titipat},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={4273--4277},
  year={2022},
  organization={IEEE}
}
```

## Authors

- [Zaw Htet Aung](https://github.com/z-zawhtet-a), Department of Biomedical Engineering, Mahidol University, Thailand
- [Kittinan Srithaworn](https://github.com/kittinan), [Looloo technology](https://loolootech.com/), Thailand
- [Titipat Achakulvisut](github.com/titipata/), Department of Biomedical Engineering, Mahidol University, Thailand
