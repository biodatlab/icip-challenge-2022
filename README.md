# Parasitic Egg Detection and Classification in Microscopic Images: ICIP 2022 Challenge

This repository hosts scripts for ICIP 2022 challenge for parasitic egg detection. See the challenge website
[here](https://icip2022challenge.piclab.ai/) and leaderboard [here](https://icip2022challenge.piclab.ai/leaderboard/).
Our technique uses continuous training using ensembling technique with pseudo label generation to achieve
the final model. The details of the technique are discussed in the paper.

Below you can see the prediction of our final models on the given test set.

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

## Requirements

See requirements in `requirements.txt` including

- [mmdetection](https://github.com/open-mmlab/mmdetection)
