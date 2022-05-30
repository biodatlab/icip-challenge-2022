# Parasitic Egg Detection and Classification in Microscopic Images: ICIP 2022 Challenge

This repository hosts scripts for ICIP 2022 challenge for parasitic egg detection. See the challenge website
[here](https://icip2022challenge.piclab.ai/) and leaderboard [here](https://icip2022challenge.piclab.ai/leaderboard/).
Our technique uses continuous training using ensembling technique with pseudo label generation to achieve
the final model. The details of the technique are discussed in the paper.

## Setup Instructions (Linux)

Assuming Pytorch with GPU support is available

```sh
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html`
cd mmdetection && pip install -e .
```

## Training

Assuming root data directory is at `/workspace/data/Chula-ParasiteEgg-11/` and pretrained weights are downloaded to `mmdetection/pretrained_models`
Optionally, you can change `data_root` and `load_from` variables in `mmdetection/custom_models/configs/MODEL_CONFIG_NAME.py` as needed.

```sh
cd mmdetection
python tools/train.py custom_models/configs/MODEL_CONFIG_NAME.py
```

## Testing

```sh
cd mmdetection
python tools/test.py workdirs/MODEL_CONFIG_NAME/MODEL_CONFIG_NAME.py workdirs/MODEL_CONFIG_NAME/epoch_12.pth --format-only --options='jsonfile_prefix=./results'
```

This will create `results.bbox.json` file with the model predictions in mmdetection dir.

```sh
python tools/test.py workdirs/MODEL_CONFIG_NAME/MODEL_CONFIG_NAME.py workdirs/MODEL_CONFIG_NAME/epoch_12.pth --eval=bbox
```

to get bbox mAP without saving the results.

## Requirements

See requirements in `requirements.txt` including

- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [Tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
