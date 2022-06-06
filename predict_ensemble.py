import os
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import (load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_dp, compat_cfg, get_device,
                         setup_multi_processes)
from tqdm import tqdm
from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion

torch.backends.cudnn.benchmark = True


NMS_THRESH=0.55
BOX_THRESH=0.55
PP_THRESH=0.55

CLASSES = ('Ascaris lumbricoides',
             'Capillaria philippinensis',
             'Enterobius vermicularis',
             'Fasciolopsis buski',
             'Hookworm egg',
             'Hymenolepis diminuta',
             'Hymenolepis nana',
             'Opisthorchis viverrine',
             'Paragonimus spp',
             'Taenia spp. egg',
             'Trichuris trichiura')
OUTPUT_CATEGORIES = [{ 'id': i, 'name': class_name, 'supercategory': None }
                     for i, class_name in enumerate(CLASSES)]

def format_annotations(file_path, image_dict):
    
    try:
        annotations = json.load(open(file_path, "r"))["annotations"]
    except TypeError as e:
        annotations = json.load(open(file_path, "r"))
    
    data = {}
    for ann in annotations:
        
        try:
            file_name = ann["file_name"]
        except KeyError as e: 
            image_id = ann["image_id"]
            file_name = "{}".format(image_id).zfill(4)
            file_name = f"{file_name}.jpg"
        
        image_info = image_dict[file_name]
        
        orig_bbox = ann["bbox"]
        bbox = norm_coco_bbox(ann["bbox"], image_info["width"], image_info["height"])
        score = ann["score"]
        category_id = ann["category_id"]

        if file_name not in data.keys():
            data[file_name] = {
                "original_boxes_list": [],
                "boxes_list": [],
                "labels_list": [],
                "scores_list": [],
            }

        data[file_name]["original_boxes_list"].append(orig_bbox)
        data[file_name]["boxes_list"].append(bbox)
        data[file_name]["labels_list"].append(category_id)
        data[file_name]["scores_list"].append(score)
        
    return data

def norm_coco_bbox(coco_bbox, w, h):
    # coco bbox format: [x,y,width,height]
    # Coordinates for boxes expected to be normalized e.g in range [0; 1]. Order: x1, y1, x2, y2.
    
    x1 = coco_bbox[0] / w
    x2 = x1 + (coco_bbox[2] / w)
    y1 = coco_bbox[1] / h
    y2 = y1 + (coco_bbox[3] / h)
    
    return [x1, y1, x2, y2]

def convert_norm_box_to_coco_bbox(norm_box, w, h):
    
    bbox = [
            norm_box[0] * w,
            norm_box[1] * h,
            (norm_box[2] - norm_box[0]) * w,
            (norm_box[3] - norm_box[1]) * h,
        ]
    return bbox

def process_fusion(image_names,
                    iou_thr,
                    skip_box_thr,
                    weight_dict, format_annotation_dict):
    fusions_dict = {}

    for path in tqdm(image_names):

        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []

        for model_name, format_ann in format_annotation_dict.items():
            #print(model_name)

            info = format_ann.get(path)

            if not info:
                # no prediction for this model
                continue

            boxes_list.append(info["boxes_list"])
            scores_list.append(info["scores_list"])
            labels_list.append(info["labels_list"])
            weights.append(weight_dict.get(model_name))

        if len(boxes_list) < 4:
            print(path, len(boxes_list) )
            print(labels_list)
            print(scores_list)
            print("-" * 20)
            
        boxes, scores, labels = weighted_boxes_fusion(
                                    boxes_list, 
                                    scores_list, 
                                    labels_list, 
                                    weights=weights, 
                                    iou_thr=iou_thr, 
                                    skip_box_thr=skip_box_thr
                                )

        fusions_dict[path] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
        
    return fusions_dict

def create_empty_coco_annotations(full_path_to_images):
    images_info = []
    image_dict = {}
    file_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.gif"]
    for extension in file_extensions:
        files = list(Path(full_path_to_images).glob(extension))
        files.sort()
        for i, filename in enumerate(files):
            image = Image.open(filename)
            width, height = image.size
            image_file_name = str(os.path.basename(filename))
            images_info.append([image_file_name, int(height), int(width)])
            image_dict[image_file_name] = {"width": width, "height": height, "image_id": i}

    images = [
        {
            "file_name": image_info[0],
            "height": image_info[1],
            "width": image_info[2],
            "id": i,
        }
        for i, image_info in enumerate(images_info, start=1)
    ]
    
    image_id_to_filename = { f"{image_info['id']}": image_info['file_name']
                            for image_info in images }
    
    annotations = {
        "categories": OUTPUT_CATEGORIES,
        "annotations": [],
        "images": images,
    }
    
    return annotations, image_id_to_filename, image_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description='Predicts and ensembles bboxes using pretrained detection models on a given set of images.')
    parser.add_argument('images_dir', help='path to folder containing images')
    parser.add_argument('models_dir', help='path to folder containing model checkpoints and configs')
    parser.add_argument('--out', help='output submission file in json format', default='submission_fusion_20220530_01_thr=0.55.json')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.json')):
        raise ValueError('The output file must be a JSON file.')

    # create list of test images in coco format since mmdet inference expects this
    annotations, image_id_to_filename, image_dict = create_empty_coco_annotations(args.images_dir)

    with open('annotations.json', 'w') as f:
        json.dump(annotations, f)

    config_paths = list(Path(args.models_dir).glob('*.py'))
    checkpoint_config_paths = [(config,
        Path(args.models_dir, config.name.split('.')[0] + '.pth')) for config in config_paths]

    # list of predictions from each model
    result_files = []

    ################################################## INFERENCE ##################################################
    for config_path, checkpoint_path in  checkpoint_config_paths:
        cfg = Config.fromfile(config_path)
        cfg.data.test.ann_file = 'annotations.json'
        cfg.data.test.img_prefix = args.images_dir
        cfg = compat_cfg(cfg)

        setup_multi_processes(cfg)
        cfg.gpu_ids = [args.gpu_id,]
        cfg.device = get_device()

        test_dataloader_default_args = dict(
                samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)

        test_loader_cfg = {
                **test_dataloader_default_args,
                **cfg.data.get('test_dataloader', {})
            }

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(dataset, **test_loader_cfg)

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, str(checkpoint_path), map_location='cpu')
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        model_name = checkpoint_path.name.split('.')[0]

        # run inference with the model on data
        outputs = single_gpu_test(model, data_loader, False, None, 0.3)

        if not os.path.exists('./predictions'):
            os.mkdir('./predictions')

        dataset.format_results(outputs, jsonfile_prefix=f"./predictions/{model_name}")
        
        raw_bbox_predictions = json.load(open(f'./predictions/{model_name}.bbox.json', 'r'))
        
        # prune and format the raw predictions for fusion
        annotations_ = []

        for r in raw_bbox_predictions:
            if (r['score'] > 0.1):
                anno = {}
                anno['id'] = r['image_id']
                anno['bbox'] = r['bbox']
                anno['category_id'] = r['category_id']
                anno['file_name'] = image_id_to_filename[str(r["image_id"])]
                anno['score'] = r['score']
                annotations_.append(anno)

        annotations = dict(annotations=annotations_)
        json.dump(annotations, open(f'./predictions/{model_name}_threshold=0.1.json', 'w'))
        
        result_files.append({
            "name": model_name,
            "path": f'./predictions/{model_name}_threshold=0.1.json',
            "weight": 0.2  # 2022-05-20
        })

    ################################################## ENSEMBLE ##################################################
    # format result files for fusion
    format_annotation_dict = {}
    file_paths = []
    for rf in result_files:
        format_ann = format_annotations(rf["path"], image_dict)

        format_annotation_dict[rf["name"]] = format_ann

        file_paths.extend(format_ann.keys())

    uniq_file_paths = sorted(list(set(file_paths)))

    weight_dict = {r["name"]: r["weight"] for r in result_files}

    fusions_dict = process_fusion(uniq_file_paths[:],
                                    NMS_THRESH, BOX_THRESH,
                                    weight_dict,
                                    format_annotation_dict)

    # convert fusion results to submission format
    items = []
    cnt_id = 0
    cnt_lower_threshold = 0
    threshold = PP_THRESH

    for file_name, info in fusions_dict.items():
        #print(file_name)
        
        image_info = image_dict[file_name]
        
        score_indexes = np.argsort(info["scores"])
        
        if len(info["scores"]) == 0:
            continue
        
        # filter max score
        index = np.argmax(info["scores"])
        
        cnt_annotate = 0
        found_category_id = None
        for index in score_indexes:
        
            label = info["labels"][index]
            score = info["scores"][index]
            box = info["boxes"][index]

            if score < threshold:
                continue

            category_id = int(label)
            coco_bbox = convert_norm_box_to_coco_bbox(box, image_info["width"], image_info["height"])

            item = {
                "id": cnt_id,
                "bbox": coco_bbox,
                "category_id": category_id,
                "file_name": file_name,
                "score": float(score),
            }
            items.append(item)

            cnt_id += 1
            cnt_annotate += 1
            found_category_id = category_id
            
    missing_pred_images = set(image_dict.keys()) - set(
        list(map(lambda r: r["file_name"], items))
    )

    # fill missing predictions with the selected model's outputs
    selected_model = "tood_r101_dconv_10epoch"
    missing_items = []
    for missing_im in missing_pred_images:
        file_name = missing_im
        image_info = image_dict[file_name]
        
        info = format_annotation_dict[selected_model].get(file_name, None)
        if info is None:
            print("Not found: {}".format(file_name))
            continue
        
        # filter max score
        index = np.argmax(info["scores_list"])
        
        label = info["labels_list"][index]
        score = info["scores_list"][index]
        box = info["boxes_list"][index]
        category_id = int(label)

        coco_bbox = convert_norm_box_to_coco_bbox(box, image_info["width"], image_info["height"])

        item = {
                "id": cnt_id,
                "bbox": coco_bbox,
                "category_id": category_id,
                "file_name": missing_im,
                "score": float(score),
            }
        items.append(item)
        cnt_id += 1

    submission_output = {"annotations": items}
    json.dump(submission_output, open(args.out, "w"), indent=2, sort_keys=False)

if __name__ == '__main__':
    main()