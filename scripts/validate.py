"""Validation script."""
import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP
from alphapose.utils.transforms import (flip, flip_heatmap,
                                        get_func_heatmap_to_coord)

def get_args():
    parser = argparse.ArgumentParser(description='AlphaPose Validate')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--checkpoint',
                        help='checkpoint file name',
                        required=True,
                        type=str)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str, default='0')
    parser.add_argument('--batch-size', dest='batch_size',
                        help='validation batch size',
                        type=int, default=40)
    parser.add_argument('--flip-test',
                        default=True,
                        dest='flip_test',
                        help='flip test',
                        action='store_true')
    parser.add_argument('--detector', dest='detector',
                        help='detector name', default="yolo")
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--iou-threshold', dest='iou_threshold', type=float, default=0.4)

    opt = parser.parse_args()
    

    gpus = [int(i) for i in opt.gpus.split(',')]
    opt.gpus = [gpus[0]]
    opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")

    return opt


def validate(m, heatmap_to_coord, opt, cfg):
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    eval_joints = det_dataset.EVAL_JOINTS

    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, drop_last=False)
    kpt_json = []
    m.eval()

    for inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds in tqdm(det_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.cuda() for inp in inps]
        else:
            inps = inps.cuda()
        output = m(inps)
        if opt.flip_test:
            if isinstance(inps, list):
                inps_flip = [flip(inp) for inp in inps]
            else:
                inps_flip = flip(inps)
            output_flip = flip_heatmap(m(inps_flip), det_dataset.joint_pairs, shift=True)
            output = (output + output_flip) / 2

        pred = output.cpu().data.numpy()
        assert pred.ndim == 4
        pred = pred[:, eval_joints, :, :]

        for i in range(output.shape[0]):
            bbox = crop_bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(pred[i][det_dataset.EVAL_JOINTS], bbox)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i, 0].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(scores[i] + np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    with open('./exp/json/validate_rcnn_kpt.json', 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP('./exp/json/validate_rcnn_kpt.json', ann_type='keypoints')
    return res['AP']


def validate_gt(m, heatmap_to_coord, opt, cfg):
    """
    sherk: 
    input:
        m: SPPE model
        cfg: configs
        heatmap_to_coord: heatmap_to_coord_simple function, convert keypoint heatmaps to global coordinates
        batch_size: keypoint net batch size
    """
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
    eval_joints = gt_val_dataset.EVAL_JOINTS

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    m.eval()

    for inps, labels, label_masks, img_ids, bboxes in tqdm(gt_val_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.cuda() for inp in inps]
        else:
            inps = inps.cuda()
        output = m(inps)
        if opt.flip_test:
            if isinstance(inps, list):
                inps_flip = [flip(inp) for inp in inps]
            else:
                inps_flip = flip(inps)
            output_flip = flip_heatmap(m(inps_flip), gt_val_dataset.joint_pairs, shift=True)
            output = (output + output_flip) / 2

        pred = output.cpu().data.numpy()
        assert pred.ndim == 4
        pred = pred[:, eval_joints, :, :]

        for i in range(output.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(pred[i][gt_val_dataset.EVAL_JOINTS], bbox)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    with open('./exp/json/validate_gt_kpt.json', 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP('./exp/json/validate_gt_kpt.json', ann_type='keypoints')
    return res['AP']


if __name__ == "__main__":
    opt = get_args()
    cfg = update_config(opt.cfg)
    # here m refers to the SPPE model
    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)


    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint))

    m = torch.nn.DataParallel(m, device_ids=[0]).cuda()    # will this be the cause of slow 2080Ti?
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    detbox_AP = 0.0
    gt_AP = 0.0
    with torch.no_grad():
        # detbox_AP = validate(m, heatmap_to_coord, opt, cfg)
        gt_AP = validate_gt(m, heatmap_to_coord, opt, cfg)
        

    print('##### gt box: {} mAP | det box: {} mAP #####'.format(gt_AP, detbox_AP))
