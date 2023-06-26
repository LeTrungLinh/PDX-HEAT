import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.outdoor_buildings import OutdoorBuildingDataset
from datasets.s3d_floorplans import S3DFloorplanDataset
from datasets.data_utils import collate_fn, get_pixel_features
from models.resnet import ResNetBackbone
from models.corner_models import HeatCorner
from models.edge_models import HeatEdge
from models.corner_to_edge import get_infer_edge_pairs
from utils.geometry_utils import corner_eval
import numpy as np
import cv2
import os
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from metrics.get_metric import compute_metrics, get_recall_and_precision
import skimage
import argparse

########## tuan #####################

def get_neighbor_node(current_node, edges, adjusted):
  neighbor_node = []
  for node, next_node in edges:
    if node == current_node :
      neighbor_node.append(next_node)
    elif next_node == current_node:
      neighbor_node.append(node)
  
  # neighbor_node = [node for node in neighbor_node if not adjusted[node]]
  return neighbor_node

def adjust_position(node1_position, node2_position, thresh=20):
  node1_x, node1_y = node1_position
  node2_x, node2_y = node2_position
  
  x_diff = abs(node1_x - node2_x)
  y_diff = abs(node1_y - node2_y)

  if x_diff < y_diff and abs(node1_x - node2_x) <= thresh:
    node1_x = node2_x = max(node1_x, node2_x)
  
  if x_diff > y_diff and abs(node1_y - node2_y) <= thresh:
    node1_y = node2_y = max(node1_y, node2_y)
  return (node1_x, node1_y), (node2_x, node2_y)

def save_vis(image, node_dct, info, output):
  corners = np.array(list(node_dct.values())).astype(np.int16)
  edges =  info['edges'].astype(np.int16)
  if edges is not None:
      preds = corners.astype(int)
      c_degrees = dict()
      for edge_i, edge_pair in enumerate(edges):
          # conf = (edge_confs[edge_i] * 2) - 1
          cv2.line(image, tuple(preds[edge_pair[0]]), tuple(preds[edge_pair[1]]), (255, 255 , 0), 1)
          c_degrees[edge_pair[0]] = c_degrees.setdefault(edge_pair[0], 0) + 1
          c_degrees[edge_pair[1]] = c_degrees.setdefault(edge_pair[1], 0) + 1
      for c in corners:
          cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 255, 0), -1)
  cv2.imwrite(output, image)
  print(f'Save img in {output}')

def straight_line(info,thresh):
  node_dct = {i:e for i, e in enumerate(info['corners'])}
  edges = info['edges']

  adjusted = {k: False for k in node_dct.keys()}

  for node, pos in node_dct.items():

    neighbor_node = get_neighbor_node(node, edges, adjusted)
    for neighbor in neighbor_node:
      node_new_position, neighbor_new_position = adjust_position(node_dct[node], node_dct[neighbor], thresh)
      
      node_dct[neighbor] = neighbor_new_position

      if not adjusted[node] or 1:
        adjusted[node] = True
        node_dct[node] = node_new_position
      
      if not adjusted[neighbor] or 1:
        adjusted[neighbor] = True

        node_dct[neighbor] = neighbor_new_position
    return node_dct

#############################3

def visualize_cond_generation(positive_pixels, confs, image, save_path, gt_corners=None, prec=None, recall=None,
                              image_masks=None, edges=None, edge_confs=None):
    image = image.copy()  # get a new copy of the original image
    if confs is not None:
        viz_confs = confs

    if edges is not None:
        preds = positive_pixels.astype(int)
        c_degrees = dict()
        for edge_i, edge_pair in enumerate(edges):
            # plot the edge which have acc more than 70%
            if edge_confs[edge_i]>0.7:
                conf = (edge_confs[edge_i] * 2) - 1
                cv2.line(image, tuple(preds[edge_pair[0]]), tuple(preds[edge_pair[1]]), (255 * conf, 255 * conf, 0), 2)
                c_degrees[edge_pair[0]] = c_degrees.setdefault(edge_pair[0], 0) + 1
                c_degrees[edge_pair[1]] = c_degrees.setdefault(edge_pair[1], 0) + 1


    for idx, c in enumerate(positive_pixels):
        if edges is not None and idx not in c_degrees:
            continue
        if confs is None:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, 255), -1)
        else:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, 255 * viz_confs[idx]), -1)
        
    if gt_corners is not None:
        for c in gt_corners:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 255, 0), -1)

    # if image_masks is not None:
    #     mask_ids = np.where(image_masks == 1)[0]
    #     for mask_id in mask_ids:
    #         y_idx = mask_id // 64
    #         x_idx = (mask_id - y_idx * 64)
    #         x_coord = x_idx * 4
    #         y_coord = y_idx * 4
    #         cv2.rectangle(image, (x_coord, y_coord), (x_coord + 3, y_coord + 3), (127, 127, 0), thickness=-1)

    cv2.imwrite(save_path, image)



def corner_nms(preds, confs, image_size):
    data = np.zeros([image_size, image_size])
    neighborhood_size = 5
    threshold = 0

    for i in range(len(preds)):
        data[preds[i, 1], preds[i, 0]] = confs[i]

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    results = np.where(maxima > 0)
    filtered_preds = np.stack([results[1], results[0]], axis=-1)

    new_confs = list()
    for i, pred in enumerate(filtered_preds):
        new_confs.append(data[pred[1], pred[0]])
    new_confs = np.array(new_confs)

    return filtered_preds, new_confs


def main(dataset, ckpt_path, image_size, viz_base, save_base, infer_times):
    ckpt = torch.load(ckpt_path)
    print('Load from ckpts of epoch {}'.format(ckpt['epoch']))
    ckpt_args = ckpt['args']


    backbone = ResNetBackbone()
    strides = backbone.strides
    num_channels = backbone.num_channels
    backbone = nn.DataParallel(backbone)
    backbone = backbone.cuda()
    backbone.eval()
    corner_model = HeatCorner(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                              backbone_num_channels=num_channels)
    corner_model = nn.DataParallel(corner_model)
    corner_model = corner_model.cuda()
    corner_model.eval()

    edge_model = HeatEdge(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                          backbone_num_channels=num_channels)
    edge_model = nn.DataParallel(edge_model)
    edge_model = edge_model.cuda()
    edge_model.eval()

    backbone.load_state_dict(ckpt['backbone'])
    corner_model.load_state_dict(ckpt['corner_model'])
    edge_model.load_state_dict(ckpt['edge_model'])
    print('Loaded saved model from {}'.format(ckpt_path))

    # get the positional encodings for all pixels
    pixels, pixel_features = get_pixel_features(image_size=image_size)

    viz_image = cv2.imread("./test_1.jpg")
    viz_image = cv2.resize(viz_image,(256,256))
    image = process_image(viz_image)
    with torch.no_grad():
        pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np = get_results(image, None, backbone,
                                                                                    corner_model,
                                                                                    edge_model,
                                                                                    pixels, pixel_features,
                                                                                    ckpt_args, infer_times,
                                                                                    corner_thresh=0.01,
                                                                                    image_size=image_size)



    # viz_image = data['raw_img'][0].cpu().numpy().transpose(1, 2, 0)
    # viz_image = (viz_image * 255).astype(np.uint8)
    




    
    # recon_path = os.path.join(viz_base, '{}_pred_corner.png'.format(data_i))
    # visualize_cond_generation(pred_corners_viz, pred_confs, viz_image, recon_path, gt_corners=None, prec=None,
    #                           recall=None)

    pred_corners, pred_confs, pos_edges = postprocess_preds(pred_corners, pred_confs, pos_edges)
    print('pred_confs', edge_confs)

    pred_data = {
        'corners': pred_corners,
        'edges': pos_edges,
    }
    node_dct = straight_line(pred_data,thresh = 30)
    node_dct = np.array(list(node_dct.values())).astype(np.int16) 
    # save_vis(viz_image, node_dct, pred_data,"./test_result.jpg")

    visualize_cond_generation(node_dct, pred_confs, viz_image, "./test_result.jpg", gt_corners=None, prec=None,
                            recall=None, edges=pos_edges, edge_confs=edge_confs)
    

def get_results(image, annot, backbone, corner_model, edge_model, pixels, pixel_features,
                args, infer_times, corner_thresh=0.5, image_size=256):
    image_feats, feat_mask, all_image_feats = backbone(image)
    pixel_features = pixel_features.unsqueeze(0).repeat(image.shape[0], 1, 1, 1)
    preds_s1 = corner_model(image_feats, feat_mask, pixel_features, pixels, all_image_feats)

    c_outputs = preds_s1
    # get predicted corners
    c_outputs_np = c_outputs[0].detach().cpu().numpy()
    pos_indices = np.where(c_outputs_np >= corner_thresh)
    pred_corners = pixels[pos_indices]
    pred_confs = c_outputs_np[pos_indices]
    pred_corners, pred_confs = corner_nms(pred_corners, pred_confs, image_size=c_outputs.shape[1])
    pred_corners, pred_confs, edge_coords, edge_mask, edge_ids = get_infer_edge_pairs(pred_corners, pred_confs)
    corner_nums = torch.tensor([len(pred_corners)]).to(image.device)
    max_candidates = torch.stack([corner_nums.max() * args.corner_to_edge_multiplier] * len(corner_nums), dim=0)

    all_pos_ids = set()
    all_edge_confs = dict()

    for tt in range(infer_times):
        if tt == 0:
            gt_values = torch.zeros_like(edge_mask).long()
            gt_values[:, :] = 2

        # run the edge model
        s1_logits, s2_logits_hb, s2_logits_rel, selected_ids, s2_mask, s2_gt_values = edge_model(image_feats, feat_mask,
                                                                                                 pixel_features,
                                                                                                 edge_coords, edge_mask,
                                                                                                 gt_values, corner_nums,
                                                                                                 max_candidates,
                                                                                                 True)
        # do_inference=True)

        num_total = s1_logits.shape[2]
        num_selected = selected_ids.shape[1]
        num_filtered = num_total - num_selected

        s1_preds = s1_logits.squeeze().softmax(0)
        s2_preds_rel = s2_logits_rel.squeeze().softmax(0)
        s2_preds_hb = s2_logits_hb.squeeze().softmax(0)
        s1_preds_np = s1_preds[1, :].detach().cpu().numpy()
        s2_preds_rel_np = s2_preds_rel[1, :].detach().cpu().numpy()
        s2_preds_hb_np = s2_preds_hb[1, :].detach().cpu().numpy()

        selected_ids = selected_ids.squeeze().detach().cpu().numpy()
        if tt != infer_times - 1:
            s2_preds_np = s2_preds_hb_np

            pos_edge_ids = np.where(s2_preds_np >= 0.9)
            neg_edge_ids = np.where(s2_preds_np <= 0.01)
            for pos_id in pos_edge_ids[0]:
                actual_id = selected_ids[pos_id]
                if gt_values[0, actual_id] != 2:
                    continue
                all_pos_ids.add(actual_id)
                all_edge_confs[actual_id] = s2_preds_np[pos_id]
                gt_values[0, actual_id] = 1
            for neg_id in neg_edge_ids[0]:
                actual_id = selected_ids[neg_id]
                if gt_values[0, actual_id] != 2:
                    continue
                gt_values[0, actual_id] = 0
            num_to_pred = (gt_values == 2).sum()
            if num_to_pred <= num_filtered:
                break
        else:
            s2_preds_np = s2_preds_hb_np

            pos_edge_ids = np.where(s2_preds_np >= 0.5)
            for pos_id in pos_edge_ids[0]:
                actual_id = selected_ids[pos_id]
                if s2_mask[0][pos_id] is True or gt_values[0, actual_id] != 2:
                    continue
                all_pos_ids.add(actual_id)
                all_edge_confs[actual_id] = s2_preds_np[pos_id]

    # print('Inference time {}'.format(tt+1))
    pos_edge_ids = list(all_pos_ids)
    edge_confs = [all_edge_confs[idx] for idx in pos_edge_ids]
    pos_edges = edge_ids[pos_edge_ids].cpu().numpy()
    edge_confs = np.array(edge_confs)

    if image_size != 256:
        pred_corners = pred_corners / (image_size / 256)

    return pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np


def postprocess_preds(corners, confs, edges):
    corner_degrees = dict()
    for edge_i, edge_pair in enumerate(edges):
        corner_degrees[edge_pair[0]] = corner_degrees.setdefault(edge_pair[0], 0) + 1
        corner_degrees[edge_pair[1]] = corner_degrees.setdefault(edge_pair[1], 0) + 1
    good_ids = [i for i in range(len(corners)) if i in corner_degrees]
    if len(good_ids) == len(corners):
        return corners, confs, edges
    else:
        good_corners = corners[good_ids]
        good_confs = confs[good_ids]
        id_mapping = {value: idx for idx, value in enumerate(good_ids)}
        new_edges = list()
        for edge_pair in edges:
            new_pair = (id_mapping[edge_pair[0]], id_mapping[edge_pair[1]])
            new_edges.append(new_pair)
        new_edges = np.array(new_edges)
        return good_corners, good_confs, new_edges


def process_image(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = skimage.img_as_float(img)
    img = img.transpose((2, 0, 1))
    img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
    img = torch.Tensor(img).cuda()
    img = img.unsqueeze(0)
    return img




def get_args_parser():
    parser = argparse.ArgumentParser('Holistic edge attention transformer', add_help=False)
    parser.add_argument('--dataset', default='outdoor',
                        help='the dataset for experiments, outdoor/s3d_floorplan')
    parser.add_argument('--checkpoint_path', default='./checkpoints_datnt200/ckpts_heat_sketch_256/checkpoint.pth',
                        help='path to the checkpoints of the model')
    # parser.add_argument('--checkpoint_path', default='./checkpoints/ckpts_heat_outdoor_256/checkpoint.pth',
    #                     help='path to the checkpoints of the model')
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--viz_base', default='./results/viz',
                        help='path to save the intermediate visualizations')
    parser.add_argument('--save_base', default='./results/npy',
                        help='path to save the prediction results in npy files')
    parser.add_argument('--infer_times', default=3, type=int)
    return parser


if __name__ == '__main__':
    import time 
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser('HEAT inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args.dataset, args.checkpoint_path, args.image_size, args.viz_base, args.save_base,
         infer_times=args.infer_times)
    end_time = time.perf_counter()
    print(f"processing time {end_time - start_time} seconds")