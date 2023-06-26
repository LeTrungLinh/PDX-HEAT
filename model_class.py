import torch
import torch.nn as nn
from models.resnet import ResNetBackbone
from models.corner_models import HeatCorner
from models.edge_models import HeatEdge
from models.corner_to_edge import get_infer_edge_pairs
import numpy as np 
import skimage
import scipy.ndimage.filters as filters
from datasets.data_utils import collate_fn, get_pixel_features
import cv2

class Heat_predictor():
    def __init__(self) -> None:
        self.model = None
        self.backbone = None
        self.corner_model = None
        self.edge_model = None
        self.ckpt_path = './checkpoints_datnt200/ckpts_heat_sketch_256/checkpoint.pth'
    def load_model(self):
        self.model = torch.load(self.ckpt_path)
        self.backbone = ResNetBackbone()
        strides = self.backbone.strides
        num_channels = self.backbone.num_channels
        self.backbone = nn.DataParallel(self.backbone)
        self.backbone = self.backbone.cuda()
        self.backbone.eval()
        self.corner_model = HeatCorner(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                                backbone_num_channels=num_channels)
        self.corner_model = nn.DataParallel(self.corner_model)
        self.corner_model = self.corner_model.cuda()
        self.corner_model.eval()

        self.edge_model = HeatEdge(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                            backbone_num_channels=num_channels)
        self.edge_model = nn.DataParallel(self.edge_model)
        self.edge_model = self.edge_model.cuda()
        self.edge_model.eval()

        self.backbone.load_state_dict(self.model['backbone'])
        self.corner_model.load_state_dict(self.model['corner_model'])
        self.edge_model.load_state_dict(self.model['edge_model'])
        print('Loaded saved model from {}'.format(self.ckpt_path))
    
    def process_image(self,img):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = skimage.img_as_float(img)
        img = img.transpose((2, 0, 1))
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
        img = torch.Tensor(img).cuda()
        img = img.unsqueeze(0)
        return img
    
    def corner_nms(self,preds, confs, image_size):
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
    
    def get_results(self,image, annot, backbone, corner_model, edge_model, pixels, pixel_features,
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
        pred_corners, pred_confs = self.corner_nms(pred_corners, pred_confs, image_size=c_outputs.shape[1])
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

    def postprocess_preds(self,corners, confs, edges):
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
    
    def wallExtraction_ai(self,img):
        image_size =256
        infer_times = 3 
        ckpt_args = self.model['args']
        # get the positional encodings for all pixels
        pixels, pixel_features = get_pixel_features(image_size=image_size)

        viz_image = np.array(img)
        image = self.process_image(viz_image)
        with torch.no_grad():
            pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np = self.get_results(image, None, self.backbone,
                                                                                        self.corner_model,
                                                                                        self.edge_model,
                                                                                        pixels, pixel_features,
                                                                                        ckpt_args, infer_times,
                                                                                        corner_thresh=0.01,
                                                                                        image_size=image_size)

        pred_corners, pred_confs, pos_edges = self.postprocess_preds(pred_corners, pred_confs, pos_edges)
        pred_data = {
            'corners': pred_corners,
            'edges': pos_edges,
            'edge_confs' :edge_confs
        }
        return pred_data
    

if __name__ == '__main__':
    model_predictor = Heat_predictor()
    load_model = model_predictor.load_model()
    img = cv2.imread('./test_1.jpg')
    img = cv2.resize(img, (256,256))
    result = model_predictor.wallExtraction_ai(img)
    print("ket qua la ", result)
