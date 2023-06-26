import cv2
import matplotlib.pyplot as plt
from datasets.outdoor_buildings_origin import OutdoorBuildingDataset
from torch.utils.data.dataloader import default_collate
import importlib
import numpy as np 
from qualitative_outdoor.plot_utils import *

def collate_fn(data):
    batched_data = {}
    for field in data[0].keys():
        if field in ['annot', 'rec_mat']:
            batch_values = [item[field] for item in data]
        else:
            batch_values = default_collate([d[field] for d in data])
        if field in ['pixel_features', 'pixel_labels', 'gauss_labels']:
            batch_values = batch_values.float()
        batched_data[field] = batch_values

    return batched_data
def convert_annot(annot):
    corners = np.array(list(annot.keys()))
    corners_mapping = {tuple(c): idx for idx, c in enumerate(corners)}
    edges = set()
    for corner, connections in annot.items():
        idx_c = corners_mapping[tuple(corner)]
        for other_c in connections:
            idx_other_c = corners_mapping[tuple(other_c)]
            if (idx_c, idx_other_c) not in edges and (idx_other_c, idx_c) not in edges:
                edges.add((idx_c, idx_other_c))
    edges = np.array(list(edges))
    gt_data = {
        'corners': corners,
        'edges': edges
    }
    return gt_data
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    DATAPATH = './data_linhlt4/'
    DET_PATH = './data/outdoor/det_final'
    train_dataset = OutdoorBuildingDataset(DATAPATH, DET_PATH, image_size=256, phase='valid')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0,
                                  collate_fn=collate_fn)

    for i, item in enumerate(train_dataloader):
        # print(item.keys())
        # if i < 20:
        # print(item['img'])
        viz_image = item['raw_img'][0].cpu().numpy().transpose(1, 2, 0)
        viz_image = (viz_image * 255).astype(np.uint8)
        image = viz_image.copy()
        

        corners = np.array(list(item['annot'][0].keys())).astype(np.int)
        edges = set()
        for c, others in item['annot'][0].items():
            for other_c in others:
                edge = (c[0], c[1], other_c[0], other_c[1])
                edge_2 = (other_c[0], other_c[1], c[0], c[1])
                if edge not in edges and edge_2 not in edges:
                    edges.add(edge)

        edges = np.array(list(edges)).astype(np.int)
        image = plot_preds(image, corners, edges)
        # plt.imshow(image)
        # plt.show()
        cv2.imwrite(f"./data_linhlt4/test/{i}.jpg",image)