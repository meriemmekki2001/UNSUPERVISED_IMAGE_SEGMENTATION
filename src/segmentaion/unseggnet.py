import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from ..utils.bilateral import bilateral_solver_output
from ..feature_extraction.unseggnet_extract import deep_features
from ..feature_extraction.unseggnet_feature_extractor import ViTExtractor
from torch_geometric.data import Data
from tqdm import tqdm
from ..utils import unseggnet_utils
from PIL import Image
from sklearn.cluster import KMeans
from ..models.unseggnet_gnn_pool import GNNpool 
from transformers import SamModel, SamProcessor


class Segmentation:
    def __init__(self, process, bs=False, epochs=20, resolution=(224, 224), activation=None, loss_type=None, threshold=None, conv_type=None):
        if process not in ["KMEANS_DINO", "DINO"]:
            raise ValueError(f'Process: {process} is not supported')
        self.process = process
        self.resolution = resolution
        self.bs = bs
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_type = loss_type
        self.threshold = threshold
        if process in ["DINO", "KMEANS_DINO"]:
            self.feats_dim = 384
            pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
            if not os.path.exists(pretrained_weights):
                url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
                unseggnet_utils.download_url(url, pretrained_weights)
            self.extractor = ViTExtractor(model_dir=pretrained_weights, device=self.device)
            self.model = GNNpool(self.feats_dim, 64, 32, 2, self.device, activation, loss_type, conv_type).to(self.device)
        torch.save(self.model.state_dict(), 'model.pt')
        self.model.train()

    def segment(self, image, mask):
        """
        @param image: Image to segment (numpy array)
        @param mask: Ground truth mask (binary numpy array)
        """
        
        image_tensor, image = unseggnet_utils.load_data_img(image, self.resolution)
        F = deep_features(image_tensor, self.extractor, device=self.device)

        if self.process == "KMEANS_DINO":
            kmeans = KMeans(n_clusters=2, random_state = 42)
            kmeans.fit(F)
            labels = kmeans.labels_
            S = torch.tensor(labels)
        else:
            W = unseggnet_utils.create_adj(F, self.loss_type, self.threshold)
            node_feats, edge_index, edge_weight = unseggnet_utils.load_data(W, F)
            data = Data(node_feats, edge_index, edge_weight).to(self.device)
            self.model.load_state_dict(torch.load('./model.pt', map_location=self.device))
            opt = optim.AdamW(self.model.parameters(), lr=0.001)

            for _ in range(self.epochs):
                opt.zero_grad()
                A, S = self.model(data, torch.from_numpy(W).to(self.device))
                loss = self.model.loss(A, S)
                loss.backward()
                opt.step()

            S = S.detach().cpu()
            S = torch.argmax(S, dim=-1)
                
        segmentation = unseggnet_utils.graph_to_mask(S, image_tensor, image)

        if self.bs:
            _ , segmentation = bilateral_solver_output(image, segmentation)

        
        segmentation = np.where(segmentation==True, 1,0).astype(np.uint8)  
        
        iou1 = Segmentation.iou(segmentation, mask)
        iou2 = Segmentation.iou(1-segmentation, mask)
        if iou2 > iou1:
            segmentation = 1 - segmentation

        segmentation_over_image  = unseggnet_utils.apply_seg_map(image, segmentation, 0.1)

        return max(iou1, iou2), segmentation, segmentation_over_image


    
    @staticmethod
    def iou(mask1, mask2):
        """
        Calculate Intersection over Union between two masks
        @param mask1: Binary numpy array
        @param mask2: Binary numpy array
        """
        x = mask1.ravel()
        y = mask2.ravel()
        intersection = np.logical_and(x, y)
        union = np.logical_or(x, y)
        similarity = np.sum(intersection)/ np.sum(union)
        return similarity