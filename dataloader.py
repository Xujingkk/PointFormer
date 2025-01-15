import torch.utils.data as data
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from tifffile import TiffFile
from PIL import Image
# import albumentations as A
import torch
import numpy as np
from skimage.util.shape import view_as_windows
import scipy.io as scio
from utils.aug import get_augmentation

from skimage.segmentation import find_boundaries
from skimage import measure, color

class PannukeDataset(data.Dataset):
    """
    img_path: original image
    masks: one-hot masks
    GT: tiff mask, one channel denote one instance
    """
    def __init__(self, data_root, is_train,seed=888,fold=1,output_stride=4):
        self.grid_size=256//output_stride

        self.images,self.edges, self.ins, self.cates= self.load_pannuke(data_root,fold)
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        self.num_class=6
        self.A_size =self.images.shape[0]

        self.mode='train' if is_train else "test"
        self.setup_augmentor(seed)

    def setup_augmentor(self, seed):
        self.shape_augs, self.input_augs = get_augmentation(self.mode, seed)

    def load_pannuke(self, data_root,fold=1):
        out_edges= []
        out_ins = []
        out_imgs=np.load(os.path.join(data_root,f'images/fold{fold}/images.npy')).astype(np.uint8)
        masks=np.load(os.path.join(data_root,f'masks/fold{fold}/masks.npy')).astype(np.int16)
        cate_masks=np.load(os.path.join(data_root,f'cates/fold{fold}/cates.npy')).astype(np.int16)
        for i in range(masks.shape[0]):
            raw_mask = masks[i, :, :, :]
            # print("raw", raw_mask)
            sem_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8)
            # swaping channels 0 and 5 so that BG is at 0th channel
            sem_mask = np.where(sem_mask == 5, 6, sem_mask)
            # print("")
            sem_mask = np.where(sem_mask == 0, 5, sem_mask)
            sem_mask = np.where(sem_mask == 6, 0, sem_mask)
            inst_mask = measure.label(sem_mask, connectivity=2)
            inst_mask [inst_mask >0] = 1
            inst_mask = inst_mask.astype(np.int16)
            instances = self.get_boundaries(raw_mask)
            out_edges.append(instances)
            out_ins.append(inst_mask)
            # print(np.unique(out_ins[i]))
        out_edges = np.stack(out_edges, 0)
        out_ins = np.stack(out_ins, 0)
        assert len(out_imgs.shape) == 4 and out_imgs.dtype == np.uint8
        assert len(out_ins.shape) == 3 and out_ins.dtype == np.int16, f'{out_ins.shape}, {out_ins.dtype}'
        assert out_ins.shape[0]==out_imgs.shape[0] and out_imgs.shape[0]==len(out_ins)
        print(f'processed data with size {len(out_imgs)}')
        # print("111", out_imgs.shape, out_masks.shape, out_edges.shape, out_ins.shape)
        # print("222", np.unique(out_masks), np.unique(out_edges), np.unique(out_ins))
        return out_imgs, out_edges, out_ins, cate_masks

    def __getitem__(self, index):
        # print("123")
        img = self.images[index]
        point= self.cates[index]
        edge = self.edges[index]
        ins = self.ins[index]

        shape_augs = self.shape_augs.to_deterministic()
        img = shape_augs.augment_image(img)
        point = shape_augs.augment_image(point)
        edge = shape_augs.augment_image(edge)
        ins = shape_augs.augment_image(ins)

        input_augs = self.input_augs.to_deterministic()
        img = input_augs.augment_image(img)

        image=self.transform(img)

        output={'image': image, 'point_labels':point, 'edge_labels':edge,'ins_labels':ins}
        return output

    def get_boundaries(self, raw_mask):
        '''
        for extracting instance boundaries form the goundtruth file
        '''
        bdr = np.zeros(shape=raw_mask.shape)
        for i in range(raw_mask.shape[-1] - 1):  # because last chnnel is background
            bdr[:, :, i] = find_boundaries(raw_mask[:, :, i], connectivity=1, mode='thick', background=0)
        bdr = np.sum(bdr, axis=-1)
        bdr[bdr > 1] = 1
        return bdr.astype(np.uint8)

    def __len__(self):
        return self.A_size
