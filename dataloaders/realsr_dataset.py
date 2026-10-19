import glob
import os
import random

import torch
from torchvision import transforms
from torch.utils import data as data
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from dataloaders.realesrgan import RealESRGAN_degradation

class PairedCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folder=None,
            null_text_ratio=0.2,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(PairedCaptionDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        self.gt_image_list = []
        # self.tag_path_list = []
        self.prompt_embeds_path_list = []
        self.pooled_prompt_embeds_path_list = []
        self.img_preproc = transforms.Compose([      
            transforms.ToTensor(),
        ])

        # for root_folder in root_folders:
        lr_path = root_folder +'/latent_lr'
        # tag_path = root_folder +'/tag'
        gt_path = root_folder +'/latent_hr'
        gt_image_path = root_folder +'/gt'
        prompt_embeds_path = root_folder + '/prompt_embeds'
        pooled_prompt_embeds_path = root_folder + '/pooled_prompt_embeds'

        self.lr_list += sorted(glob.glob(os.path.join(lr_path, '*.pt')))
        self.gt_list += sorted(glob.glob(os.path.join(gt_path, '*.pt')))
        self.gt_image_list += sorted(glob.glob(os.path.join(gt_image_path, '*.png')))
        # self.tag_path_list += glob.glob(os.path.join(tag_path, '*.txt'))
        self.prompt_embeds_path_list += sorted(glob.glob(os.path.join(prompt_embeds_path, '*.pt')))
        self.pooled_prompt_embeds_path_list += sorted(glob.glob(os.path.join(pooled_prompt_embeds_path, '*.pt')))

        # self.null_pooled_prompt_embeds_path = "/home/test/zw/DiT4SR/present/NULL_pooled_prompt_embeds.pt"
        self.null_pooled_prompt_embeds_path = os.path.join(pooled_prompt_embeds_path, 'NULL_pooled_prompt_embeds.pt')
        self.null_prompt_embeds_path = os.path.join(prompt_embeds_path, 'NULL_prompt_embeds.pt')
        # self.null_prompt_embeds_path = "/home/test/zw/DiT4SR/present/NULL_prompt_embeds.pt"

        self.prompt_embeds_path_list.remove(self.null_prompt_embeds_path)
        self.pooled_prompt_embeds_path_list.remove(self.null_pooled_prompt_embeds_path)


        def sample_id(path):
            return os.path.splitext(os.path.basename(path))[0]

        for name, paths in (
                ('latent_hr', self.gt_list),
                ('gt', self.gt_image_list),
                ('prompt_embeds', self.prompt_embeds_path_list),
                ('pooled_prompt_embeds', self.pooled_prompt_embeds_path_list)):
            if len(paths) != len(self.lr_list):
                raise ValueError(
                    f'Dataset size mismatch: latent_lr={len(self.lr_list)}, {name}={len(paths)}')
            for lr_file, paired_file in zip(self.lr_list, paths):
                if sample_id(lr_file) != sample_id(paired_file):
                    raise ValueError(
                        f'Dataset ID mismatch: latent_lr={lr_file}, {name}={paired_file}')

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):

        gt_image_path = self.gt_image_list[index]
        gt_image = Image.open(gt_image_path).convert("RGB")
        gt_image = self.img_preproc(gt_image)

        gt_path = self.gt_list[index]
        gt_latent = torch.load(gt_path)
        
        lq_path = self.lr_list[index]
        lq_latent = torch.load(lq_path)

        null_prompt_embeds = torch.load(self.null_prompt_embeds_path)
        null_pooled_prompt_embeds = torch.load(self.null_pooled_prompt_embeds_path)
        
        if random.random() < self.null_text_ratio:
            prompt_embeds = torch.load(self.null_prompt_embeds_path)
            pooled_prompt_embeds = torch.load(self.null_pooled_prompt_embeds_path)
        else:
            prompt_embeds_path = self.prompt_embeds_path_list[index]
            prompt_embeds = torch.load(prompt_embeds_path)

            pooled_prompt_embeds_path = self.pooled_prompt_embeds_path_list[index]
            pooled_prompt_embeds = torch.load(pooled_prompt_embeds_path)

        example = dict()
        example["gt_image"] = gt_image.squeeze(0) * 2.0 - 1.0
        example["conditioning_pixel_values"] = lq_latent.squeeze(0)
        example["pixel_values"] = gt_latent.squeeze(0)
        example['prompt_embeds'] = prompt_embeds.squeeze(0)
        example['pooled_prompt_embeds'] = pooled_prompt_embeds.squeeze(0)
        example['null_prompt_embeds'] = null_prompt_embeds.squeeze(0)
        example['null_pooled_prompt_embeds'] = null_pooled_prompt_embeds.squeeze(0)


        return example

    def __len__(self):
        return len(self.gt_list)


        
class PairedSROnlineTxtDatase_sd3(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((512, 512)),
                transforms.RandomHorizontalFlip(),
            ])

            self.gt_list = []
            assert len(args.dataset_txt_paths_list) == len(args.dataset_prob_paths_list)
            for idx_dataset in range(len(args.dataset_txt_paths_list)):
                with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_list)
                        self.gt_list += dataset_list
                        print(f'=====> append {len(self.gt_list) - gt_length} data.')
            # self.tag_path_list = []
            # self.tag_path_list += glob.glob(os.path.join("/home/test/zw/OSEDiff/preset/tag", '*.txt'))


    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            gt_img = self.crop_preproc(gt_img)

            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            prompt_embeds = torch.load("/home/test/zw/TSD-SR/dataset/default/prompt_embeds.pt", map_location="cpu").squeeze()
        # pooled_prompt_embeds = torch.load(self.pool_prompt_embeds_name[idx], map_location=self.device).squeeze()
            pooled_prompt_embeds = torch.load("/home/test/zw/TSD-SR/dataset/default/pool_embeds.pt", map_location="cpu").squeeze()

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["prompt_embeds_input"] = prompt_embeds
            example["pooled_prompt_embeds_input"] = pooled_prompt_embeds

            return example
