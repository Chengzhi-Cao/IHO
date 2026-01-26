import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length # max_seq_length=60
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):  # idx=325
        example = self.examples[idx]
        image_id = example['id']    # 'CXR1677_IM-0446'
        image_path = example['image_path']# ['CXR1677_IM-0446/0.png', 'CXR1677_IM-0446/1.png']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')# image_1=[512,420]正面图
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')# image_2=[512,624]侧面图
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)# image=[2,3,224,224]
        report_ids = example['ids']     # report_ids=[1,45]
        report_masks = example['mask']  # report_masks=[1,45]
        seq_length = len(report_ids)    # seq_length=45
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
