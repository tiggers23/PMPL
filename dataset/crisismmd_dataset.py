import copy
import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption
import torch
import random
from tqdm import tqdm
import time
from random import randint
import numpy as np
from vilt.transforms import keys_to_transforms

class InputExample(object):
    def __init__(self, text, img_id,  text_label, image_label, information_label, label=None,id=None):
        """Constructs an InputExample."""
        self.text = text
        self.img_id = img_id
        self.label = label
        self.information_label = information_label
        self.text_label = text_label
        self.image_label = image_label
        self.id = id

class crisismmd_dataset(Dataset):
    def __init__(self, args, split, ann_file, test_file, transform, image_root, image_size, max_words=1024):
        self.ann = self._create_examples(ann_file, test_file, split)
        self.transform = keys_to_transforms(transform, size=image_size)[0]
        self.image_root = image_root
        self.max_words = max_words
        self.type = args.type
        self.label_map = {True:1, False:0}

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, '%s' % ann.img_id)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = ann.label
        information_label = ann.information_label

        # if ann.information_label == 1:
        #     information_label = 2
        # else:
        #     if label == ann.text_label:
        #         information_label = 0
        #     else:
        #         information_label = 1
                
        text_label = ann.text_label
        image_label = ann.image_label              
        text = ann.text
        temp_text = pre_caption(text, self.max_words)
        if self.type == 'image':
            temp_text = ''
        elif self.type =='text':
            image = torch.ones(image.size()).float()
            
        labels = torch.tensor(label, dtype=float)
        text_labels = torch.tensor(text_label, dtype=float)
        image_labels = torch.tensor(image_label, dtype=float)
        information_label = torch.tensor(information_label, dtype=float)
        return image, temp_text, labels, text_labels, image_labels, information_label, index

    def _create_examples(self, data_file, test_file, split):
        """Creates examples for the training and dev sets."""
        examples = []
        infor_sample = []
        notinfor_sample = []
        id = 0
        with open(data_file, encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                lineLS = eval(line)

                img_id = lineLS['id']
                
                text = lineLS['text']
                label = lineLS['label']
                text_label = lineLS['text_label']
                image_label = lineLS['image_label']
                information_label = lineLS['information_label']

                examples.append(InputExample(text=text, img_id=img_id, text_label=text_label, image_label=image_label, information_label=information_label, label=label,id=id))
                id += 1
                if information_label == 1:
                    infor_sample.append(1)
                else:
                    notinfor_sample.append(1)
        print('\n\n', f'{len(infor_sample)}    {len(notinfor_sample)}')
        if split == 'train':
            with open(test_file, encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                        lineLS = eval(line)

                        img_id = lineLS['id']
                        
                        text = lineLS['text']
                        label = lineLS['label']
                        text_label = lineLS['text_label']
                        image_label = lineLS['image_label']
                        information_label = lineLS['information_label']
                        # if information_label == 0 and random.random() <= 0.5:
                        #     examples.append(InputExample(text=text, img_id=img_id, text_label=text_label, image_label=image_label, information_label=information_label, label=label,id=id))
                        #     id += 1
                        # if information_label == 1 and random.random() <= 0.2:
                        #     examples.append(InputExample(text=text, img_id=img_id, text_label=text_label, image_label=image_label, information_label=information_label, label=label,id=id))
                        #     id += 1
                        id += 1
                        if information_label == 1:
                            infor_sample.append(1)
                        else:
                            notinfor_sample.append(1)

        return examples

    