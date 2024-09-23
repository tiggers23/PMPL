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
    def __init__(self, text, img_id,  text_label, image_label, information_label, label=None):
        """Constructs an InputExample."""
        self.text = text
        self.img_id = img_id
        self.label = label
        self.information_label = information_label
        self.text_label = text_label
        self.image_label = image_label

class mmimdb_dataset(Dataset):
    def __init__(self, args, split, ann_file, transform, image_root, image_size, max_words=1024):
        self.ann = self._create_examples(ann_file)
        self.transform = keys_to_transforms(transform, size=image_size)[0]
        self.image_root = image_root
        self.max_words = max_words
        self.type = args.type
        self.label_map = {True:1, False:0}
        

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = image_path = os.path.join(self.image_root, '%s.jpeg' % ann.img_id)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = ann.label
        information_label = ann.information_label
        # if information_label == 1:
        #     text_label = 1
        #     image_label = 1
        # else:
        #     if label == ann.text_label:
        #         text_label = 0
        #         image_label = 1
        #     else:
        #         text_label = 1
        #         image_label = 0 
        text_label = ann.text_label
        image_label = ann.image_label              
        text = ann.text
        if len(text)>1:
            temp_text = None
            for i in text:
                if temp_text==None:
                    temp_text = i
                else:
                    temp_text = temp_text+i
        else:
            temp_text = ann.text[0]        
        temp_text = pre_caption(temp_text, self.max_words)
        
            
        labels = torch.tensor(label, dtype=float)
        text_labels = torch.tensor(text_label, dtype=float)
        image_labels = torch.tensor(image_label, dtype=float)
        information_label = torch.tensor(information_label, dtype=float)
        return image, temp_text, labels, text_labels, image_labels, information_label, ann.img_id

    def _create_examples(self, data_file):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_file, encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                lineLS = eval(line)

                img_id = lineLS['id']
                text = lineLS['text']
                label = lineLS['label']
                text_label = lineLS['label']
                image_label = lineLS['label']
                information_label = 1

                examples.append(InputExample(text=text, img_id=img_id, text_label=text_label, image_label=image_label, information_label=information_label, label=label))
        return examples