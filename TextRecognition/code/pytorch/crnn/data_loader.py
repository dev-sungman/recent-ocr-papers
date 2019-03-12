from torch.utils.data import Dataset

import os
import json
import cv2

from PIL import Image

class CrnnDataLoader():
    def __init__(self, data_path, mode="train", transform=None):
        super().__init__()
        self.data_path = data_path
        self.config = json.load(open(os.path.join(data_path, "label.json")))
        self.mode = mode
        self.transform = transform
    
    def cls_len(self):
        return len(self.config["classes"])

    def get_cls(self):
        return self.config["classes"]

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode =="test":
            return int(len(self.config[self.mode]) * 0.01)
        return len(self.config[self.mode])

    def __getitem__(self, idx):
        name = self.config[self.mode][idx]["name"]
        text = self.config[self.mode][idx]["text"]
        
        #print(name, text)
        img = cv2.imread(os.path.join(self.data_path, name))

        #img = Image.open(os.path.join(self.data_path, name))
        seq = self.text_to_seq(text)
        print(seq, len(seq))
        sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": self.mode =="train"}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.config["classes"].find(c))
        return seq

