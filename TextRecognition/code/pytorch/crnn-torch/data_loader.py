import torch

import os
import json
import cv2

class DataLoader():
    def __init__(self, data_path, mode="train", transform=None):
        self.data_path = data_path
        self.config = json.load(open(os.path.join(data_path, "label.json")))
        self.mode = mode
        self.transform = transform

    def get_name_text(self):
        name = self.config[self.mode]["classes"]["name"]
        text = self.config[self.mode]["classes"]["text"]

        img = cv2.imread(os.path.join(self.data_path, name))
        seq = self.text_to_seq(text)
        sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": self.mode == "train"}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.config["classes"].find(c)+1)
        return seq
