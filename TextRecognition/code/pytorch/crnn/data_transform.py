import cv2
import torch

class ToTensor(object):
    def __call__(self, sample):
        sample["img"] = torch.from_numpy(sample["img"].transpose((2,0,1))).float()
        sample["seq"] = torch.Tensor(sample["seq"]).int()
        sample["seq_len"] = torch.Tensor(sample["seq_len"]).int()
        print("len(sample[seq_len])",len(sample["seq_len"]))
        return sample

class Resize(object):
    def __init__(self, size=(320, 32)):
        self.size = size

    def __call__(self, sample):
        sample["img"] = cv2.resize(sample["img"], self.size)
        #sample["img"] = sample["img"].resize(self.size)
        return sample

class Rotation(object):
    def __init__(self, angle=20, fill_value=0, p=0.5):
        self.angle = angle
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h, w, _ = sample["img"].shape
        ang_rot = np.random.uniform(self.angle) - self.angle/2
        transform = cv2.getRotationMatrix2D((w/2, h/2), ang_rot, 1)
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue=self.fill_value)
        return sample
