import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import logging


def gazeto2d(gaze):
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([yaw, pitch])


class loader(Dataset):
    def __init__(self, path, root, header=True):
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    line = f.readlines()
                    if header:
                        line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                self.lines = f.readlines()
                if header:
                    self.lines.pop(0)

        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        name = line[3]
        left_gaze2d = line[8]
        right_gaze2d = line[9]
        lefteye = line[1]
        righteye = line[2]
        face = line[0]

        left_label = np.array(left_gaze2d.split(",")).astype("float")
        right_label = np.array(right_gaze2d.split(",")).astype("float")

        label = np.concatenate((left_label, right_label))

        face_img = cv2.imread(os.path.join(self.root, face))
        face_img = self.transform(face_img)

        rimg = cv2.imread(os.path.join(self.root, righteye))  #

        limg = cv2.imread(os.path.join(self.root, lefteye))
        limg = self.transform(limg)

        img = {"left_rgb": limg,
               "right_rgb": rimg,
               # "head_pose":headpose,
               "face_rgb": face_img,
               "name": name,
               "facename": face
               }

        return img, label


def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
    dataset = loader(labelpath, imagepath, header)
    logging.info(f"[Read Data]: Total num: {len(dataset)}")
    logging.info(f"[Read Data]: Label path: {labelpath}")
    load = DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return load


if __name__ == "__main__":
    path = './p00.label'
    d = loader(path)
    print(len(d))
    (data, label) = d.__getitem__(0)
