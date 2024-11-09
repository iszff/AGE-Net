import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch
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
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")
    #Face Left Right metapath 3DGaze 3DHead 2DGaze 2DHead Rvec Svec GazeOrigin
    #Face Left Right Face_depth Left_depth Right_depth metapath 3DGaze 3DHead 2DGaze 2DHead Rvec Svec GazeOrigin
    face = line[0]
    lefteye = line[1]
    righteye = line[2]
    face_depth = line[3]

    name = line[6]
    gaze3d = line[7]
    gaze2d = line[9]
    head2d = line[10]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    headpose = np.array(head2d.split(",")).astype("float")
    headpose = torch.from_numpy(headpose).type(torch.FloatTensor)
    
    facepath = os.path.join(self.root, face)
    # print("================facepath================", facepath)
    # print("=====self.root======",self.root)
    # print("======face=====",face)


    fimg = cv2.imread(os.path.join(self.root, face))/255.0
    fimg = fimg.transpose(2, 0, 1)

    fdepth = cv2.imread(os.path.join(self.root, face_depth))/255.0
    fdepth = fdepth.transpose(2, 0, 1)

    rimg = cv2.imread(os.path.join(self.root, righteye))/255.0
    rimg = rimg.transpose(2, 0, 1)

    limg = cv2.imread(os.path.join(self.root, lefteye))/255.0
    limg = limg.transpose(2, 0, 1)

    img = {"left_rgb":torch.from_numpy(limg).type(torch.FloatTensor),
            "right_rgb":torch.from_numpy(rimg).type(torch.FloatTensor),
            "face_rgb":torch.from_numpy(fimg).type(torch.FloatTensor),
            "face_depth":torch.from_numpy(fdepth).type(torch.FloatTensor),
            "facename":face}


    # img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
    #        "right":torch.from_numpy(rimg).type(torch.FloatTensor),
    #        "face":torch.from_numpy(fimg).type(torch.FloatTensor),
    #        "head_pose":headpose,
    #        "name":name}

    return img, label

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True ):
  dataset = loader(labelpath, imagepath, header)
  logging.info(f"[Read Data]: Total num: {len(dataset)}")
  logging.info(f"[Read Data]: Label path: {labelpath}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,drop_last=True )
  return load


if __name__ == "__main__":
  path = './p00.label'
  d = loader(path)
  print(len(d))
  (data, label) = d.__getitem__(0)

