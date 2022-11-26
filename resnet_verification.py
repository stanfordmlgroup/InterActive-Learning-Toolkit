import os
import torch
import torchvision
import pytorch_lightning as pl
import urllib.request
from urllib.error import HTTPError
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchmetrics
import pandas as pd

from torchvision import transforms

class ManifestData(torch.utils.data.Dataset):
  def __init__(self,manifest,label_map,transform):
    self.manifest_df = pd.read_csv(manifest)
    self.transform=transform
    self.label_map = label_map

  def __len__(self):
    return(len(self.manifest_df))

  def __getitem__(self,idx):
    img = Image.open(self.manifest_df.loc[idx,"path"]).convert('RGB')
    if self.transform:
      img = self.transform(img)
    label = torch.tensor(self.label_map[self.manifest_df.loc[idx,"label"]])
    return (img,label)


class VerificationModel(pl.LightningModule):

    def __init__(self, lr, weight_decay, num_classes, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        self.convnet = torchvision.models.resnet18(pretrained=True)
        self.convnet.fc = torch.nn.Linear(512,num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def forward(self,x):
        return self.convnet(x)
    
    def training_step(self, batch, batch_idx):
        img, labels = batch
        logits = self(img)
        return self.loss(logits, labels)

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        logits = self(img)
        self.val_acc(logits, labels)
        self.log("val_acc",self.val_acc,on_step=False,on_epoch=True,prog_bar=True)

