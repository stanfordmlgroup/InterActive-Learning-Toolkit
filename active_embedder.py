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

from torchvision import transforms

class ActiveData(torch.utils.data.Dataset):
  def __init__(self,image_list,size,transform):
    self.image_list = image_list
    self.size = size
    self.transform=transform

  def __len__(self):
    return(len(self.image_list))

  def __getitem__(self,idx):
    img = Image.open(self.image_list[idx]).convert('RGB')
    if self.transform:
      img = self.transform(img)
    return img

CHECKPOINT_PATH = "simclr"

os.makedirs(CHECKPOINT_PATH, exist_ok=True)
pl.seed_everything(42)
import numpy as np

class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

def data_splitter(image_list,image_size,transform):

  clr_data = ActiveData(image_list,image_size,transform=ContrastiveTransformations(transform, n_views=2))
  clr_loader = DataLoader(clr_data, batch_size=1, shuffle=False)
  train_samples = int(np.floor(len(clr_loader)*.9))
  val_samples = int(len(clr_loader)-train_samples)
  train_use, val_use = torch.utils.data.random_split(clr_loader.dataset, (train_samples, val_samples))
  train_indices = train_use.indices
  val_indices = val_use.indices
  dict_loader = {'train': torch.utils.data.DataLoader(train_use,batch_size=64,shuffle=True,drop_last=True,pin_memory=True,num_workers=4),\
                     'val': torch.utils.data.DataLoader(val_use,batch_size=64,shuffle=False,drop_last=False,pin_memory=True,num_workers=4)}
  data_dict = {k: dict_loader[k].dataset for k in ['train','val']}
  data_indices = {'train': train_indices, 'val': val_indices}
  return dict_loader,data_dict,data_indices


class SimCLR(pl.LightningModule):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=2):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        imgs = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll

    def forward(self,x):
        return self.convnet(x)
    
    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')

def train_simclr(train_loader,val_loader,max_epochs=500, accelerator=None,**kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR'),
                         #Global seed set to 42accelerator="gpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(dirpath="simclr", mode='max', monitor='val_acc_top5'),
                                    LearningRateMonitor('epoch')])
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = SimCLR.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
        if accelerator!=None:
          trainer.fit(model, accelerator=accelerator,train_loader, val_loader)
        else: 
          trainer.fit(model, train_loader, val_loader)
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    return model

class Embedder():

  def __init__(self, image_list, image_size, embedding_option = -1,embeddings_loc=None,transform=None,embedding_ckpt=None):
    self.image_list = image_list
    self.image_size = image_size
    self.embedding_ckpt=embedding_ckpt
    if embedding_option==-1 and embeddings_loc==None:
      print("You need to select an option to create embeddings")
      raise NotImplementedError
    if embeddings_loc!=None:
      print("Thanks for supplying your embeddings")
      return
    self.embedding_option = embedding_option
    self.transform = transform
    self.load_embedder(embedding_option)

  def load_embedder(self,embedding_option):
    if embedding_option==1:
      model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
      self.model = model
      self.model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
      pipeline = torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms()
      self.pipeline = pipeline
    if embedding_option==2:
      model = SimCLR(hidden_dim=128,
                            lr=5e-4,
                            temperature=0.07,
                            weight_decay=1e-4)
      model.load_from_checkpoint('simclr/SimCLR.ckpt')
      self.model = model
      self.pipeline = self.transform
    if embedding_option==3:
      dict_loader,dict_data,dict_indices=data_splitter(self.image_list,self.image_size,self.transform)
      simclr_model = train_simclr(dict_loader["train"],dict_loader["val"],
                            hidden_dim=128,
                            lr=5e-4,
                            temperature=0.07,
                            weight_decay=1e-4,
                            max_epochs=1)
      self.model = simclr_model
      self.pipeline = self.transform
    if embedding_option==4: 
      model = SimCLR(hidden_dim=128,
                            lr=5e-4,
                            temperature=0.07,
                            weight_decay=1e-4)
      model.load_from_checkpoint(self.embedding_ckpt)
      self.model = model
      self.pipeline = self.transform
 
  def get_embeddings(self,transform):
    self.active_data = ActiveData(self.image_list,self.image_size,transform)
    self.active_loader = DataLoader(self.active_data, batch_size=1, shuffle=False)
    feature_tens = None
    for i, image in enumerate(self.active_loader):
      features = self.model(image)
      print(i)
      if feature_tens==None:
        feature_tens=features
      else:
        feature_tens=torch.cat((feature_tens,features),0)
    print(f"Features extracted have the shape: { feature_tens.shape }")
    return feature_tens

