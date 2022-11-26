from torchvision import datasets, models, transforms
import torch
from sklearn.cluster import KMeans
import numpy as np
#from torch_cluster import radius_graph, knn
import pandas as pd
from tqdm import tqdm
import os
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from active_embedder import data_splitter

class ProbCover():

  def __init__(self, save_dir='embeddings', image_list=None, image_size=224, embeddings_loc=None,num_classes=5,delta=None,input_size=32):
    batch_size = 256
    self.dir = save_dir
    self.dict_loader, self.dict_data, self.dict_indices = data_splitter(image_list, image_size,transform=None)
    self.embeddings_loc = embeddings_loc
    self.num_classes = num_classes
    if embeddings_loc == None:
      print("You need to use the Embedder class to create embeddings")
    if embeddings_loc[-3:]=="npy":
      embeddings = np.load(os.path.join(save_dir, to_load))
      self.embeddings = torch.tensor(embeddings)
    elif embeddings_loc[-2:]=="pt":
      self.embeddings = torch.load(embeddings_loc)
    elif embeddings_loc[-3:]=="pth":
      self.embeddings = torch.load(embeddings_loc)
    else:
      print("Embeddings are neither numpy nor pytorch files")
      raise NotImplementedError
    self.embeddings = self.embeddings / torch.norm(self.embeddings, dim=1, keepdim=True)
    if delta==None:
      delta = self.determine_delta()
    self.delta=delta
    
    self.graph_df = self.construct_graph(delta=self.delta)

  def determine_delta(self):
    print('Loading data')
    embeddings = self.embeddings
    print('Determining deltas')
    kmeans = KMeans(self.num_classes).fit(embeddings.detach().numpy())
    labels_dict = {i: kmeans.labels_[i] for i in range(len(kmeans.labels_))}
    last_purity = 1
    last_delta = {"key":0.2}
    for temp_delta in np.linspace(0.02,0.2,40):
      temp_graph = self.construct_graph(delta=temp_delta)
      print("made graph for ", temp_delta)
      temp_graph["x_label"]=temp_graph["x"].map(labels_dict)
      temp_graph["y_label"]=temp_graph["y"].map(labels_dict)
      temp_graph["match"]=temp_graph["x_label"]==temp_graph["y_label"]
      temp_group = temp_graph.groupby(["x"]).mean()
      purity = len(temp_group[temp_group["match"]>=.99])/len(temp_group)
      print(f'Purity={purity} with delta={temp_delta}')
      if last_purity > .95 and purity <= .95:
        print(f'best delta result is {last_delta["key"]}')  
        return last_delta["key"]
      last_purity = purity
      last_delta["key"] = temp_delta
    return -1

  def construct_graph(self, batch_size=256,delta=1):
    self.train_embeddings = self.embeddings[self.dict_indices['train']]
    self.train_embeddings = self.train_embeddings.detach().numpy()

    print('Finished loading data...')
    xs, ys, ds = [], [], []
    print(f'Start constructing graph using delta={delta}')

    cuda_feats = torch.tensor(self.train_embeddings)
    for i in range(self.train_embeddings.shape[0] // batch_size):
      cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
      dist = torch.cdist(cur_feats, cuda_feats)
      mask = dist < delta

      x, y = mask.nonzero().T
      xs.append(x.cpu() + batch_size * i)
      ys.append(y.cpu())
      ds.append(dist[mask].cpu())

    xs = torch.cat(xs).numpy()
    ys = torch.cat(ys).numpy()
    ds = torch.cat(ds).numpy()

    df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
    print(f'Finished constructing graph using delta={delta}')
    print(f'Graph contains {len(df)} edges.')
    return df

  def select_samples(self,budget):
    self.budgetSize = budget
    self.lSet = []
    print(f'Start selecting {self.budgetSize} samples.')
    selected = []
    aux = []

    edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
    covered_samples = self.graph_df.y[edge_from_seen].unique()
    cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
    for i in range(self.budgetSize):
      coverage = len(covered_samples) / self.train_embeddings.shape[0]

      degrees = np.bincount(cur_df.x, minlength=self.train_embeddings.shape[0])
      print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
      cur = degrees.argmax()
      new_covered_samples = cur_df.y[(cur_df.x == cur)].values
      assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
      cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

      covered_samples = np.concatenate([covered_samples, new_covered_samples])
      selected.append(cur)

    return selected

  def get_PC_loader(self,num_label):
    self.graph_df = self.construct_graph()
    self.oracle_results = self.select_samples(num_label)
    
class CoverNN():

  def __init__(self, save_dir='embeddings', image_list=None, image_size=224, embeddings_loc=None,num_classes=5,k=None,input_size=32):
    batch_size = 256
    self.dir = save_dir
    self.k = k
    self.dict_loader, self.dict_data, self.dict_indices = data_splitter(image_list, image_size,transform=None)
    self.embeddings_loc = embeddings_loc
    self.num_classes = num_classes
    if embeddings_loc == None:
      print("You need to use the Embedder class to create embeddings")
    if embeddings_loc[-3:]=="npy":
      embeddings = np.load(os.path.join(save_dir, to_load))
      self.embeddings = torch.tensor(embeddings)
    elif embeddings_loc[-2:]=="pt":
      self.embeddings = torch.load(embeddings_loc)
    elif embeddings_loc[-3:]=="pth":
      self.embeddings = torch.load(embeddings_loc)
    else:
      print("Embeddings are neither numpy nor pytorch files")
      raise NotImplementedError
    self.embeddings = self.embeddings / torch.norm(self.embeddings, dim=1, keepdim=True)
    
    self.graph_df = self.construct_graph(k=self.k)

  def construct_graph(self, batch_size=32,k=30):
    self.train_embeddings = self.embeddings[self.dict_indices['train']]
    self.train_embeddings = self.train_embeddings.detach().numpy()

    print('Finished loading data...')
    xs, ys, ds = [], [], []
    print(f'Start constructing graph using k={k}')

    cuda_feats = torch.tensor(self.train_embeddings)
    for i in range((self.train_embeddings.shape[0] // batch_size)+1):
      cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
      dist = torch.cdist(cur_feats, cuda_feats)
      d, mask = torch.topk(dist,30,dim=1,largest=False)
      ones_group = torch.ones(mask.shape)
      x, _ = torch.nonzero(ones_group,as_tuple=True)
      y = mask.flatten()
      xs.append(x.cpu() + batch_size * i)
      ys.append(y.cpu())
      ds.append(d.flatten())

    xs = torch.cat(xs).numpy()
    ys = torch.cat(ys).numpy()
    ds = torch.cat(ds).numpy()

    df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
    print(f'Finished constructing graph using k={k}')
    print(f'Graph contains {len(df)} edges.')
    return df

  def select_samples(self,budget):
    self.budgetSize = budget
    self.lSet = []
    print(f'Start selecting {self.budgetSize} samples.')
    selected = []
    aux = []
    edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
    covered_samples = self.graph_df.y[edge_from_seen].unique()
    cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
    ds = cur_df.groupby(['x']).mean()
    for i in range(self.budgetSize):
      coverage = len(covered_samples) / self.train_embeddings.shape[0]
      print(f'Iteration is {i}.\tMin distance is {ds.d.min():.3f}.\tCoverage is {coverage:.3f}')
      cur = ds['d'].idxmin()
      new_covered_samples = cur_df.y[(cur_df.x == cur)].values
      selected_here = []
      selected_here.append(cur)
      selected_here.extend(new_covered_samples)
      cur_df = cur_df[(~np.isin(cur_df.y,new_covered_samples))]
      covered_samples = np.concatenate([covered_samples, new_covered_samples])
      ds = ds[(~np.isin(ds.index, new_covered_samples))]
      selected.append(list(set(selected_here)))

    return selected
