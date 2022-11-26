#MAYBE WORKS

#adapted from https://github.com/fastai/fastai2/blob/f9231256e2a8372949123bda36e44cb0e1493aa2/fastai2/vision/widgets.py

import IPython.display
import multiprocessing as mp
from ipywidgets import HBox,VBox,widgets,Button,Checkbox,Dropdown,Layout,Box,Output,Label,FileUpload
from PIL import Image
from tqdm import tqdm
import pandas as pd

#all_wids = {}
#for i in tqdm(range(0,len(groups)), total=len(groups)):
#  for path in groups[i]:
#    all_wids[path] = i2widget(path)

#with mp.Pool(mp.cpu_count()) as p:
#  wid_list = list(p.imap(i2widget,all_paths))
#i2image = {i: wid for i,wid in zip(groups,wid_list)}


class DataLabeler:
  def __init__(self,groups,labels):
    self.groups = groups
    self.labels = labels
    self.df = pd.DataFrame(columns=['path','label'])
    all_wids = {}
    for i in tqdm(range(0,len(groups)), total=len(groups)):
      for path in groups[i]:
        file = open(path, "rb")
        this_image = file.read() 
        all_wids[path] = widgets.Image(value=this_image,width=256,height=256)
        file.close()
    self.all_wids = all_wids
    
  def display_pictures_button(self,groups,counter,df):
    if counter==len(self.groups):
      print("Nice work!")
      return 
    self.df = df
    print("Loading images in cluster "+str(counter)+"...")
    labels = self.labels
    self.boxes = []
    drops_reference = []
    urls = []
    for i in groups[counter]:
        drop_options = []
        drop_options.append("Cluster Label")
        drop_options.extend(self.labels)
        this_drop = Dropdown(options=drop_options,value=drop_options[0])
        drops_reference.append(this_drop)
        urls.append(i)
        box_i = VBox([self.all_wids[i], this_drop])
        self.boxes.append(box_i)
    

    def change_cluster(b):
      #counter += 1
      #print(b)
      values = [ref.value for ref in drops_reference]
      labels = []
      for i in values:
        if i == "Cluster Label":
          labels.append(cluster_label_box.value)
        else:
          labels.append(i)
      this_dict = {'path': urls, 'label' : labels}
      df2 = pd.DataFrame(this_dict)
      IPython.display.clear_output(wait=False)
      self.display_pictures_button(groups,counter+1,pd.concat([df,df2],ignore_index=True))
    
    #print(drops_reference[2].value)

    cluster_box = Button(description="Ready? Next cluster")

    cluster_box.on_click(change_cluster)
    cluster_label_box = Dropdown(options=labels)
    orig_list = [num for num in range(0,len(self.boxes),5)]

    new_list = [cluster_label_box]
    for item in orig_list:
        if orig_list[-1]==item:
            new_list.append(HBox(self.boxes[item:]))
        else:
            new_list.append(HBox(self.boxes[item:item+5]))
    new_list.append(cluster_box)
    display_widgets = VBox(new_list)
    display(display_widgets)
