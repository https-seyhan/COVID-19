#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""
import os
import pandas as pd
import math
import numpy as np
import torch
import os
import glob
import json
import torch.nn.functional as F
from tqdm import tqdm,trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pytorch_transformers import (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer)
from os import listdir
from os.path import isfile, join

documents = [['','']]
os.chdir("/home/saul/corona/CORD-19-research-challenge/2020-03-13") # change this to your local directory
data_file_address = 'all_sources_metadata_2020-03-13.csv'
json_files = '/home/saul/corona/CORD-19-research-challenge/2020-03-13/'

dirs = [x[1] for x in os.walk(json_files)]
#delete empty lists
dirs = [x for x in dirs if x !=[]]
print("Directories :", dirs[0])

ListFiles = os.walk(os.getcwd())
FileNames = []
for walk_output in ListFiles:
    for file_name in walk_output[-1]:
        if file_name.split(".")[-1] == "json":
            FileNames.append(file_name.split(".")[0])

list(set(FileNames))  #remove duplicate elements in the list
#print("Split Types :", FileNames)
print(len(data_file_address))

#df_data = pd.read_csv(data_file_address,sep=",",encoding="utf-8",names=['labels','texts'])
df_data = pd.read_csv(data_file_address,sep=",",encoding="utf-8")
print(len(df_data))
#print(df_data.columns)
print("Number of Json files :", len(FileNames))

person_dict = {"name": "Bob",
"languages": ["English", "Fench"],
"married": True,
"age": 32
}

with open('person.txt', 'w') as json_file:
  json.dump(person_dict, json_file)

person_string = '{"name": "Bob", "languages": "English", "numbers": [2, 1.6, null]}'
# Getting dictionary
person_dict = json.loads(person_string)

# Pretty Printing JSON string back
#print(json.dumps(person_dict, indent = 4, sort_keys=True))

# Reading the json as a dict
with open('/home/saul/corona/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/1a9c229f4f866db4f9c23a77b8c1275b52407c64.json') as json_data:
    data = json.load(json_data)

# using the from_dict load function. Note that the 'orient' parameter
#is not using the default value (or it will give the same error than you had)
# We transpose the resulting df and set index column as its index to get this result
print(data["body_text"])
jsondata = pd.DataFrame.from_dict(data, orient='index').T.set_index('paper_id')
#print(jsondata)
