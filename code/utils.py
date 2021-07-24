import collections
import datetime
import itertools
import json
import math
import os
os.environ['MOVERSCORE_MODEL'] = "distilbert-base-multilingual-cased"
import moverscore_v2 as mv_sc
import torch
import torch.nn as nn
import tqdm
import numpy as np

def display(*msg):
    """Format message"""
    print(datetime.datetime.now(), *msg)
