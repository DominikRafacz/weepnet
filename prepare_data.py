import numpy as np
import pandas as pd
import os

img_size = (32, 32)
batch_size = 100
# build a structure for imagedatagenerator
img_dir = "data/train"
labels = pd.read_csv("data/trainLabels.csv").loc[:, 'label']
np.random.seed(420)
perm = np.random.permutation(50000)
inds_train = perm[0:40000]
inds_val = perm[40000:50000]
for ind in inds_train:
    cls = labels[ind]
    if not os.path.exists("data/train/{}".format(cls)):
        os.makedirs("data/train/{}".format(cls))
    if os.path.exists("data/train/{}.png".format(ind+1)):
        os.rename("data/train/{}.png".format(ind+1), "data/train/{0}/{1}.png".format(cls, ind+1))
for ind in inds_val:
    cls = labels[ind]
    if not os.path.exists("data/validation/{}".format(cls)):
        os.makedirs("data/validation/{}".format(cls))
    if os.path.exists("data/train/{}.png".format(ind+1)):
        os.rename("data/train/{}.png".format(ind+1), "data/validation/{0}/{1}.png".format(cls, ind+1))
