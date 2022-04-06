from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def PreProcessing(args):
    ann_files1 = pd.read_csv("../../data/cigro_ad_infos_form - thesmc/table1.csv")
    ann_files2 = pd.read_csv("../../data/cigro_ad_infos_form - thesmc/table2.csv")
    ann_files3 = pd.read_csv("../../data/cigro_ad_infos_form - thesmc/table3.csv")
    ann_files4 = pd.read_csv("../../data/cigro_ad_infos_form - thesmc/table4.csv")
    ann_files5 = pd.read_csv("../../data/cigro_ad_infos_form - thesmc/table5.csv")
    ann_files6 = pd.read_csv("../../data/cigro_ad_infos_form - thesmc/table6.csv")


    files = list(ann_files1["광고명"]) + list(ann_files2["광고명"]) + list(ann_files3["광고명"]) + list(ann_files4["광고명"]) +\
        list(ann_files5["광고명"]) + list(ann_files6["광고명"])
    label = [0] * len(list(ann_files1["광고명"])) + [1] * len(list(ann_files2["광고명"])) + [2] * len(list(ann_files3["광고명"])) +\
        [3] * len(list(ann_files4["광고명"])) + [4] * len(list(ann_files5["광고명"])) + [5] * len(list(ann_files6["광고명"]))
    skf = StratifiedKFold(n_splits = 5, random_state=42, shuffle=True)

    folds = []
    for idx, (train_idx, val_idx) in enumerate(skf.split(files, label)):
        folds.append((train_idx, val_idx))

    train_idx, val_idx = folds[0]
    train_ann_files = pd.DataFrame(np.array([files, label])[:, train_idx].transpose())
    val_ann_files = pd.DataFrame(np.array([files, label])[:, val_idx].transpose())
    print('train data length: ', len(train_ann_files))
    print('val data length: ', len(val_ann_files))
    train_ann_files.columns = ['text', 'label']
    val_ann_files.columns = ['text', 'label']

    ann_files = pd.DataFrame(np.array([files, label]).transpose())
    ann_files.columns = ['text', 'label']
    ann_files.to_csv("data.csv")
            
    return train_ann_files, val_ann_files
    
