import numpy as np
import pandas as pd
import torch
from definitions import *

def sigmoid_np(z):
    return 1/(1+np.exp(-z))
    
def filter(s):
    try:s = s.split("/")[1]
    except: s = s
    try:s = s.split("__")[1]
    except: s = s
    return s.lower().replace("-hf","").replace("_","").replace("-","")

def get_true_indices(bool_array):
    return np.where(bool_array)[0]

def create_irt_train_mask(Y, validation_fraction, random_seed=42):
    mask = torch.ones(Y.shape, dtype=torch.bool).reshape(-1)
    local_state = np.random.RandomState(random_seed)
    mask[local_state.choice(len(mask), int(validation_fraction*len(mask)+1))] = False
    mask = mask.reshape(Y.shape)
    return mask

def load_data(todelete_names=None):
    data = pd.read_csv('../data/data_v2.csv')
    data['logS'] = np.log10(data['#Params (B)'])
    data['logT'] = np.log10(data['Pretraining Data Size (T)'])
    data['logSlogT'] = data['logS']*data['logT']
    data['GreatFamily'] = data['Family']
    data['Family'] = data['Family2']
    X_names = ['logS','logT','logSlogT'] 
    Y_names = ['GSM8K',
               'MATH Lvl 5',
               'GPQA',
               'MMLU',
               'MMLU-PRO',
               'BBH',
               'MUSR',
               'TruthfulQA',
               'ARC',
               'HellaSwag',
               'Winogrande',
               'IFEval']
    data = data.loc[:,['Model','Family','GreatFamily','Instruct']+X_names+Y_names].dropna()
    if todelete_names:
        for m in todelete_names:
            data = data.loc[data['Model'] != m]
    data = data.reset_index(drop=True)
    
    D = np.array(pd.get_dummies(np.array(data.Family))).astype(float)
    X = np.array(data.loc[:,X_names]).astype(float)
    Y = np.array(data.loc[:,Y_names]).astype(float)
    
    Cs = []
    for s in Y_names:
        Cs.append(lower_bounds[s])
    Cs = np.array(Cs).astype(float)[None,:]

    return data, D, X, Y, Cs, Y_names