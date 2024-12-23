import numpy as np
import pandas as pd
import pickle
import copy
import argparse
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from utils import *
from sloth import *
from definitions import *

#python experiment.py --min_models 2

def prep_data(data, benchs_names, min_models=2):
    data['T'] = data['Pretraining Data Size (T)']
    data['S'] = data['#Params (B)']
    data['F'] = data['FLOPs (1E21)']
    data['family'] = data['Family']
    data = data.sort_values(by=['family','S']).reset_index(drop=True)
    data['logF'] = np.log(data['F'])
    data['logT'] = np.log(data['T'])
    data['logS'] = np.log(data['S'])
    data['logSlogT'] = data['logS']*data['logT']
    data['logS2'] = data['logS']**2
    data['logT2'] = data['logT']**2
    data = data[['family','logF','logT','logS','logSlogT','logS2','logT2','Instruct'] + benchs_names] 
    data = data.dropna(how='any')
    unique_families, counts_families = np.unique(data.family, return_counts=True)
    #print(unique_families, counts_families)
    avail_families = unique_families[counts_families>=min_models]
    data = data.loc[[f in avail_families for f in data.family]]
    return data, unique_families, avail_families

def prep_data2(data, test_families, benchs_names, n_train_models = 1):
    # creating train/test data
    data_train = data.loc[[f not in test_families for f in data.family]]
    data_test = {}
    for c in data_train.columns: data_test[c] = []
    data_test = pd.DataFrame(data_test)
    for f in test_families:
        data_test = pd.concat((data_test,data.loc[data.family==f].iloc[n_train_models:]), axis=0)
        data_train = pd.concat((data.loc[data.family==f].iloc[:n_train_models],data_train), axis=0)
    
    # creating Ds
    fam_encoder = LabelEncoder()
    fam_encoder.fit(data['family'])
    n_families = np.max(fam_encoder.transform(data['family']))+1
    D_train = np.zeros((data_train.shape[0],n_families))
    D_test = np.zeros((data_test.shape[0],n_families))
    for i,j in enumerate(fam_encoder.transform(data_train['family'])):
        D_train[i,j]=1
    for i,j in enumerate(fam_encoder.transform(data_test['family'])):
        D_test[i,j]=1
    
    # creating X,Y,F
    Y_train = np.array(data_train.loc[:,benchs_names]).astype(float)
    X_train = np.array(data_train.loc[:,['logT','logS','logSlogT']]).astype(float)
    X2_train = np.array(data_train.loc[:,['logS','logT','logSlogT','logS2','logT2']]).astype(float)
    F_train = np.array(data_train.loc[:,['logF']]).astype(float)
    Y_test = np.array(data_test.loc[:,benchs_names])
    X_test = np.array(data_test.loc[:,['logT','logS','logSlogT']]).astype(float)
    X2_test = np.array(data_test.loc[:,['logS','logT','logSlogT','logS2','logT2']]).astype(float) #
    F_test = np.array(data_test.loc[:,['logF']]) #

    return X_train, X2_train, F_train, D_train, Y_train, X_test, X2_test, F_test, D_test, Y_test, data_test.loc[:,['Instruct']]

def run_exp(X_train, Inter_train, F_train, D_train, Y_train, X_test, Inter_test, F_test, D_test, Y_test, Cs, ds = [1,2,3,4]):
    ### fitting models
    # models based in logF
    models_logF_sigmoid = []
    for j in range(Y_train.shape[1]):
        models_logF_sigmoid.append(Sloth(d=1))
        models_logF_sigmoid[-1].fit(F_train, Inter_train, Y_train[:,j:(j+1)], Cs[:,j:(j+1)], train_link=False, fit_C=True, positive_w=False, verbose=False)
    
    models_logF_trainlink = []
    for j in range(Y_train.shape[1]):
        model = models_logF_sigmoid[j]
        models_logF_trainlink.append(Sloth(d=1))
        models_logF_trainlink[-1].fit(F_train, Inter_train, Y_train[:,j:(j+1)],
                                      C0=model.C.numpy(),
                                      W1_X0=model.W1_X.numpy(),
                                      W1_D0=model.W1_D.numpy(),
                                      W20=model.W2.numpy(),
                                      b20=model.b2.numpy(),
                                      train_link=True, fit_C=True, positive_w=False, verbose=False)
    
    models_logF_sigmoid_faminter = []
    for j in range(Y_train.shape[1]):
        models_logF_sigmoid_faminter.append(Sloth(d=1))
        models_logF_sigmoid_faminter[-1].fit(F_train, D_train, Y_train[:,j:(j+1)], Cs[:,j:(j+1)], train_link=False, fit_C=True, positive_w=False, verbose=False)
    
    models_logF_trainlink_faminter = []
    for j in range(Y_train.shape[1]):
        model = models_logF_sigmoid_faminter[j]
        models_logF_trainlink_faminter.append(Sloth(d=1))
        models_logF_trainlink_faminter[-1].fit(F_train, D_train, Y_train[:,j:(j+1)],
                                               C0=model.C.numpy(),
                                               W1_X0=model.W1_X.numpy(),
                                               W1_D0=model.W1_D.numpy(),
                                               W20=model.W2.numpy(),
                                               b20=model.b2.numpy(),
                                               train_link=True, fit_C=True, positive_w=False, verbose=False)
        
    # models based in logS, logT
    models_logSlogT_sigmoid = []
    for j in range(Y_train.shape[1]):
        models_logSlogT_sigmoid.append(Sloth(d=1))
        models_logSlogT_sigmoid[-1].fit(X_train, Inter_train, Y_train[:,j:(j+1)], Cs[:,j:(j+1)], train_link=False, fit_C=False, positive_w=False, verbose=False)
    
    models_logSlogT_trainlink = []
    for j in range(Y_train.shape[1]):
        model = models_logSlogT_sigmoid[j]
        models_logSlogT_trainlink.append(Sloth(d=1))
        models_logSlogT_trainlink[-1].fit(X_train, Inter_train, Y_train[:,j:(j+1)],
                                           C0=model.C.numpy(),
                                           W1_X0=model.W1_X.numpy(),
                                           W1_D0=model.W1_D.numpy(),
                                           W20=model.W2.numpy(),
                                           b20=model.b2.numpy(),
                                          train_link=True, fit_C=True, positive_w=False, verbose=False)
    
    models_logSlogT_sigmoid_faminter = []
    for j in range(Y_train.shape[1]):
        models_logSlogT_sigmoid_faminter.append(Sloth(d=1))
        models_logSlogT_sigmoid_faminter[-1].fit(X_train, D_train, Y_train[:,j:(j+1)], Cs[:,j:(j+1)], train_link=False, fit_C=False, positive_w=False, verbose=False)
    
    models_logSlogT_trainlink_faminter = []
    for j in range(Y_train.shape[1]):
        model = models_logSlogT_sigmoid_faminter[j]
        models_logSlogT_trainlink_faminter.append(Sloth(d=1))
        models_logSlogT_trainlink_faminter[-1].fit(X_train, D_train, Y_train[:,j:(j+1)],
                                                   C0=model.C.numpy(),
                                                   W1_X0=model.W1_X.numpy(),
                                                   W1_D0=model.W1_D.numpy(),
                                                   W20=model.W2.numpy(),
                                                   b20=model.b2.numpy(),
                                                   train_link=True, fit_C=True, positive_w=False, verbose=False)

        
    # pca
    Y_hat_pca = []
    for d in ds:
        pca = PCA(n_components=d)
        pca.fit(Y_train)
        reg = LinearRegression(fit_intercept=False).fit(np.hstack((F_train, D_train)),
                                                        pca.transform(Y_train))
        Y_hat_pca.append(pca.inverse_transform(reg.predict(np.hstack((F_test, D_test)))).tolist())
        
    # factors
    models_factors_sigmoid = []
    for d in ds:
        models_factors_sigmoid.append(Sloth(d=d))
        models_factors_sigmoid[-1].fit(X_train, Inter_train, Y_train, Cs, train_link=False, fit_C=False, positive_w=False, verbose=False)
        
    
    models_factors_trainlink = []
    for j,d in enumerate(ds):
        model = models_factors_sigmoid[j]
        models_factors_trainlink.append(Sloth(d=d))
        models_factors_trainlink[-1].fit(X_train, Inter_train, Y_train,
                                         C0=model.C.numpy(),
                                         W1_X0=model.W1_X.numpy(),
                                         W1_D0=model.W1_D.numpy(),
                                         W20=model.W2.numpy(),
                                         b20=model.b2.numpy(),
                                         train_link=True, fit_C=True, positive_w=False, verbose=False)
        
    models_factors_sigmoid_faminter = []
    for d in ds:
        models_factors_sigmoid_faminter.append(Sloth(d=d))
        models_factors_sigmoid_faminter[-1].fit(X_train, D_train, Y_train, Cs, train_link=False, fit_C=False, positive_w=False, verbose=False)
        
    
    models_factors_trainlink_faminter = []
    for j,d in enumerate(ds):
        model = models_factors_sigmoid_faminter[j]
        models_factors_trainlink_faminter.append(Sloth(d=d))
        models_factors_trainlink_faminter[-1].fit(X_train, D_train, Y_train,
                                         C0=model.C.numpy(),
                                         W1_X0=model.W1_X.numpy(),
                                         W1_D0=model.W1_D.numpy(),
                                         W20=model.W2.numpy(),
                                         b20=model.b2.numpy(),
                                         train_link=True, fit_C=True, positive_w=False, verbose=False)

    ### results
    Y_hats = []
    
    # models based in logF
    Y_hats.append(np.hstack([m.predict(F_test, Inter_test) for m in models_logF_sigmoid]).tolist())
    Y_hats.append(np.hstack([m.predict(F_test, Inter_test) for m in models_logF_trainlink]).tolist())
    Y_hats.append(np.hstack([m.predict(F_test, D_test) for m in models_logF_sigmoid_faminter]).tolist())
    Y_hats.append(np.hstack([m.predict(F_test, D_test) for m in models_logF_trainlink_faminter]).tolist())
    
    # models based in logS, logT
    Y_hats.append(np.hstack([m.predict(X_test, Inter_test) for m in models_logSlogT_sigmoid]).tolist())
    Y_hats.append(np.hstack([m.predict(X_test, Inter_test) for m in models_logSlogT_trainlink]).tolist())
    Y_hats.append(np.hstack([m.predict(X_test, D_test) for m in models_logSlogT_sigmoid_faminter]).tolist())
    Y_hats.append(np.hstack([m.predict(X_test, D_test) for m in models_logSlogT_trainlink_faminter]).tolist())
    
    # factor models
    Y_hats += Y_hat_pca
    Y_hats += [m.predict(X_test, Inter_test).tolist() for m in models_factors_sigmoid]
    Y_hats += [m.predict(X_test, Inter_test).tolist() for m in models_factors_trainlink]
    Y_hats += [m.predict(X_test, D_test).tolist() for m in models_factors_sigmoid_faminter]
    Y_hats += [m.predict(X_test, D_test).tolist() for m in models_factors_trainlink_faminter]

    # output
    return np.abs(np.array(Y_hats)-np.array(Y_test)[None,:,:])



def run_exp2(X_train, D_train, Y_train, X_test, D_test, Y_test, Cs, ds = [1,2,3,4]):
    ### fitting models
        
    # models based in logS, logT    
    models_logSlogT_sigmoid_faminter = []
    for j in range(Y_train.shape[1]):
        models_logSlogT_sigmoid_faminter.append(Sloth(d=1))
        models_logSlogT_sigmoid_faminter[-1].fit(X_train, D_train, Y_train[:,j:(j+1)], Cs[:,j:(j+1)], train_link=False, fit_C=False, positive_w=False, verbose=False)
    
    models_logSlogT_trainlink_faminter = []
    for j in range(Y_train.shape[1]):
        model = models_logSlogT_sigmoid_faminter[j]
        models_logSlogT_trainlink_faminter.append(Sloth(d=1))
        models_logSlogT_trainlink_faminter[-1].fit(X_train, D_train, Y_train[:,j:(j+1)],
                                                   C0=model.C.numpy(),
                                                   W1_X0=model.W1_X.numpy(),
                                                   W1_D0=model.W1_D.numpy(),
                                                   W20=model.W2.numpy(),
                                                   b20=model.b2.numpy(),
                                                   train_link=True, fit_C=True, positive_w=False, verbose=False)

    # factors (data halves)
    models_factors_sigmoid_faminter_halves = {}
    models_factors_trainlink_faminter_halves = {}
    indices = [[0,1,2,3,4,5],
               [6,7,8,9,10,11]]
    
    for j in range(2):
        models_factors_sigmoid_faminter_halves[j] = []
        
        for d in ds:
            models_factors_sigmoid_faminter_halves[j].append(Sloth(d=d))
            models_factors_sigmoid_faminter_halves[j][-1].fit(X_train, D_train, Y_train[:,indices[j]], Cs[:,indices[j]], train_link=False, fit_C=False, positive_w=False, verbose=False)
            
        
        models_factors_trainlink_faminter_halves[j] = []
        for k,d in enumerate(ds):
            model = models_factors_sigmoid_faminter_halves[j][k]
            models_factors_trainlink_faminter_halves[j].append(Sloth(d=d))
            models_factors_trainlink_faminter_halves[j][-1].fit(X_train, D_train, Y_train[:,indices[j]],
                                             C0=model.C.numpy(),
                                             W1_X0=model.W1_X.numpy(),
                                             W1_D0=model.W1_D.numpy(),
                                             W20=model.W2.numpy(),
                                             b20=model.b2.numpy(),
                                             train_link=True, fit_C=True, positive_w=False, verbose=False)

    
    # factors (full data)
    models_factors_sigmoid_faminter = []
    for d in ds:
        models_factors_sigmoid_faminter.append(Sloth(d=d))
        models_factors_sigmoid_faminter[-1].fit(X_train, D_train, Y_train, Cs, train_link=False, fit_C=False, positive_w=False, verbose=False)
        
    
    models_factors_trainlink_faminter = []
    for k,d in enumerate(ds):
        model = models_factors_sigmoid_faminter[k]
        models_factors_trainlink_faminter.append(Sloth(d=d))
        models_factors_trainlink_faminter[-1].fit(X_train, D_train, Y_train,
                                         C0=model.C.numpy(),
                                         W1_X0=model.W1_X.numpy(),
                                         W1_D0=model.W1_D.numpy(),
                                         W20=model.W2.numpy(),
                                         b20=model.b2.numpy(),
                                         train_link=True, fit_C=True, positive_w=False, verbose=False)

    ### results
    Y_hats = []

    # models based in logS, logT
    Y_hats.append(np.hstack([m.predict(X_test, D_test) for m in models_logSlogT_sigmoid_faminter]).tolist())
    Y_hats.append(np.hstack([m.predict(X_test, D_test) for m in models_logSlogT_trainlink_faminter]).tolist())
    
    # factor models (data halves)
    Y_hats += [np.hstack((m0.predict(X_test, D_test).tolist(),m1.predict(X_test, D_test).tolist())) for m0,m1 in zip(models_factors_sigmoid_faminter_halves[0],models_factors_sigmoid_faminter_halves[1])]
    Y_hats += [np.hstack((m0.predict(X_test, D_test).tolist(),m1.predict(X_test, D_test).tolist())) for m0,m1 in zip(models_factors_trainlink_faminter_halves[0],models_factors_trainlink_faminter_halves[1])]

    # factors (full data)
    Y_hats += [m.predict(X_test, D_test).tolist() for m in models_factors_sigmoid_faminter]
    Y_hats += [m.predict(X_test, D_test).tolist() for m in models_factors_trainlink_faminter]

    # output
    return np.abs(np.array(Y_hats)-np.array(Y_test)[None,:,:])

def get_results(test_families, benchs_names):
    data = pd.read_csv('../data/data_v2.csv')
    if select_models:
        data = data.loc[[f not in families_to_delete for f in np.array(data['Family2'])]]
        
    data['Family'] = data['Family2']
    data, unique_families, avail_families = prep_data(data, benchs_names, min_models)

    X_train, X2_train, F_train, D_train, Y_train, X_test, X2_test, F_test, D_test, Y_test, Instruct_test = prep_data2(data, test_families, benchs_names, n_train_models=n_train_models)
    Inter_train = np.ones((X_train.shape[0],1))
    Inter_test = np.ones((X_test.shape[0],1))
    
    Cs = []
    for s in benchs_names:
        Cs.append(lower_bounds[s])
    Cs = np.array(Cs).astype(float)[None,:]

    if n_train_models==2:
        F_train = F_train*D_train
        F_test = F_test*D_test

    return run_exp(X_train, Inter_train, F_train, D_train, Y_train, X_test, Inter_test, F_test, D_test, Y_test, Cs), Instruct_test, test_families


if __name__=="__main__":
    device='cpu'
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--min_models', help="Minimum number of models per considered family. In [2,3].",type=int)
    parser.add_argument('--select_models', help="Delete 'bad' models from analysis", action='store_true')
    parser.set_defaults(select_models=False)
    args = parser.parse_args()
    min_models = args.min_models
    select_models = args.select_models
    assert min_models in [2,3]
    
    for exp in tqdm(range(3)):

        n_train_models = min_models-1
        benchs_names = benchs_names_list[exp]
        
        if n_train_models==1:
            test_families_list = test_families[1]
        elif n_train_models==2:
            test_families_list = test_families[2]
        else:
            raise NotImplementedError
        
        if select_models:
            thresh_MMLU = .35
            thresh_MMLU_PRO = .15
            data = pd.read_csv('../data/data_v2.csv')
            data = data.sort_values(by=['Family','#Params (B)'])
            biggest_model_data = data.drop_duplicates(subset=['Family'], keep='last')
            families_to_delete = np.unique(biggest_model_data.loc[(biggest_model_data.loc[:,'MMLU']<thresh_MMLU) | (biggest_model_data.loc[:,'MMLU-PRO']<thresh_MMLU_PRO)].Family).tolist()
            families_to_delete = np.unique(data.loc[[f in families_to_delete for f in data.Family]].Family2).tolist()
            test_families_list = [[x for x in y if not np.sum([z in families_to_delete for z in x])>0] for y in test_families_list]
            print('families_to_delete:',families_to_delete)
            
        errors = Parallel(n_jobs=-1, verbose=True)(delayed(get_results)(test_families, benchs_names) for test_families in test_families_list[exp])
        
        np.save(f'../results/errors_exp-{exp}_n-train-models-{n_train_models}_select-models-{select_models}.npy', {'out':errors})