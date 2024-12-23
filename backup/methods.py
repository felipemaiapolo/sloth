import pickle
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import pickle
import torch
from torch import nn
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from utils import *

device='cuda'

def filter(s):
    try:s = s.split("/")[1]
    except: s = s
    try:s = s.split("__")[1]
    except: s = s
    return s.lower().replace("-hf","").replace("_","").replace("-","")
    
def search(s, s_list):
    scores = [fuzz.token_sort_ratio(filter(s), filter(s_try)) for s_try in s_list]
    return [s_list[np.argmax(scores)], np.max(scores)]

def create_irt_data(correctness, models_part, models_full):
    Y = 1*(correctness>.5)
    M = np.zeros((len(models_part), len(models_full)))
    for i,mod in enumerate(models_part):
        M[i,np.argmax(np.array(models_full)==mod)]=1
    return Y, M

def create_arena_data(arena_data, models_full):
    Y_arena = np.array(arena_data.loc[:,['winner_model_a', 'winner_model_b', 'winner_tie']])
    M_arena = np.zeros((Y_arena.shape[0], len(models_full)))
    for i in tqdm(range(Y_arena.shape[0])):
        M_arena[i, np.argmax(np.array(models_full) == filter(arena_data.model_a[i]))] = 1
        M_arena[i, np.argmax(np.array(models_full) == filter(arena_data.model_b[i]))] = -1
    return Y_arena, M_arena

class GlobalModel(nn.Module):
    def __init__(self, ds, device='cuda'):
        super(GlobalModel, self).__init__()
        self.ds = ds
        self.device = device
        self.sigmoid = nn.Sigmoid()  

    def check_inputs(self, Ys, Ms, models):
        n_llms = Ms[0].shape[1]
        assert len(Ys)==len(Ms)
        assert len(Ys)==len(models)
        assert np.mean([M.shape[1]==n_llms for M in Ms])==1
        
    def forward_irt(self, M_irt, Theta, Alpha, beta):
        logits = (M_irt@Theta@Alpha.T)-beta.T
        return self.sigmoid(logits)

    def forward_rasch(self, M_irt, Theta, beta):
        logits = (M_irt@Theta)-beta.T
        return self.sigmoid(logits)
        
    def forward_bt(self, M_bt, Theta, gamma, eta):
        l_1 = (M_bt@Theta@gamma)
        p_1 = self.sigmoid(l_1 - eta.exp()).reshape(-1,1)
        p_2 = self.sigmoid(-l_1 - eta.exp()).reshape(-1,1)
        p_tie = (1 - p_1 - p_2).reshape(-1,1)
        return torch.cat([p_1, p_2, p_tie], dim=1)
        
    def irt_loss(self, Y, M_irt, Theta, Alpha, beta, eps):
        p = self.forward_irt(M_irt, Theta, Alpha, beta)
        return -(Y*(p+eps).log() + (1-Y)*(1-p+eps).log()).mean()

    def rasch_loss(self, Y, M_irt, Theta, beta, eps):
        p = self.forward_rasch(M_irt, Theta, beta)
        return -(Y*(p+eps).log() + (1-Y)*(1-p+eps).log()).mean()
        
    def irt_loss2(self, Y, M_irt, Theta, Alpha, beta, eps):
        p_hat = self.forward_irt(M_irt, Theta, Alpha, beta).mean(1)
        p = Y.mean(1)
        return torch.abs(p-p_hat).mean()
                
    def bt_loss(self, Y, M_bt, Theta, gamma, eta, eps):
        p = self.forward_bt(M_bt, Theta, gamma, eta)
        return -(Y*(p+eps).log()).sum(1).mean()

    def create_theta(self, random_seed, n_llms, d, device):
        torch.manual_seed(random_seed)
        Theta = torch.nn.Parameter(torch.normal(0, 1, (n_llms, d), dtype=torch.float64, device=device))
        return Theta
    
    def create_irt_weights(self, random_seed, n_examples, d, device):
        torch.manual_seed(random_seed)
        Alpha = torch.nn.Parameter(torch.normal(0, 1, (n_examples, d), dtype=torch.float64, device=device))
        beta = torch.nn.Parameter(torch.normal(0, 1, (n_examples, 1), dtype=torch.float64, device=device))
        return {'Alpha':Alpha, 'beta':beta}

    def create_rasch_weights(self, random_seed, n_examples, d, device):
        torch.manual_seed(random_seed)
        beta = torch.nn.Parameter(torch.normal(0, 1, (n_examples, 1), dtype=torch.float64, device=device))
        return {'beta':beta}
        
    def create_bt_weights(self, random_seed, d, device):
        torch.manual_seed(random_seed)
        gamma = torch.nn.Parameter(torch.normal(0, 1, (d,1), dtype=torch.float64, device=device))
        eta = torch.nn.Parameter(torch.normal(0, 1, (1,), dtype=torch.float64, device=device)) #(1/d)*
        return {'gamma':gamma, 'eta':eta}

    def split_data(self, Ms, Ys, validation_fraction, random_seed):

        M_target, Y_target = Ms[0], Ys[0]
        M_other = np.vstack(Ms[1:])
        
        target_llm_ind = ((M_target).std(0)>0)
        target_llm_ind = [i for i in range(M_target.shape[1]) if target_llm_ind[i]]
        target_llm_ind = [m for i,m in enumerate(target_llm_ind) if (np.abs(M_other[:,target_llm_ind]).sum(0)>0)[i]]

        local_state = np.random.RandomState(random_seed)
        validation_ind = np.sort(local_state.choice(target_llm_ind, int(validation_fraction*len(target_llm_ind)+1))).tolist()
        train_rows = M_target[:,validation_ind].std(1)==0
        val_rows = M_target[:,validation_ind].std(1)>0
        
        M_target_train = M_target[train_rows]
        Y_target_train = Y_target[train_rows]
        M_target_val = M_target[val_rows]
        Y_target_val = Y_target[val_rows]

        return M_target_train, Y_target_train, M_target_val, Y_target_val

    def train(self, Ys, Ms, Y_val, M_val, d, norm_importances, outputs, lr, n_epochs, eps, n_llms, random_seed, patience=10, tol=1e-5, val_bool = True):
                    
        #Initializing parameters
        patience_count = 0
        weights = {'Theta': self.create_theta(random_seed, n_llms, d, device)}
        for k, out in enumerate(outputs):
            if out == 'irt':
                n_examples = Ys[k].shape[1]
                weights[k] = self.create_irt_weights(random_seed, n_examples, d, device)
            if out == 'rasch':
                n_examples = Ys[k].shape[1]
                weights[k] = self.create_rasch_weights(random_seed, n_examples, d, device)
            elif out == 'bt':
                weights[k] = self.create_bt_weights(random_seed, d, device)
        
        #Define optimizer for a specific parameter
        def optimize_parameter(parameter):
            optimizer = optim.LBFGS(parameter, lr=lr, line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                loss = 0
                for k, out in enumerate(outputs):
                    if out == 'irt': 
                        loss += norm_importances[k]*self.irt_loss(Ys[k], Ms[k], weights['Theta'], weights[k]['Alpha'], weights[k]['beta'], eps)
                    if out == 'rasch': 
                        loss += norm_importances[k]*self.rasch_loss(Ys[k], Ms[k], weights['Theta'], weights[k]['beta'], eps)
                    elif out == 'bt': 
                        loss += norm_importances[k]*self.bt_loss(Ys[k], Ms[k], weights['Theta'], weights[k]['gamma'], weights[k]['eta'], eps)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            return loss.item()
            
        #Fitting
        train_losses = []
        train_target_losses = []
        val_target_losses = []
        for epoch in range(n_epochs):

            #Defining params to optimize in each step (coordinate descent)
            parameters = [[weights['Theta']], []]
            for k, out in enumerate(outputs):
                if out == 'irt': 
                    parameters[-1] += [weights[k]['Alpha'], weights[k]['beta']]
                elif out == 'rasch': 
                    parameters[-1] += [weights[k]['beta']]
                elif out == 'bt': 
                    parameters[-1] += [weights[k]['eta'], weights[k]['gamma']] 
                    
            #Optimizing
            for parameter in parameters:
                loss = optimize_parameter(parameter)
            train_losses.append(loss)

            if val_bool:
                #Storing losses
                with torch.no_grad():
                    if outputs[0] == 'bt':
                        train_target_losses.append(self.bt_loss(Ys[0], Ms[0], weights['Theta'], weights[0]['gamma'], weights[0]['eta'], eps).item())
                    elif outputs[0] == 'irt':          
                        train_target_losses.append(self.irt_loss2(Ys[0], Ms[0], weights['Theta'], weights[0]['Alpha'], weights[0]['beta'], eps).item())
                    elif outputs[0] == 'rasch':
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                        
                with torch.no_grad():
                    if outputs[0] == 'bt':
                        val_target_losses.append(self.bt_loss(Y_val, M_val, weights['Theta'], weights[0]['gamma'], weights[0]['eta'], eps).item())
                    elif outputs[0] == 'irt':          
                        val_target_losses.append(self.irt_loss2(Y_val, M_val, weights['Theta'], weights[0]['Alpha'], weights[0]['beta'], eps).item())
                    else:
                        raise NotImplementedError
    
                #Break if the improvement is small
                if epoch>=1:
                    if val_target_losses[-1]-val_target_losses[-2]>0:
                        patience_count += 1
                        if patience_count == patience:
                            break
            else:
                if epoch>=1:
                    if train_losses[-2]-train_losses[-1]<tol:
                        break

        return weights, train_losses, train_target_losses, val_target_losses, epoch
        
    def fit(self, 
            Ys, # the first dataset is the target one
            Ms, # the first dataset is the target one
            outputs, #Ex: ['irt', 'bt', 'rasch', 'irt']
            importances_list,
            validation_fraction=.1,
            lr=1, n_epochs=500, eps = 1e-10, random_seed=42, verbose=True):

        ### Checks and defs
        assert validation_fraction>=0
        assert np.mean(np.array(importances_list)>0)==1
        if np.sum([o=='rasch' for o in outputs])>0: assert d==1
        if validation_fraction==0: 
            assert len(self.ds)==1
            assert len(importances_list)==1
            norm_importances = (np.array(importances_list[0])/np.sum(importances_list[0])).tolist()
            d = self.ds[0]
        self.check_inputs(Ys, Ms, outputs)
        n_llms = Ms[0].shape[1]
        ds = self.ds
        device = self.device
        
        ### Splitting target data
        Ms_full, Ys_full = copy.deepcopy(Ms), copy.deepcopy(Ys)
        Ms, Ys = copy.deepcopy(Ms), copy.deepcopy(Ys)
        if validation_fraction>0:
            Ms[0], Ys[0], M_val, Y_val = self.split_data(Ms, Ys, validation_fraction, random_seed)
            M_val = torch.tensor(M_val, requires_grad=False).double().to(self.device)
            Y_val = torch.tensor(Y_val, requires_grad=False).double().to(self.device)
            print(M_val.shape, Y_val.shape)
        else:
            M_val, Y_val = None, None
            
        ### Converting data to torch
        Ys = [torch.tensor(Y, requires_grad=False).double().to(self.device) for Y in Ys]
        Ms = [torch.tensor(M, requires_grad=False).double().to(self.device) for M in Ms]

        ### Using validation to find the best d
        if validation_fraction>0:
            best_loss = math.inf
            for d in tqdm(ds, disable = not verbose):
                for importances in importances_list:
    
                    norm_importances = (np.array(importances)/np.sum(importances)).tolist()
    
                    weights, train_losses, train_target_losses, val_target_losses, _ = self.train(Ys, Ms, Y_val, M_val, d, norm_importances, outputs, lr, n_epochs, eps, n_llms, random_seed)
                    
                    if np.min(val_target_losses)<=best_loss:
                        best_loss = np.min(val_target_losses)
                        best_epoch = np.argmin(val_target_losses)
                        self.best_params = {'d':d, 'importances':importances, 'n_epochs':best_epoch}
                        
                    if verbose:
                        tqdm.write(f"d={d}, train loss={train_losses[-1]:.5f}, train target loss={train_target_losses[-2]:.5f}, val target loss={np.min(val_target_losses):.5f}, best val target loss={best_loss:.5f}")
    
            #Retrieving the best model
            norm_importances = (np.array(self.best_params['importances'])/np.mean(self.best_params['importances'])).tolist()
            n_epochs = self.best_params['n_epochs']
            d = self.best_params['d']
            del(Ys); del(Ms)
            Ys = [torch.tensor(Y, requires_grad=False).double().to(self.device) for Y in Ys_full]
            Ms = [torch.tensor(M, requires_grad=False).double().to(self.device) for M in Ms_full]

            if verbose:
                print(self.best_params)

        self.weights, _, _, _, _ = self.train(Ys, Ms, None, None, d, norm_importances, outputs, lr, n_epochs, eps, n_llms, random_seed, val_bool=False)
        self.weights['Theta'] = self.weights['Theta'].detach().cpu().numpy()
        for k in range(len(self.weights.keys())-1):
            for k2 in self.weights[k].keys():
                self.weights[k][k2] = self.weights[k][k2].detach().cpu().numpy()
                
class GlobalModel_mask(nn.Module):
    def __init__(self, ds, device='cuda'):
        super(GlobalModel, self).__init__()
        self.ds = ds
        self.device = device
        self.sigmoid = nn.Sigmoid()  

    def check_inputs(self, Ys, Ms, models):
        n_llms = Ms[0].shape[1]
        assert len(Ys)==len(Ms)
        assert len(Ys)==len(models)
        assert np.mean([M.shape[1]==n_llms for M in Ms])==1
        
    def forward_irt(self, M_irt, Theta, Alpha, beta):
        logits = (M_irt@Theta@Alpha.T)-beta.T
        return self.sigmoid(logits)
        
    def forward_bt(self, M_bt, Theta, gamma, eta):
        l_1 = (M_bt@Theta@gamma)
        p_1 = self.sigmoid(l_1 - eta.exp()).reshape(-1,1)
        p_2 = self.sigmoid(-l_1 - eta.exp()).reshape(-1,1)
        p_tie = (1 - p_1 - p_2).reshape(-1,1)
        return torch.cat([p_1, p_2, p_tie], dim=1)

    def irt_loss(self, Y, M_irt, Theta, Alpha, beta, eps, mask_irt=None):
        p = self.forward_irt(M_irt, Theta, Alpha, beta)
        if not torch.is_tensor(mask_irt): #if mask is not defined, we use the full data to compute the loss
            mask_irt = torch.ones(p.shape, dtype=torch.bool)
        return -(Y*(p+eps).log() + (1-Y)*(1-p+eps).log())[mask_irt].mean()
                
    def bt_loss(self, Y, M_bt, Theta, gamma, eta, eps, mask_bt=None):
        p = self.forward_bt(M_bt, Theta, gamma, eta)
        if not torch.is_tensor(mask_bt): #if mask is not defined, we use the full data to compute the loss
            mask_bt = torch.ones(p.sum(1).shape, dtype=torch.bool)
        return -(Y*(p+eps).log()).sum(1)[mask_bt].mean()

    def create_theta(self, random_seed, n_llms, d, device):
        torch.manual_seed(random_seed)
        Theta = torch.nn.Parameter(torch.normal(0, 1, (n_llms, d), dtype=torch.float64, device=device))
        #Theta[-1].requires_grad = False #Freezing the last layer
        #Theta[-1] = 0
        #Theta = torch.nn.Parameter(torch.zeros((n_llms, d), dtype=torch.float64, device=device))
        return Theta
    
    def create_irt_weights(self, random_seed, n_examples, d, device):
        torch.manual_seed(random_seed)
        Alpha = torch.nn.Parameter(torch.normal(0, 1, (n_examples, d), dtype=torch.float64, device=device))
        beta = torch.nn.Parameter(torch.normal(0, 1, (n_examples, 1), dtype=torch.float64, device=device))
        return {'Alpha':Alpha, 'beta':beta}
        
    def create_bt_weights(self, random_seed, d, device):
        torch.manual_seed(random_seed)
        gamma = torch.nn.Parameter(torch.normal(0, 1, (d,1), dtype=torch.float64, device=device))
        #gamma = torch.zeros((d,1), requires_grad=False).double().to(device)
        #gamma[0] = 1 #need to adapt this for the general case
        eta = torch.nn.Parameter(torch.normal(0, 1, (1,), dtype=torch.float64, device=device)) #(1/d)*
        return {'gamma':gamma, 'eta':eta}

    def create_irt_train_mask(self, Y, validation_fraction, random_seed):
        mask = torch.ones(Y.shape, dtype=torch.bool).reshape(-1)
        local_state = np.random.RandomState(random_seed)
        mask[local_state.choice(len(mask), int(validation_fraction*len(mask)+1))] = False
        mask = mask.reshape(Y.shape)
        return mask
    
    def create_bt_train_mask(self, Y, validation_fraction, random_seed):
        mask = torch.ones(Y.shape[0], dtype=torch.bool).reshape(-1)
        local_state = np.random.RandomState(random_seed)
        mask[local_state.choice(len(mask), int(validation_fraction*len(mask)+1))] = False
        mask = mask.reshape(Y.shape[0])
        return mask
        
    def fit(self, 
            Ys, 
            Ms, 
            outputs, #Ex: ['irt', 'bt', 'irt', 'irt']
            importances,
            validation_fraction=.1,
            tol=1e-5, lr=1, n_epochs=500, eps = 1e-10, random_seed=42, verbose=True):
        
        assert validation_fraction>0
        self.check_inputs(Ys, Ms, outputs)
        n_llms = Ms[0].shape[1]
        ds = self.ds
        device = self.device
        importances = (np.array(importances)/np.sum(importances)).tolist()
        Ys = [torch.tensor(Y, requires_grad=False).double().to(self.device) for Y in Ys]
        Ms = [torch.tensor(M, requires_grad=False).double().to(self.device) for M in Ms]

        best_loss = math.inf

        ### Creating training masks
        train_masks = []
        for k, out in enumerate(outputs):
            if out == 'irt':
               train_masks.append(self.create_irt_train_mask(Ys[k], validation_fraction, random_seed))
            elif out == 'bt':
               train_masks.append(self.create_bt_train_mask(Ys[k], validation_fraction, random_seed))
                
        ### 
        for d in ds: #tqdm(ds, disable = not verbose):
            
            #Initializing parameters
            weights = {'Theta': self.create_theta(random_seed, n_llms, d, device)}
            for k, out in enumerate(outputs):
                if out == 'irt':
                    n_examples = Ys[k].shape[1]
                    weights[k] = self.create_irt_weights(random_seed, n_examples, d, device)
                elif out == 'bt':
                    weights[k] = self.create_bt_weights(random_seed, d, device)
            self.weights = weights
            #Define optimizer for a specific parameter
            def optimize_parameter(parameter):
                optimizer = optim.LBFGS(parameter, lr=lr, line_search_fn='strong_wolfe')
                def closure():
                    optimizer.zero_grad()
                    loss = 0
                    for k, out in enumerate(outputs):
                        if out == 'irt': 
                            loss += importances[k]*self.irt_loss(Ys[k], Ms[k], weights['Theta'], weights[k]['Alpha'], weights[k]['beta'], eps, mask_irt = train_masks[k])
                        elif out == 'bt': 
                            loss += importances[k]*self.bt_loss(Ys[k], Ms[k], weights['Theta'], weights[k]['gamma'], weights[k]['eta'], eps, mask_bt = train_masks[k])
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                return loss.item()
                
            #Fitting
            train_losses = []
            for epoch in range(n_epochs):

                #Defining params to optimize in each step (coordinate descent)
                parameters = [[weights['Theta']], []]
                for k, out in enumerate(outputs):
                    if out == 'irt': 
                        parameters[-1] += [weights[k]['Alpha'], weights[k]['beta']]
                    elif out == 'bt': 
                        parameters[-1] += [weights[k]['eta'], weights[k]['gamma']] 
                        
                #Optimizing
                for parameter in parameters:
                    loss = optimize_parameter(parameter)
                train_losses.append(loss)

                #Break if the improvement is small
                if epoch>=1:
                    if train_losses[-2]-train_losses[-1]<tol:
                        break
        
            #Computing val loss 
            with torch.no_grad():
                val_loss = 0
                for k, out in enumerate(outputs):
                    if out == 'irt': 
                        val_loss += importances[k]*self.irt_loss(Ys[k], Ms[k], weights['Theta'], weights[k]['Alpha'], weights[k]['beta'], eps, mask_irt = ~train_masks[k])
                    elif out == 'bt':
                        val_loss += importances[k]*self.bt_loss(Ys[k], Ms[k], weights['Theta'], weights[k]['gamma'], weights[k]['eta'], eps, mask_bt = ~train_masks[k])
                val_loss = val_loss.item()
                
            if val_loss<=best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(weights)
            
            if verbose:
                tqdm.write(f"d={d}, train loss={train_losses[-1]:.5f}, val loss={val_loss:.5f}")

        #Retrieving the best model
        best_model['Theta'] = best_model['Theta'].detach().cpu().numpy()
        for k in range(len(best_model.keys())-1):
            for k2 in best_model[k].keys():
                best_model[k][k2] = best_model[k][k2].detach().cpu().numpy()
        self.weights = best_model