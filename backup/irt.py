import pickle
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import copy
import torch
from torch import nn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from methods import *
from utils import *
device = 'cuda'

sigmoid = nn.Sigmoid()

def prepare_data(chosen_scenarios, scenarios, data):
    i = 0
    subscenarios_position = {}
    
    # Iterate through each chosen scenario and its subscenarios to record item positions
    for scenario in chosen_scenarios:
        subscenarios_position[scenario] = {}
        for sub in scenarios[scenario]:
            subscenarios_position[scenario][sub] = []
            for j in range(data['data'][sub]['correctness'].shape[0]):
                subscenarios_position[scenario][sub].append(i)
                i += 1
    
    # Prepare a simplified mapping of scenarios to their item positions
    scenarios_position = {}
    for scenario in chosen_scenarios:
        scenarios_position[scenario] = []
        for key in subscenarios_position[scenario].keys():
            scenarios_position[scenario] += subscenarios_position[scenario][key]
    return scenarios_position, subscenarios_position

def create_irt_train_mask(Y, validation_fraction, random_seed):
    mask = torch.ones(Y.shape, dtype=torch.bool).reshape(-1)
    local_state = np.random.RandomState(random_seed)
    mask[local_state.choice(len(mask), int(validation_fraction*len(mask)+1))] = False
    mask = mask.reshape(Y.shape)
    return mask

def pirt(Y, design, n_examples, Theta, Alpha, beta):
    budget = len(design)
    unseen = [i for i in range(n_examples) if i not in design]
    data_part = Y[:,design].mean(1)
    model_part = irt_forward(Theta, Alpha, beta)[:,unseen].mean(1)
    return (budget/n_examples)*data_part + (1-budget/n_examples)*model_part

def forward(Theta, Alpha, beta):
    return sigmoid(Theta@Alpha.T-beta.T)

def loss_matrix(Y, P, eps=1e-5):
    return -(Y*(P+eps).log() + (1-Y)*(1-P+eps).log())
        
def fit_IRT(Y,
            d,
            logAlpha=None,
            beta=None,
            Theta=None,
            lr=1,
            n_epochs=100,
            validation_fraction=.2,
            tol=1e-3,
            random_seed=42,
            verbose=False,
            device='cuda'):

    ### Basic defs
    if validation_fraction>0:
        train_mask = create_irt_train_mask(Y, validation_fraction, random_seed)
        val_mask = ~train_mask
    Y = torch.tensor(Y, requires_grad=False).double().to(device)
    n_llms = Y.shape[0]

    ### Defining training variables
    parameters = []
    torch.manual_seed(random_seed)

    #beta
    if beta is None:
        beta = torch.nn.Parameter(torch.normal(0, 1/(d**.5), size=(Y.shape[1],1,), dtype=torch.float64, device=device))
        parameters.append([beta])
    else:
        beta = torch.tensor(beta, requires_grad=False).double().to(device)

    #logAlpha
    if logAlpha is None:
        logAlpha = torch.nn.Parameter(torch.normal(0, 1/(d**.5), size=(Y.shape[1],d,), dtype=torch.float64, device=device))
        parameters.append([logAlpha])
    else:
        logAlpha = torch.tensor(logAlpha, requires_grad=False).double().to(device)

    #Theta
    if Theta is None:
        Theta = torch.nn.Parameter(torch.normal(0, 1/(d**.5), size=(n_llms,d,), dtype=torch.float64, device=device))
        parameters.append([Theta])
    else:
        Theta = torch.tensor(Theta, requires_grad=False).double().to(device)
    
    ### Defining sub-function for optimization
    def optimize_parameter(parameter):
        optimizer = optim.LBFGS(parameter, lr=lr, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            P = forward(Theta, logAlpha.exp(), beta)
            if validation_fraction>0: loss = loss_matrix(Y, P)[train_mask].mean()
            else: loss = loss_matrix(Y, P).mean()
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        return loss.item()

    ### Running training loop
    train_losses =[]
    val_losses =[]
    val_accs =[]
    
    for epoch in tqdm(range(n_epochs), disable=not verbose):
        
        for parameter in parameters:
            train_losses.append(optimize_parameter(parameter))
    
        with torch.no_grad():
            P = forward(Theta, logAlpha.exp(), beta)
            
            if validation_fraction>0:
                loss = loss_matrix(Y, P)[val_mask].mean()
                val_losses.append(loss.item())
                val_accs.append((Y==((forward(Theta, logAlpha.exp(), beta))>.5).double()).float()[~train_mask].mean().item())
                if verbose: 
                    tqdm.write(f"epoch={epoch}, d={d}, train loss={train_losses[-1]:.5f}, val loss={val_losses[-1]:.5f}, val acc={val_accs[-1]:.5f}")
                if epoch>=1:
                    if val_losses[-1]-val_losses[-2]>=0 or val_losses[-2]-val_losses[-1]<=tol:
                        break
            else:
                if epoch>=1:
                    if train_losses[-2]-train_losses[-1]<=tol:
                        break
                
    ### Output
    beta = beta.detach().cpu().numpy()
    logAlpha = logAlpha.detach().cpu().numpy()
    Theta = Theta.detach().cpu().numpy()

    if validation_fraction>0:
        val_loss = val_losses[-1]
    else:
        val_loss = None

    return logAlpha, beta, Theta, val_loss

class IRT:
    def __init__(self, ds=[1,3,5,10], device='cuda'):
        self.ds = ds
        self.device = device
 
    def fit(self, Y, lr=1, n_epochs=100, validation_fraction=.1, tol=1e-3, random_seed=42, verbose=True):

        self.best_loss = math.inf
        
        for d in tqdm(self.ds, disable=not verbose):
            logAlpha, beta, Theta, val_loss = fit_IRT(Y,
                                                      d,
                                                      logAlpha=None,
                                                      beta=None,
                                                      Theta=None,
                                                      lr=lr,
                                                      n_epochs=n_epochs,
                                                      validation_fraction=validation_fraction,
                                                      tol=tol,
                                                      random_seed=random_seed,
                                                      verbose=verbose,
                                                      device=self.device)

            if val_loss < self.best_loss:
                self.d = d
                self.beta = beta
                self.logAlpha = logAlpha
                self.Theta = Theta
                self.best_loss = val_loss

    def get_params(self):
        return self.logAlpha, self.beta, self.Theta

    def fit_theta(self, Y, selected_items, lr=1, n_epochs=100, tol=1e-4, random_seed = 42):

        #Y = torch.tensor(Y, requires_grad=False).double().to(self.device)
        
        _, _, Theta_test, _ = fit_IRT(Y,
                                     self.d,
                                     logAlpha=self.logAlpha[selected_items],
                                     beta=self.beta[selected_items],
                                     Theta=None,
                                     lr=lr,
                                     n_epochs=n_epochs,
                                     validation_fraction=0,
                                     tol=tol,
                                     random_seed=random_seed,
                                     device=self.device)

        return Theta_test

    def fit_logalpha_beta(self, Y, selected_test_takers, lr=1, n_epochs=100, tol=1e-4, random_seed = 42):

        #Y = torch.tensor(Y, requires_grad=False).double().to(self.device)
        
        logAlpha, beta, _, _ = fit_IRT(Y,
                                      self.d,
                                      logAlpha=None,
                                      beta=None,
                                      Theta=self.Theta[selected_test_takers],
                                      lr=lr,
                                      n_epochs=n_epochs,
                                      validation_fraction=0,
                                      tol=tol,
                                      random_seed=random_seed,
                                      device=self.device)

        return logAlpha, beta
        

def optimize_information(I, budget = 100, iterations = 100000, random_seed = 42, verbose = True):

    def loss(ind, I, norm=1):
        return -np.linalg.det(I[:,ind,:,:].sum(1)).mean()/norm
        
    def temp(t,cte=1):
        return(cte/t)
        
    local_state = np.random.default_rng(random_seed)
    best_loss = 999
    n_examples = I.shape[1]
    initial_cand = [local_state.choice(n_examples, budget, replace=False).tolist() for _ in tqdm(range(10), disable = not verbose)] # we do this to get a normalizer
    initial_losses = [loss(ind, I) for ind in tqdm(initial_cand, disable = not verbose)]
    current = initial_cand[np.argmin(initial_losses)]
    norm = np.abs(np.min(initial_losses))
    losses = []

    current_loss = loss(current, I, norm)
    
    for t in tqdm(range(iterations), disable = not verbose):
        gamma=-1/temp(t+1)
    
        #candidate
        coord = local_state.choice(budget)
        while True:
            x = local_state.choice(n_examples)
            if x not in current:
                break
        cand = current.copy()
        cand[coord] = x
    
        #accept/reject
        cand_loss = loss(cand, I, norm)
        A=min(1,np.exp(np.clip(gamma*(cand_loss-current_loss), -50, 50)))
        u=local_state.uniform()
    
        if u<=A:
            current_loss = cand_loss
            current = cand

        if current_loss <= best_loss:
            best_loss = current_loss
            best_design = current.copy()
            
        losses.append(current_loss)

    return np.sort(best_design)

def anchor_points(Alpha, beta, budget = 100, random_seed = 42):
    trials = 5
    E = np.hstack((Alpha, beta))
    kmeans_models = [KMeans(n_clusters=budget, random_state=1000*t+random_seed, n_init="auto").fit(E) for t in range(trials)]
    kmeans = kmeans_models[np.argmin([m.inertia_ for m in kmeans_models])]
    anchor_points = pairwise_distances(kmeans.cluster_centers_, E, metric='euclidean').argmin(axis=1)
    return np.sort(anchor_points)