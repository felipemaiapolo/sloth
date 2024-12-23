import numpy as np
import copy
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=.01)
softplus = nn.Softplus()
softmax = nn.Softmax(dim=0)
sigmoid = nn.Sigmoid()

def logit(x,eps=1e-5):
    return torch.log((x+eps)/(1-x+eps))
    
def forward1(X, D, W1_X, W1_D):
    return X@W1_X + D@W1_D

def forward2(Z1, W2, a2, b2):
    return a2*(Z1@W2) + b2 

class IncreasingNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super(IncreasingNN, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act = nn.Tanh() #nn.Softplus()

        # Initialize weights to be non-negative
        self.fc1.weight.data = torch.abs(self.fc1.weight.data)
        self.fc2.weight.data = torch.abs(self.fc2.weight.data)
        self.fc3.weight.data = torch.abs(self.fc3.weight.data)
        
    def forward(self, x):
        x = self.act(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return sigmoid(x)

    def enforce_non_negative_weights(self):
        with torch.no_grad():
            self.fc1.weight.data.clamp_(0)
            self.fc2.weight.data.clamp_(0)
            self.fc3.weight.data.clamp_(0)
            #self.fc1.weight.data = nn.Softplus(beta=100)(self.fc1.weight.data) #.clamp_(0)
            #self.fc2.weight.data = nn.Softplus(beta=100)(self.fc2.weight.data) #.clamp_(0)
            #self.fc3.weight.data = nn.Softplus(beta=100)(self.fc3.weight.data) #.clamp_(0)
            
def fitFR(X,
          D,
          Y,
          d,
          C0=None,
          W1_X=None,
          W1_D=None,
          W2=None,
          a2=None,
          b2=None,
          fit_C=True,
          train_link=False,
          positive_w=False,
          weight_decay=1e-20,
          lr = .1,
          n_epochs = 5000,
          scale = .1,
          scheduler_factor = .95,
          scheduler_patience = 5,
          earlystop_patience = 200,
          earlystop_tol = 1e-5,
          random_seed = 42, 
          verbose = True,
          device = 'cpu'):

    dim_X = X.shape[1]
    dim_D = D.shape[1]
    dim_Y = Y.shape[1]

    # initializing weights
    torch.manual_seed(random_seed)
    if W1_X is None: W1_X = nn.Parameter(torch.abs(torch.normal(0, scale , size=(dim_X, d), device=device)))
    else: W1_X = nn.Parameter(torch.tensor(W1_X)).to(device)
    if W1_D is None: W1_D = nn.Parameter(torch.normal(0, scale , size=(dim_D, d), device=device))
    else: W1_D = nn.Parameter(torch.tensor(W1_D)).to(device)
    if W2 is None: W2 = nn.Parameter(torch.abs(torch.normal(0, scale , size=(d, dim_Y), device=device)))
    else: W2 = nn.Parameter(torch.tensor(W2)).to(device)
    if a2 is None: a2 = nn.Parameter(torch.abs(torch.normal(0, scale , size=(1, dim_Y), device=device)))
    else: a2 = nn.Parameter(torch.tensor(a2)).to(device)
    if b2 is None: b2 = nn.Parameter(torch.normal(0, scale , size=(1, dim_Y), device=device))
    else: b2 = nn.Parameter(torch.tensor(b2)).to(device)
        
    vars_optim = []
    if train_link:
        links = [IncreasingNN() for _ in range(Y.shape[1])]
        for link in links:
            vars_optim += list(link.parameters())
    else:
        links = [sigmoid for _ in range(Y.shape[1])]
        
    if C0 is None:
        logit_c = nn.Parameter(torch.normal(0, scale , size=(1, dim_Y), device=device))
        vars_optim += [W1_X, W1_D, W2, a2, b2, logit_c]
    else:
        logit_c = nn.Parameter(logit(C0.clone().detach()).reshape((1, dim_Y)))
        if fit_C:
            vars_optim += [W1_X, W1_D, W2, a2, b2, logit_c]
        else:
            vars_optim += [W1_X, W1_D, W2, a2, b2]

    # optimizer
    optimizer = torch.optim.Adam(vars_optim, weight_decay=weight_decay, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)

    # fitting model
    mse = []
    mae = []
    lrs = []
    best_loss = math.inf
    earlystop_count = 0

    for step in tqdm(range(n_epochs), disable = not verbose):
        optimizer.zero_grad() 

        if positive_w:
            Z1 = forward1(X, D, softplus(W1_X), W1_D)
        else:
            Z1 = forward1(X, D, W1_X, W1_D)
        Z2 = forward2(Z1, softmax(W2), softplus(a2), b2)
        Z3 = torch.hstack([link(Z2[:,j].reshape(-1,1)) for j,link in enumerate(links)])
        Y_hat =  sigmoid(logit_c) + (1-sigmoid(logit_c))*Z3 
        
        loss = criterion(Y_hat,Y)
        mse.append(loss.item())
        mae.append(((Y_hat-Y).abs()).mean().item())
        lrs.append(scheduler.optimizer.param_groups[0]['lr'])
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        if train_link:
            for link in links:
                link.enforce_non_negative_weights()
            
        if best_loss > mae[-1] + earlystop_tol:
            if positive_w:
                W1_X2 = softplus(W1_X).detach().clone()
            else:
                W1_X2 = W1_X.detach().clone()
            best_W1_X, best_W1_D, best_W2, best_a2, best_b2, best_logit_c, best_Y_hat, best_Z1 = W1_X2, W1_D.detach().clone(), softmax(W2.detach().clone()), softplus(a2).detach().clone(), b2.detach().clone(), logit_c.detach().clone(), Y_hat.detach().clone(), Z1.detach().clone()
            
            best_links = copy.deepcopy(links)
            if train_link:
                for link in best_links:
                    for param in link.parameters():
                        param.requires_grad = False
                        
            best_loss = mae[-1]
            earlystop_count = 0
        else:
            earlystop_count += 1
            if earlystop_count >= earlystop_patience:
                break
                
    # output
    return best_W1_X, best_W1_D, best_W2, best_a2, best_b2, sigmoid(best_logit_c), best_links, best_Y_hat, best_Z1, best_loss, mse, mae, lrs

class FactorRegression:
    def __init__(self, d=1):
        self.d = d

    def fit(self, 
            X,
            D,
            Y,
            C0=None,
            W1_X0=None,
            W1_D0=None,
            W20=None,
            a20=None,
            b20=None,
            fit_C=True,
            train_link=False,
            positive_w=False,
            weight_decay=0,
            lrs = np.logspace(-1,-3,5),
            n_epochs = 100000,
            scale = .01,
            scheduler_factor = .95,
            scheduler_patience = 20,
            earlystop_patience = 500,
            earlystop_tol = 1e-5,
            random_seed = 42, 
            verbose = True,
            device = 'cpu'):

        assert Y.shape[1]>=self.d
        self.device = device
        
        X = torch.tensor(X).float().to(device)
        D = torch.tensor(D).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        if C0 is not None:
            assert C0.shape[1]>=self.d
            assert C0.shape[1]==Y.shape[1]
            if isinstance(C0, np.ndarray):
                C0 = torch.tensor(C0).float().to(device)
            else:
                C0 = torch.tensor(C0.clone().detach().cpu().numpy()).float().to(device)

        self.best_loss = math.inf

        for lr in lrs:
            W1_X, W1_D, W2, a2, b2, C, links, Y_hat, Z1, best_loss_fit, mse, mae, lrs = fitFR(X,
                                                                                              D,
                                                                                              Y,
                                                                                              self.d,
                                                                                              C0,
                                                                                              W1_X0,
                                                                                              W1_D0,
                                                                                              W20,
                                                                                              a20,
                                                                                              b20,
                                                                                              fit_C = fit_C,
                                                                                              train_link=train_link,
                                                                                              positive_w=positive_w,
                                                                                              weight_decay=weight_decay,
                                                                                              lr = lr,
                                                                                              n_epochs = n_epochs,
                                                                                              scale = scale,
                                                                                              scheduler_factor = scheduler_factor,
                                                                                              scheduler_patience = scheduler_patience,
                                                                                              earlystop_patience = earlystop_patience,
                                                                                              earlystop_tol = earlystop_tol,
                                                                                              random_seed =random_seed, 
                                                                                              verbose = verbose,
                                                                                              device = device)

            if verbose: print(lr,best_loss_fit)
            if best_loss_fit<self.best_loss:
                self.best_loss = best_loss_fit
                self.W1_X, self.W1_D, self.W2, self.a2, self.b2, self.C, self.links, self.Y_hat, self.Z1, self.mse, self.mae, self.lrs = W1_X, W1_D, W2, a2, b2, C, links, Y_hat, Z1, mse, mae, lrs

        
    def predict(self,
                X,
                D):

        device = self.device #pass links to cpu and change this
        X = torch.tensor(X).float().to(device)
        D = torch.tensor(D).float().to(device)

        Z1 = forward1(X, D, self.W1_X, self.W1_D)
        Z2 = forward2(Z1, self.W2, self.a2, self.b2)
        Z3 = torch.hstack([link(Z2[:,j].reshape(-1,1)) for j,link in enumerate(self.links)])  
        Y_hat =  self.C + (1-self.C)*Z3
        
        return Y_hat.cpu().numpy()