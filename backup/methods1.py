import torch
import math
import numpy as np
from tqdm.auto import tqdm

device = 'cpu'

eps = .001
def logit(p, eps=eps):
    return np.log((eps+p)/(1-p+eps))
    
def sigmoid(z, use_torch=False):
    if use_torch:
        return torch.nn.Sigmoid()(z)
    else:
        return 1/(1+np.exp(-z))
        
########
def logistic_forward(coefs, X, guess, fea, use_torch=False):
    return guess+(1-guess)*fea*sigmoid(X@coefs, use_torch)
    
def train_logistic_coefs(coefs, X, y, guess=0, fea=1, epochs=100, tol=1e-5):
    optimizer = torch.optim.LBFGS([coefs], lr=.1, line_search_fn='strong_wolfe')
    losses = []
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            loss = ((logistic_forward(coefs, X, guess, fea, use_torch=True)-y)**2).mean()
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        abs_loss = ((logistic_forward(coefs, X, guess, fea, use_torch=True)-y).abs()).mean()
        losses.append(abs_loss.item())

        if epoch>=1:
            if losses[-2]-losses[-1]<=tol:
                break
        
    return coefs.cpu().detach().numpy(), abs_loss.item(), epoch+2

class Logistic:
    def __init__(self, guesses=np.linspace(0,.3,20), feas=[1]):
        self.guesses = guesses
        self.feas = feas

    def fit(self, X, y):
        X = torch.tensor(X).reshape((X.shape[0],-1)).double().to(device)
        y = torch.tensor(y).reshape((y.shape[0],-1)).double().to(device)
        
        best_loss = math.inf
        best_coefs = None
        best_guess = None
        best_fea = None
        
        for guess in self.guesses:
            for fea in self.feas:
                coefs = torch.nn.Parameter(torch.normal(0, 1, size=(X.shape[1],1), dtype=torch.float64, device=device))
                coefs, loss, _ = train_logistic_coefs(coefs, X, y, guess=guess, fea=fea)
                if loss<=best_loss:
                    best_coefs = coefs
                    best_loss = loss
                    best_guess = guess
                    best_fea = fea

        self.best_coefs=best_coefs
        self.best_guess=best_guess
        self.best_fea=best_fea

    def predict(self,X):
        return logistic_forward(self.best_coefs, X, self.best_guess, self.best_fea).squeeze()

########
def interact_logistic_forward(coefs, interact_slopes, non_intercept_terms, X, guess, fea, use_torch=False):
    pred = X@coefs + (X[:,:non_intercept_terms]*(X[:,non_intercept_terms:]@((coefs[non_intercept_terms:]@interact_slopes.T)))).sum(axis=1)[:,None]
    return guess+(1-guess)*fea*sigmoid(pred, use_torch)
    
def train_interact_logistic_coefs(coefs, interact_slopes, non_intercept_terms, X, y, guess=0, fea=1, epochs=100, tol=1e-5):
    
    params = [[coefs],[interact_slopes]]
    losses = []
    
    for epoch in range(epochs):

        for param in params:
            optimizer = torch.optim.LBFGS(param, lr=1, line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                loss = ((interact_logistic_forward(coefs, interact_slopes, non_intercept_terms, X, guess, fea, use_torch=True)-y)**2).mean()
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            
        abs_loss = ((interact_logistic_forward(coefs, interact_slopes, non_intercept_terms, X, guess, fea, use_torch=True)-y).abs()).mean()
        losses.append(abs_loss.item())
        if epoch>=1:
            if losses[-2]-losses[-1]<=tol:
                break
        
    return coefs.cpu().detach().numpy(), interact_slopes.cpu().detach().numpy(), abs_loss.item(), epoch+2

class InteractLogistic:
    def __init__(self, non_intercept_terms, guesses=np.linspace(0,.3,20), feas=[1]):
        self.non_intercept_terms = non_intercept_terms
        self.guesses = guesses
        self.feas = feas

    def fit(self, X, y):
        X = torch.tensor(X).reshape((X.shape[0],-1)).double().to(device)
        y = torch.tensor(y).reshape((y.shape[0],-1)).double().to(device)
        
        best_loss = math.inf
        best_coefs = None
        best_interact_slopes = None
        best_guess = None
        best_fea = None
        
        for guess in self.guesses:
            for fea in self.feas:
                coefs = torch.nn.Parameter(torch.normal(0, 1, size=(X.shape[1],1), dtype=torch.float64, device=device))
                interact_slopes = torch.nn.Parameter(torch.normal(0, .001, size=(self.non_intercept_terms,1), dtype=torch.float64, device=device))
                

                coefs, interact_slopes, loss, _ = train_interact_logistic_coefs(coefs, interact_slopes, self.non_intercept_terms, X, y, guess=guess, fea=fea)
                if loss<=best_loss:
                    best_coefs = coefs
                    best_interact_slopes = interact_slopes
                    best_loss = loss
                    best_guess = guess
                    best_fea = fea

        self.best_coefs=best_coefs
        self.best_interact_slopes=best_interact_slopes
        self.best_guess=best_guess
        self.best_fea=best_fea

    def predict(self,X):
        return interact_logistic_forward(self.best_coefs, self.best_interact_slopes, self.non_intercept_terms, X, self.best_guess, self.best_fea).squeeze()
       