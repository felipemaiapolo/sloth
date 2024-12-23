import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm.auto import tqdm
from sklearn.linear_model import RidgeCV
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch.optim as optim
from joblib import Parallel, delayed
from utils import *

device='cpu'
data_path = '/llmthonskdir/felipe/download_openllmlb/'
h = 10
d = 3

def kernel(x, h):
    return torch.exp(-(x/h)**2)
    
def logit(x, c=0, eps=.001):
    x2 = (np.array(x)-c)/(1-c)
    return np.log((x2+eps)/(1-x2+eps))

def sigmoid(z, use_torch=True):
    if use_torch:
        return torch.nn.Sigmoid()(z)
    else:
        return 1/(1+np.exp(-z))
        
def prep_data(data, benchs_names):
    fam_encoder = LabelEncoder()
    fam_encoder.fit(data['Model Family'])
    data['T'] = data['Pretraining Data Size (T)']
    data['S'] = data['Model Size (B)']
    data['F'] = data['FLOPs (1E21)']
    data['family'] = data['Model Family']
    data = data.sort_values(by=['family','S']).reset_index(drop=True)
    data['logT'] = np.log(data['T'])
    data['logS'] = np.log(data['S'])
    data['logF'] = np.log(data['F'])
    data['logS*logT'] = data['logS']*data['logT']
    data['logS^2'] = data['logS']**2
    data['logS^2*logT'] = data['logS^2']*data['logT']
    data = data[['family','logT','logS','logF','logS*logT', 'logS^2', 'logS^2*logT'] + benchs_names] 
    data = data.dropna(how='any')
    unique_families, counts_families = np.unique(data.family, return_counts=True)
    avail_families = unique_families[counts_families>=2]
    return data, unique_families, avail_families

def pred_data2(data, test_family, benchs_names):
    data_train = data.loc[data.family != test_family]
    data_test = data.loc[data.family == test_family]
    data_train = pd.concat((data_test.iloc[:1],data_train), axis=0).reset_index(drop=True)
    data_test = data_test.iloc[1:].reset_index(drop=True)
    Y_train = torch.tensor(np.array(data_train.loc[:,benchs_names]))
    X_train = np.array(data_train.loc[:,['logT','logS','logS*logT']]) #
    X_train = torch.tensor(X_train).double()
    X2_train = np.array(data_train.loc[:,['logS','logS*logT','logS^2', 'logS^2*logT']]) #
    X2_train = torch.tensor(X_train).double()
    F_train = np.array(data_train.loc[:,['logF']]) #
    F_train = torch.tensor(F_train).double()
    Y_test = torch.tensor(np.array(data_test.loc[:,benchs_names]))
    X_test = np.array(data_test.loc[:,['logS','logS*logT','logS^2', 'logS^2*logT']])
    X_test = torch.tensor(X_test).double()
    F_test = np.array(data_test.loc[:,['logF']]) #
    F_test = torch.tensor(F_test).double()
    D_train = torch.tensor(np.array(pd.get_dummies(np.array(data_train.family)))).double()
    D_test = torch.tensor(np.vstack([D_train[0,:].numpy() for _ in range(Y_test.shape[0])])).double()
    return X_train, F_train, D_train, Y_train, X_test, F_test, D_test, Y_test

def train_pca_regression_model(X_train, D_train, Y_train, X_test, D_test, Y_test, d):

    scaler = StandardScaler().fit(Y_train)
    pca = PCA(n_components=d)
    pca.fit(scaler.transform(Y_train))
    reg = LinearRegression().fit(np.hstack((X_train.numpy(), D_train.numpy())),
                                 pca.transform(scaler.transform(Y_train)))
    Y_hat = scaler.inverse_transform(pca.inverse_transform(reg.predict(np.hstack((X_test.numpy(), D_test.numpy())))))
    return np.abs(Y_test-Y_hat)

def compute_Y_hat_train(j, X_train, D_train, Y_train, beta, thetas, gamma, kernel, h, device='cpu'):
    # Calculate Z
    Z = ((X_train @ torch.clamp(beta, min=0)) + (D_train @ thetas)) @ torch.clamp(gamma[:, j:j+1], min=0)
    
    # Calculate the kernel matrix K
    K = kernel(Z - Z.T, h)
    
    # Create the mask and set diagonal to zero
    mask = torch.ones(K.shape, device=device)
    for i in range(mask.shape[0]):
        mask[i, i] = 0
    
    # Calculate the numerator (frac1) and denominator (frac2)
    frac1 = (mask * K * (Y_train[:, j:j+1] @ torch.ones((1, Y_train.shape[0]), dtype=torch.float64, device=device)).T).sum(1)
    frac2 = (mask * K).sum(1)
    
    # Compute Y_hat
    Y_hat = frac1 / frac2
    
    return Y_hat, Z
    
def train_single_index_model(X_train, D_train, Y_train, X_test, D_test, Y_test, benchs_names, kernel, h, d, num_epochs=10000, patience = 500, tol=1e-3, device='cpu'):
    # Initialize parameters
    thetas = torch.nn.Parameter(torch.normal(0, 1, size=(D_train.shape[1], d), dtype=torch.float64, device=device))
    beta = torch.nn.Parameter(torch.abs(torch.normal(0, 1, size=(X_train.shape[1], d), dtype=torch.float64, device=device)))
    gamma = torch.nn.Parameter(torch.abs(torch.normal(0, 1, size=(d, len(benchs_names)), dtype=torch.float64, device=device)))

    # Initialize the Adam optimizer
    optimizer = optim.Adam([thetas, beta, gamma], lr=0.1, weight_decay=1e-10)
    
    # Lists to track the loss and MAE at each epoch
    losses = []
    maes = []

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero the gradients before the backward pass

        # Compute the loss and MAE
        loss = 0
        mae = 0
        for j, _ in enumerate(benchs_names): 
            # Forward pass
            #Z = ((X_train @ torch.clamp(beta, min=0)) + (D_train @ thetas)) @ torch.clamp(gamma[:, j:j+1], min=0)
            #K = kernel(Z - Z.T, h)
            #mask = torch.ones(K.shape, device=device)
            #for i in range(mask.shape[0]):
            #    mask[i, i] = 0
    
            #frac1 = (mask * K * (Y_train[:, j:j+1] @ torch.ones((1, Y_train.shape[0]), dtype=torch.float64, device=device)).T).sum(1)
            #frac2 = (mask * K).sum(1)
        
            #Y_hat = frac1 / frac2

            Y_hat, _ = compute_Y_hat_train(j, X_train, D_train, Y_train, beta, thetas, gamma, kernel, h, device=device)
            
            loss += (Y_train[:, j:j+1].squeeze() - Y_hat).pow(2).mean()
            mae += torch.abs(Y_train[:, j:j+1].squeeze() - Y_hat).mean()
            
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Track the loss and MAE
        if np.isnan(loss.item()):
            raise ValueError("NaN encountered in loss")
        epoch_loss = loss.item() / len(benchs_names)
        losses.append(epoch_loss)
        epoch_mae = mae.item() / len(benchs_names)
        maes.append(epoch_mae)  # Average MAE across all elements

        # Check if this is the best loss we've seen so far
        if epoch_mae + tol < best_loss:
            best_loss = epoch_mae
            best_epoch = epoch
            best_weights = {
                'thetas': thetas.detach(),
                'beta': beta.detach(),
                'gamma': gamma.detach()
            }

        # Early stopping check
        if epoch - best_epoch > patience:
            print(f"Early stopping at epoch {epoch + 1}, best mae={best_loss}")
            break
        
        # Print the loss and MAE for the current epoch
        #print(f'h={h}, Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, MAE: {maes[-1]}')

    # Calculate errors on the test set using the best model
    thetas = best_weights['thetas']
    beta = best_weights['beta']
    gamma = best_weights['gamma']

    Y_hats = []
    Y_tests = []
    for j, _ in enumerate(benchs_names): 
        Z_train = ((X_train @ torch.clamp(beta, min=0)) + (D_train @ thetas)) @ torch.clamp(gamma[:, j:j+1], min=0)
        Z_test = ((X_test @ torch.clamp(beta, min=0)) + (D_test @ thetas)) @ torch.clamp(gamma[:, j:j+1], min=0)
        K = kernel(Z_test - Z_train.T, h)
        Y_hats.append((K * Y_train[:, j:j+1].T).sum(1) / K.sum(1))
        Y_tests.append(Y_test[:, j])
    
    return torch.abs(torch.vstack(Y_tests) - torch.vstack(Y_hats))

def train_logistic_model(X_train, D_train, Y_train, X_test, D_test, Y_test, benchs_names, d, num_epochs=10000, patience = 500, tol=1e-3, device='cpu'):
    # Initialize parameters
    thetas = torch.nn.Parameter(torch.normal(0, 1, size=(D_train.shape[1], d), dtype=torch.float64, device=device))
    beta = torch.nn.Parameter(torch.abs(torch.normal(0, 1, size=(X_train.shape[1], d), dtype=torch.float64, device=device)))
    gamma = torch.nn.Parameter(torch.abs(torch.normal(0, 1, size=(d, len(benchs_names)), dtype=torch.float64, device=device)))
    w1 = torch.nn.Parameter(torch.abs(torch.normal(0, 1, size=(len(benchs_names), 1), dtype=torch.float64, device=device)))
    w2 = torch.nn.Parameter(torch.normal(0, 1, size=(len(benchs_names), 1), dtype=torch.float64, device=device))
    cs = torch.nn.Parameter(0*torch.normal(0, 1, size=(len(benchs_names), 1), dtype=torch.float64, device=device))
    
    # Initialize the Adam optimizer
    optimizer = optim.Adam([thetas, beta, gamma, w1, w2, cs], lr=0.1, weight_decay=1e-10)
    
    # Lists to track the loss and MAE at each epoch
    losses = []
    maes = []

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero the gradients before the backward pass

        # Compute the loss and MAE
        loss = 0
        mae = 0
        for j, _ in enumerate(benchs_names): 
            # Forward pass
            Z = ((X_train @ torch.clamp(beta, min=0)) + (D_train @ thetas)) @ torch.clamp(gamma[:, j:j+1], min=0)
            Y_hat = sigmoid(cs[j]) + (1-sigmoid(cs[j]))*sigmoid((Z*torch.clamp(w1.T, min=0) + w2.T)[:, j:j+1].squeeze())
        
            loss += (Y_train[:, j:j+1].squeeze() - Y_hat).pow(2).mean()
            mae += torch.abs(Y_train[:, j:j+1].squeeze() - Y_hat).mean()
            
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Track the loss and MAE
        if np.isnan(loss.item()):
            raise ValueError("NaN encountered in loss")
        epoch_loss = loss.item() / len(benchs_names)
        losses.append(epoch_loss)
        epoch_mae = mae.item() / len(benchs_names)
        maes.append(epoch_mae)  # Average MAE across all elements

        # Check if this is the best loss we've seen so far
        if epoch_mae + tol < best_loss:
            best_loss = epoch_mae
            best_epoch = epoch
            best_weights = {
                'thetas': thetas.detach(),
                'beta': beta.detach(),
                'gamma': gamma.detach(),
                'w1': w1.detach(),
                'w2': w2.detach(),
                'cs': cs.detach()
            }

        # Early stopping check
        if epoch - best_epoch > patience:
            print(f"Early stopping at epoch {epoch + 1}, best mae={best_loss}")
            break
        
        # Print the loss and MAE for the current epoch
        #print(f'h={h}, Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, MAE: {maes[-1]}')

    # Calculate errors on the test set using the best model
    thetas = best_weights['thetas']
    beta = best_weights['beta']
    gamma = best_weights['gamma']
    w1 = best_weights['w1']
    w2 = best_weights['w2']
    cs = best_weights['cs']

    Y_hats = []
    Y_tests = []
    for j, _ in enumerate(benchs_names): 
        Z_test = ((X_test @ torch.clamp(beta, min=0)) + (D_test @ thetas)) @ torch.clamp(gamma[:, j:j+1], min=0)
        Y_hats.append(sigmoid(cs[j]) + (1-sigmoid(cs[j]))*sigmoid((Z_test*torch.clamp(w1.T, min=0) + w2.T)[:, j:j+1].squeeze()))
        Y_tests.append(Y_test[:, j])
    
    return torch.abs(torch.vstack(Y_tests) - torch.vstack(Y_hats))

def train_basic_logistic_model(X_train, Y_train, X_test, Y_test, benchs_names, num_epochs=10000, patience = 500, tol=1e-3, device='cpu'):
    # Initialize parameters
    beta = torch.nn.Parameter(torch.abs(torch.normal(0, 1, size=(X_train.shape[1], len(benchs_names)), dtype=torch.float64, device=device)))
    alpha = torch.nn.Parameter(torch.abs(torch.normal(0, 1, size=(len(benchs_names), 1), dtype=torch.float64, device=device)))
    cs = torch.nn.Parameter(0*torch.normal(0, 1, size=(len(benchs_names), 1), dtype=torch.float64, device=device))
    
    # Initialize the Adam optimizer
    optimizer = optim.Adam([beta, alpha, cs], lr=0.1, weight_decay=1e-10)
    
    # Lists to track the loss and MAE at each epoch
    losses = []
    maes = []

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero the gradients before the backward pass

        # Compute the loss and MAE
        loss = 0
        mae = 0
        for j, _ in enumerate(benchs_names): 
            # Forward pass
            Y_hat = sigmoid(cs[j]) + (1-sigmoid(cs[j]))*sigmoid(X_train @ beta + alpha.T)[:, j:j+1].squeeze()
            loss += (Y_train[:, j:j+1].squeeze() - Y_hat).pow(2).mean()
            mae += torch.abs(Y_train[:, j:j+1].squeeze() - Y_hat).mean()
            
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Track the loss and MAE
        if np.isnan(loss.item()):
            raise ValueError("NaN encountered in loss")
        epoch_loss = loss.item() / len(benchs_names)
        losses.append(epoch_loss)
        epoch_mae = mae.item() / len(benchs_names)
        maes.append(epoch_mae)  # Average MAE across all elements

        # Check if this is the best loss we've seen so far
        if epoch_mae + tol < best_loss:
            best_loss = epoch_mae
            best_epoch = epoch
            best_weights = {
                'beta': beta.detach(),
                'alpha': alpha.detach(),
                'cs': cs.detach()
            }

        # Early stopping check
        if epoch - best_epoch > patience:
            print(f"Early stopping at epoch {epoch + 1}, best mae={best_loss}")
            break
        
        # Print the loss and MAE for the current epoch
        #print(f'h={h}, Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, MAE: {maes[-1]}')

    # Calculate errors on the test set using the best model
    beta = best_weights['beta']
    alpha = best_weights['alpha']
    cs = best_weights['cs']

    Y_hats = []
    Y_tests = []
    for j, _ in enumerate(benchs_names): 
        Y_hats.append(sigmoid(cs[j]) + (1-sigmoid(cs[j]))*sigmoid(X_test @ beta + alpha.T)[:, j:j+1].squeeze())
        Y_tests.append(Y_test[:, j])

    return torch.abs(torch.vstack(Y_tests) - torch.vstack(Y_hats))

def train_basic_intercept_logistic_model(X_train, D_train, Y_train, X_test, D_test, Y_test, benchs_names, num_epochs=10000, patience = 500, tol=1e-3, device='cpu'):
    # Initialize parameters
    thetas = torch.nn.Parameter(torch.normal(0, 1, size=(D_train.shape[1], len(benchs_names)), dtype=torch.float64, device=device))
    beta = torch.nn.Parameter(torch.abs(torch.normal(0, 1, size=(X_train.shape[1], len(benchs_names)), dtype=torch.float64, device=device)))
    cs = torch.nn.Parameter(0*torch.normal(0, 1, size=(len(benchs_names), 1), dtype=torch.float64, device=device))
    
    # Initialize the Adam optimizer
    optimizer = optim.Adam([thetas, beta, cs], lr=0.1, weight_decay=1e-10)
    
    # Lists to track the loss and MAE at each epoch
    losses = []
    maes = []

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero the gradients before the backward pass

        # Compute the loss and MAE
        loss = 0
        mae = 0
        for j, _ in enumerate(benchs_names): 
            # Forward pass
            Y_hat = sigmoid(cs[j]) + (1-sigmoid(cs[j]))*sigmoid(X_train @ beta + (D_train @ thetas))[:, j:j+1].squeeze()
            loss += (Y_train[:, j:j+1].squeeze() - Y_hat).pow(2).mean()
            mae += torch.abs(Y_train[:, j:j+1].squeeze() - Y_hat).mean()
            
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Track the loss and MAE
        if np.isnan(loss.item()):
            raise ValueError("NaN encountered in loss")
        epoch_loss = loss.item() / len(benchs_names)
        losses.append(epoch_loss)
        epoch_mae = mae.item() / len(benchs_names)
        maes.append(epoch_mae)  # Average MAE across all elements

        # Check if this is the best loss we've seen so far
        if epoch_mae + tol < best_loss:
            best_loss = epoch_mae
            best_epoch = epoch
            best_weights = {
                'thetas': thetas.detach(),
                'beta': beta.detach(),
                'cs': cs.detach()
            }

        # Early stopping check
        if epoch - best_epoch > patience:
            print(f"Early stopping at epoch {epoch + 1}, best mae={best_loss}")
            break
        
        # Print the loss and MAE for the current epoch
        #print(f'h={h}, Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, MAE: {maes[-1]}')

    # Calculate errors on the test set using the best model
    thetas = best_weights['thetas']
    beta = best_weights['beta']
    cs = best_weights['cs']

    Y_hats = []
    Y_tests = []
    for j, _ in enumerate(benchs_names): 
        Y_hats.append(sigmoid(cs[j]) + (1-sigmoid(cs[j]))*sigmoid(X_test @ beta + (D_test @ thetas))[:, j:j+1].squeeze())
        Y_tests.append(Y_test[:, j])
    
    return torch.abs(torch.vstack(Y_tests) - torch.vstack(Y_hats))

def run_results(test_family, data, benchs_names, cov='X'):
    X_train, F_train, D_train, Y_train, X_test, F_test, D_test, Y_test = pred_data2(data, test_family, benchs_names)

    if cov=='X':
        x_test = X_test[:,:2]
        x_train = X_train[:,:2]
    elif cov=='X_inter':
        x_test = X_test
        x_train = X_train
    else:
        x_test = F_test
        x_train = F_train
        
    pca_error = train_pca_regression_model(x_train, D_train, Y_train, x_test, D_test, Y_test, d)
    logistic_error = train_logistic_model(x_train, D_train, Y_train, x_test, D_test, Y_test, benchs_names, d)
    logistic_basic_intercept_error = train_basic_intercept_logistic_model(x_train, D_train, Y_train, x_test, D_test, Y_test, benchs_names)
    
    single_index_error = train_single_index_model(x_train, D_train, Y_train, x_test, D_test, Y_test, benchs_names, kernel, h, d)
    
    logistic_basic_error = train_basic_logistic_model(x_train, Y_train, x_test, Y_test, benchs_names)
    
    return pca_error, logistic_error, single_index_error, logistic_basic_error, logistic_basic_intercept_error