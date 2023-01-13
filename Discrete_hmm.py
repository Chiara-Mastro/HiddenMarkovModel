#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:12:20 2022

@author: Chiara
"""

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import matplotlib as mlp

def Generating_Data(A, Phi, pi, nT = 100):
    
    # states in the latent variable
    nStates = len(A[:,0])
    # states in the observable variable
    nObs = len(Phi[0,:])
    
    #nT gives me the length of the simulation so we define empty data vectors
    z_data = -1*np.ones(nT, dtype=(int))
    x_data =  -1*np.ones(nT, dtype=(int))
    #x_label for each time step has zeros but one in the row corresponding to the activated obs state
    x_label = np.zeros((nObs, nT))
    
    #Pi tells me the probability of starting in a random point
    z_data[0] = np.random.choice(nStates, p = pi)
    # Phi tells me the probability of observing one of the x obs states from the latent
    x_data[0] = np.random.choice(nObs, p = Phi[int(z_data[0]), :])
     
    #First I simulate the dynamics of the latent variable with transitions given by the probability A
    #Then I simulate the probability Phi of observing a sequence of x 
    for i in range(1, nT):
        z_data[i] = np.random.choice(nStates, p = A[int(z_data[i-1]), :])
        x_data[i] = np.random.choice(nObs, p = Phi[int(z_data[i]), :])
        
    #Easily generating the matrix
    for i in range(nT):
        x_label[int(x_data[i]), i] = 1
    
    
    return x_data, z_data, x_label


def ForwardBackward(A, Phi, pi, x_data):
    
    #inferring the parameters
    nT = x_data.shape[0]
    nStates, nObs = Phi.shape 
    
    #Creating the forward a, backward b and scaling c variables
    a_hat = -1*np.ones((nT, nStates), dtype = np.float128) # forward probabilities p(z_t|x_{1:t})
    b_hat = -1*np.ones((nT, nStates) ,dtype = np.float128) # backward probabilities p(x_{t+1:T}|z_t)/p(x_{t+1:T} | x_{1:T})
    

    ## FORWARD PASS : cumulating a over time
    #Time 0
    #The forward probability of observing z_0 given that we see x_0 is the prob of starting from z_0 ivided by the mapping from z_0 to x_0
    a_hat[0, :] = pi * Phi[:, x_data[0]]
      
    #a_hat evolves through Eq 13.59 in the Bishop
    for t in range(1, nT):
        for j in range(nStates):
            a_hat[t, j] = (a_hat[t-1] @ A[:, j]) * Phi[j, x_data[t]]

    ## BACKWARD PASS : cumulating b over future time
    b_hat[nT-1] = 1 # Certain event if no future is involved!
    
    for t in range(nT - 2, -1, -1):
        for j in range(nStates):
            #b_hat evolves through Eq 13.62 in the Bishop
            #First I accumulate over the elements of z_{t-1} to z_t
            b_hat[t, j] = ( b_hat[t+1] * Phi[:, x_data[t+1]] ) @ A[j]

    return a_hat, b_hat
    
    
    
def ExpectationMaximization(A_hat, Phi_hat, Pi_hat, x_data, Niter):
    
    #Creating the variables to store the evolution of the log likelihood
    dlogP_threshold = 10**(-10) # I will stop the algorithm if the log does not increase of something larger than this threshold
    dlogP_prev = - 10**(10) # Ausilary variable to build the exit condition
    dlogP_diff = 10**(10) # Difference between current and previous step in logP, setting initially to arbitrary large
    log_p = []
    nT = len(x_data)
    nStates, nObs = Phi_hat.shape
    it = 0
    
    while it < Niter and np.absolute(dlogP_diff) > dlogP_threshold:
    #for it in range(Niter):
        # E # 
        ## Expectation step is realized through the forwardbackward function
        a_hat, b_hat = ForwardBackward(A_hat, Phi_hat, Pi_hat, x_data)
        #print(a_hat, b_hat)
        ## COMPUTING gamma and xi as Eq. 13.64 and 13.65 in the Bishop
        #Marginal likelihood over time
        log_likelihood = np.sum(np.multiply(a_hat, b_hat))
        log_p.append(np.log(log_likelihood))
        
        #Creating the variables returned by the ForwBack step
        #gamma_hat = -1*np.ones((nStates, nT)) # Marginal posterior probabilities p(z_t|x_{1:t})
        xi_hat = np.zeros((nStates, nStates, nT - 1))  #AVERAGE over time marginals posteriors among pairs: p(z_{t-1}, z_t | x_{1:t})
        
        
        for t in range(nT - 1):
            # joint probab of observed data up to time t @ transition prob * emisssion prob as t+1 @
            # joint probab of observed data from time t+1
            denominator = ((a_hat[t] @ A_hat) * Phi_hat[:, x_data[t + 1]]) @ b_hat[t + 1]
            for i in range(nStates):
                numerator = a_hat[t, i] * A_hat[i] * Phi_hat[:, x_data[t + 1]] * b_hat[t + 1]
                xi_hat[i, :, t] = numerator / denominator
        # Calculate the gammma table.
        #Posterior in one state as the contribution from forward and backward pass
        gamma_hat = np.sum(xi_hat, axis=1)
        

        # M # 
        # ## In the maximization step we get the best possible parameter for the configuration
        A_hat = np.sum(xi_hat, 2) / np.sum(gamma_hat, axis=1).reshape((-1, 1))
        Pi_hat = gamma_hat[:, 0] / np.sum(gamma_hat[:, 0])   
        #Add additional T'th element in gamma matrix
        gamma_hat = np.hstack((gamma_hat, np.sum(xi_hat[:, :, nT - 2], axis=0).reshape((-1, 1))))
        # Update the emission probabilities
        Phi_hat = np.dot(gamma_hat , x_label.T)
        Phi_hat /= np.sum(Phi_hat, axis = 1).reshape(-1, 1) #and normalizing
        
        # UPDATE #
        dlogP_diff = log_p[it] - dlogP_prev 
        dlogP_prev = log_p[it]
        it += 1
    
    return A_hat, Phi_hat, Pi_hat, log_p


#%% GENERATING THE DATA

## Fixing the parameters of the simulation
nStates = 2 #Number of states for the latent variable
nObs = 3 #Number of states for the observable variable
nT = 1000 #Length of the simulation 
Niter = 10  #Number of iterations used to converge to the optimal set of parameters

### These are the true matrices from which I generate the data I want to recover lateer
### A must be a squared matrix : sum over rows gives me one so row label gives me the starting state
### Phi rows must sum to one so again row label gives me the starting latent state


#Extracting random initial TRUE conditions for the matrices
# A_true = np.random.uniform(size=(nStates, nStates))
# The Dirichlet distribution is a conjugate prior of a multinomial distribution in Bayesian inference.
# a good assumption is to make the diagonal 'stronger'
alpha_diag = 10 #strenght of the parameter alpha along the diagonal 
alpha_out = 50 #strength outside
alpha = alpha_out * np.random.randint(1, alpha_out, size=(nStates, nStates)) + alpha_diag * np.eye(nStates, dtype=int)

A_true = np.empty(alpha.shape)
for i in range (len(alpha)):
    A_true[i] = np.random.dirichlet(alpha[i])
#And normalizing
A_true /= np.sum(A_true, axis=1).reshape(-1, 1)

# Observation matrix mapping the latent z to my observable x, again Dirichlet in this discrete HMM 
# In the continuous gaussian case I will just give my two parameters my muy and sigma
alpha_emission = 25 
alpha_em = alpha_emission * np.random.randint(1, alpha_emission, size=(nStates, nObs))
Phi_true = np.empty(alpha_em.shape)
for i in range (len(alpha_em)):
    Phi_true[i] = np.random.dirichlet(alpha_em[i])
#And normalizinf
Phi_true /= np.sum(Phi_true, axis=1).reshape(-1, 1)

Pi_true = np.ones((nStates, ))
Pi_true /= np.sum(Pi_true)

###And generating the data
x_data, z_data, x_label = Generating_Data(A_true, Phi_true, Pi_true, nT=1000)
# x_data, z_data = gen_data(A_true, Phi_true, nT=10000)

#%% TRAINING THE MODEL

## Fixing the parameters of the simulation
nStates = 2 #Number of states for the latent variable
nObs = 3 #Number of states for the observable variable
nT = 10000 #Length of the simulation 
Niter = 100  #Number of iterations used to converge to the optimal set of parameters


###I start with new random matrices
#Extracting random initial conditions for the matrices
#A_0 = np.random.uniform(size=(nStates, nStates))/np.sum(A_true, axis=0)
A_0 = np.random.uniform(size=(nStates, nStates))
A_0 /= np.sum(A_0, axis=1).reshape(-1, 1)
Phi_0 = np.random.uniform(size=(nStates, nObs))
Phi_0 /= np.sum(Phi_0, axis=1).reshape(-1, 1)
Pi_0 = np.ones((nStates, ))
Pi_0 /= np.sum(Pi_0)

A_hat, Phi_hat, Pi_hat, logTraj = ExpectationMaximization(A_0.copy(), Phi_0.copy(), Pi_0.copy(), x_data.copy(),  Niter)


#%% AND COMPUTE THE DIFFERENCE WITH THE hmm PYTHON PACKAGES

# compore to hmmlearn
model = hmm.MultinomialHMM(n_components = nStates, n_iter = Niter, algorithm = "map",
                           init_params = "", params = "ste",
                           tol = 1e-12, verbose = True)
model.startprob_ = Pi_0
model.transmat_ = A_0
model.emissionprob_ = Phi_0

model.fit([x_data])
print(f'hmmlearn A \n{model.transmat_}')
print(f'hmmlearn Phi \n{model.emissionprob_}')


latent_states = ["$z_1$", "$z_2$"]
true_states = ["$x_1$", "$x_2$", "$x_3$"]

#A

fig, axes = plt.subplots(nrows=1, ncols=3)

min_colorbar = min(min([min(x) for x in A_true]), min([min(x) for x in A_hat]), min([min(x) for x in model.transmat_]))
max_colorbar =  max(max([max(x) for x in A_true]), max([max(x) for x in A_hat]), max([max(x) for x in model.transmat_]))

axes[0].imshow(A_true, cmap=plt.cm.Oranges, vmin = min_colorbar, vmax = max_colorbar)
axes[0].set_title('Original A', fontsize = 12)
axes[0].set_xticks(np.arange(len(latent_states)), labels=latent_states)
axes[0].set_yticks(np.arange(len(latent_states)), labels=latent_states)

axes[1].imshow(A_hat, cmap=plt.cm.Oranges, vmin = min_colorbar, vmax = max_colorbar)
axes[1].set_title('Custom a', fontsize = 12)
axes[1].set_xticks(np.arange(len(latent_states)), labels=latent_states)
axes[1].set_yticks(np.arange(len(latent_states)), labels=latent_states)

im = axes[2].imshow(model.transmat_, cmap=plt.cm.Oranges, vmin = min_colorbar, vmax = max_colorbar)
axes[2].set_title('hmm A', fontsize = 12)
axes[2].set_xticks(np.arange(len(latent_states)), labels=latent_states)
axes[2].set_yticks(np.arange(len(latent_states)), labels=latent_states)


cax,kw = mlp.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(im, cax=cax, **kw)

plt.show()


#Phi

fig, axes = plt.subplots(nrows=1, ncols=3)


min_colorbar = min(min([min(x) for x in Phi_true]), min([min(x) for x in Phi_hat]), min([min(x) for x in model.emissionprob_]))
max_colorbar =  max(max([max(x) for x in Phi_true]), max([max(x) for x in Phi_hat]), max([max(x) for x in model.emissionprob_]))

axes[0].imshow(Phi_true, cmap=plt.cm.Blues, vmin = min_colorbar, vmax = max_colorbar)
axes[0].set_title('Original Phi', fontsize = 12)
axes[0].set_yticks(np.arange(len(latent_states)), labels=latent_states)
axes[0].set_xticks(np.arange(len(true_states)), labels=true_states)

axes[1].imshow(Phi_hat, cmap=plt.cm.Blues, vmin = min_colorbar, vmax = max_colorbar)
axes[1].set_title('Custom Phi', fontsize = 12)
axes[1].set_yticks(np.arange(len(latent_states)), labels=latent_states)
axes[1].set_xticks(np.arange(len(true_states)), labels=true_states)

im = axes[2].imshow(model.emissionprob_, cmap=plt.cm.Blues, vmin = min_colorbar, vmax = max_colorbar)
axes[2].set_title('hmm Phi', fontsize = 12)
axes[2].set_yticks(np.arange(len(latent_states)), labels=latent_states)
axes[2].set_xticks(np.arange(len(true_states)), labels=true_states)

cax,kw = mlp.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(im, cax=cax, **kw)

plt.show()


#LogTraj
plt.figure()
plt.plot(range(Niter), logTraj)
plt.xlabel('EM iteration')
plt.ylabel('logL')
plt.title('Log likelihood')
plt.show()

