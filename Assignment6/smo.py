### Libararies - Joe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

### Define constants - Joe
RANDOM_SEED = 0
EPS = 10**-8
TAU = 10**-3
MAX_ITER = 10**4

### Kernel Functions where x and z are numpy arrays - Joe
def K1(x,z): # linear
    return x.T @ z
def K2(x,z): # feature space of K
    return np.square(K1(x,z)+1)

### 1, init - Joe
def InitSMO(X,y, KernelFunc, seed=RANDOM_SEED):
    m=len(y)
    np.random.seed(seed)

    # precompute gram mat
    K = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            K[i,j] = KernelFunc(X[i], X[j])

    # initialize alpha randomly
    U =np.random.uniform(0,1,m)
    alpha = np.zeros(m)
    alpha[y==1] = U[y==1] / np.sum(U[y==1]) # positive examples
    alpha[y==-1] = U[y==-1] / np.sum(U[y==-1]) # negative examples

    bias = 0
    return K,alpha,bias

### 2, evaluate decision func - Joe
def evalSMO(K,y,alpha,bias):
    return K @ (alpha * y) + bias
def evalSMO_Alt(X,y,alpha,bias):
    w = X.T @ (alpha*y)
    return X @ w + bias

### 3, compute errors and KKT quantities, and violations - Joe
def computeErrorSmo(K,y,alpha,bias):
    return evalSMO(K,y,alpha,bias) - y
def computeKKTSmo(K,y,alpha,bias):
    margin = y * evalSMO(K,y,alpha,bias)
    KKT = alpha * (margin-1)
    return KKT, margin
def computeKKTViolations(alpha,margin):
    return np.where(alpha>0,np.abs(margin-1), np.maximum(0,1-margin))

### 5, update ai2, clip it to feasible segment - Joe
def updateAlphaI2(alpha,y,i1,i2,e,k):
    ai1 = alpha[i1]
    ai2 = alpha[i2]

    # check bounds
    if y[i1] != y[i2]:
        L = max(0,ai2-ai1)
        H = np.inf
    else:
        L = 0
        H = ai1 +ai2

    # update values
    ai2_unc = ai2 + (y[i2] * e) / k
    ai2_new = np.clip(ai2_unc,L,H) #same as min(max(unc,L),H)

    return ai2_new

### 6, update ai1, preserving cond - Joe
def updateAlphaI1(alpha, y, i1, i2, ai2_new):
    ai1 = alpha[i1]
    ai2 = alpha[i2]

    # compute new and return
    return ai1 + y[i1]*y[i2]*(ai2-ai2_new)

### 8 - compute new bias - Joe
def computeBias(K,y,alpha,sV,b):
    if np.any(sV):
        b = np.mean(y[sV] - (K @ (alpha * y))[sV]) # [sV] is boolean that selects only support vectors
    return b

### pick pair loop - Joe
def pickPairLoop(K,y,alpha,bias, Error, KKT, margin,v, eps=EPS):
    updated = False

    # 4 - select i1 by next-worst violator
    i1List = np.argsort(-v)
    for i1 in i1List:
        # 4-(b) e(i)
        e = Error[i1] - Error

        # 4-(c) compute k for each i2
        ks = K[i1,i1] + np.diag(K) - (2*K[i1,:]) #K_ii is same as diag mat

        ## check greater than zero
        pos = ks >0
        pos[i1] = False # i \ne i1 condition

        ## edge case, none are valid
        if not np.any(pos):
            continue

        # 4 - select  i2 by next-best from the argmax
        absErrors = np.where(pos,np.abs(e), -np.inf) # assigns negative infinity to invalid candidates
        i2List = np.argsort(-absErrors)
        for i2 in i2List:
            # edgecheck - i2 not positive
            if not pos[i2]:
                continue

            k = ks[i2]

            # 5 - update alpha ai2
            ai2_new = updateAlphaI2(alpha,y,i1,i2,e[i2],k)

            # 6 - check if ai2 changed
            if abs(ai2_new - alpha[i2]) <= eps:
                continue

            # 6 - update alpha ai1
            ai1_new = updateAlphaI1(alpha,y,i1,i2,ai2_new)

            # apply updates
            alpha[i1] = ai1_new
            alpha[i2] = ai2_new

            updated = True
            break
        if updated:
            break
    return updated, alpha

### Implement SMO - Combining 1-6 with parts 7-10 in training loop - Joe
def smo(X,y,kernel, eps=10**-8,tau=10**-3,T=10**4,seed=0):
    m=len(y)

    # 1 - init
    K,alpha,bias = InitSMO(X,y,kernel, seed=seed)

    # 9 - Monitor
    wObjectives = []
    numSVs = []
    numMiscs= []
    constraintErrors = []
    converged = False


    iter = 0
    for iter in range(T):
        # 2 - evaluate decision function
        f = evalSMO(K,y,alpha,bias)

        # 3 - compute errors, KKT, quantities, violations
        Error = computeErrorSmo(K,y,alpha,bias)
        KKT, margin = computeKKTSmo(K,y,alpha,bias)
        v = computeKKTViolations(alpha,margin)

        # stopping test after step 3
        if np.max(v) < tau:
            print("Converged, iteration number {}".format(iter))
            converged = True
            break

        # 4 - select pair in order of worst KKT violations
        updated,alpha = pickPairLoop(K,y,alpha,bias,Error,KKT, margin,v,eps=eps)

        if not updated:
            print("Could not update any pair, end on iteration number {}".format(iter))
            break
        
        # 7 - Clean up
        alpha[alpha < eps] = 0

        # 8 - Recompute bias
        supportVectors = alpha > 0
        bias = computeBias(K,y,alpha,supportVectors,bias)

        # 9 - monitor

        ## objective function
        wObjective = np.sum(alpha) - 0.5 * ((alpha * y) @ K @ (alpha * y))

        ## number of support vectors
        numSV = np.sum(supportVectors)

        ## number of misclassified examples
        f = evalSMO(K,y,alpha,bias)
        miscs = np.sum(np.sign(f) != y)

        ## sum of constraint violations
        constraintError = np.sum(y*alpha)

        ## append to lists
        wObjectives.append(wObjective)
        numSVs.append(numSV)
        numMiscs.append(miscs)
        constraintErrors.append(constraintError)
        
    ## make into dict
    monitor = {
        "wObjectives": wObjectives,
        "numSVs": numSVs,
        "numMiscs": numMiscs,
        "constraintErrors": constraintErrors,
        "iters": iter,
        "converged": converged,
        "alpha": alpha.copy(),
        "bias": bias,
        "wNorm": np.sqrt(max(0, (alpha * y) @ K @ (alpha * y)))
    }

    return alpha, bias, monitor
