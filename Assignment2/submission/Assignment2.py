### Assignment PDF given code
from iris_utils import *
import numpy as np
import matplotlib.pyplot as plt
X, y = load_iris_csv("iris.csv")
Xtr_raw, Xte_raw, ytr, yte = stratified_split(X, y, test_fraction=0.3, seed=0)
Xtr, Xte, mu, sd = standardize(Xtr_raw, Xte_raw)

### Implement sigmoid(z)
def sigmoid(z):
    # no overflow warning for z=abs(1000)
    # clip to abs(500), beyond that call np.exp()
    z = np.clip(z, -500,500)
    return 1 / (1 + np.exp(-z))

### Implement binary loss and binary grad
def binary_loss(theta, Xb, y, lam):
    loss = 0
    # calculate first part
    first_sum = 0
    p = sigmoid(Xb @ theta) #@ is matrix multiplication
    p = np.clip(p, 1e-12, 1 - 1e-12)
    first_term = y * np.log(p)
    second_term = (1 - y) * np.log(1-p)
    first_sum = -np.mean(first_term+second_term)

    # calculate second part
    second_sum = 0
    theta_copy = theta.copy() # prevent from destroying bias terms - learned the hard way
    theta_copy[0] = 0
    second_sum = lam/2 * np.sum(theta_copy**2)

    #combine
    loss = first_sum+second_sum
    return loss

def binary_grad(theta, Xb, y, lam):
    gradient = 0
    #claculate p
    p = sigmoid(Xb @ theta) #@ is matrix multiplication
    p = np.clip(p, 1e-12, 1 - 1e-12)
    first_term = np.dot(Xb.T,p - y)
    first_term *= (1/len(y))

    theta_copy = theta.copy() # prevent from destroying bias terms - learned the hard way
    theta_copy[0] = 0

    second_term = lam * theta_copy

    grad = first_term+second_term
    return grad

### Implement class binary logistic regression
class BinaryLogisticRegression:
    def __init__(self, iters=2000, lam=0.01,n=0.5):
        #default values
        self.iters = iters
        self.lam = lam
        self.n = n
        self.theta = None

    ### Implement fit
    def fit(self, X, y):
        # add bias
        Xb = add_bias(X)

        # initialize weights
        self.theta = np.zeros(Xb.shape[1])

        # needs to return loss
        loss = []

        # iterate
        for i in range(self.iters):
            loss_i = binary_loss(self.theta, Xb, y, self.lam)
            loss.append(loss_i)
            grad = binary_grad(self.theta, Xb, y, self.lam)
            self.theta = self.theta - self.n*grad

        return loss

    
    def predict_proba(self, X):
        Xb = add_bias(X)
        return sigmoid(Xb @ self.theta)

    def predict(self, X):
        probs = self.predict_proba(X)
        preds = probs>= 0.5
        return preds.astype(int)

### Stage 1 - need to predict setosa
ytr_st1 = ytr.copy()
yte_st1 = yte.copy()

# convert 0's(setosa) to 1's and everything else to 0's
ytr_st1 = (ytr_st1==0).astype(int)
yte_st1 = (yte_st1==0).astype(int)

# train and test
stage1 = BinaryLogisticRegression(lam=0.1)
stage1_losses = stage1.fit(Xtr, ytr_st1)

stage1_train_preds = stage1.predict(Xtr)
stage1_test_preds = stage1.predict(Xte)

print("Binary Logistic Regression Train Accuracy : ", accuracy(ytr_st1, stage1_train_preds))
print("Binary Logistic Regression Test Accuracy : ", accuracy(yte_st1, stage1_test_preds))

plt.plot(stage1_losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

### End Stage 1

### Stage 2
class TwoStageClassifier:
    def __init__(self, iters =2000, lam=0.1, n=0.5):
        self.iters = iters
        self.lam = lam
        self.n = n
        self.stage1_blr = None #blr = binary logistic regression
        self.stage2_blr = None

    ### Implement fit
    def fit(self, X, y):
        # keep only setosas
        y_1 = (y==0).astype(int)

        # fit
        self.stage1_blr = BinaryLogisticRegression(self.iters, self.lam, self.n)
        self.stage1_blr.fit(X,y_1)

        # now only non-setosa rows
        not_setosa_rows = y!=0 #booleans of correct rows
        X_2 = X[not_setosa_rows] #just use mask, do not convert to int
        y_2 = (y[not_setosa_rows]==1).astype(int) #here convert to int for labels

        # fit
        self.stage2_blr = BinaryLogisticRegression(self.iters, self.lam, self.n)
        self.stage2_blr.fit(X_2,y_2)

    def predict_proba(self,X):
        # setosa then versicolor predictions
        q_1 = self.stage1_blr.predict_proba(X)
        q_2 = self.stage2_blr.predict_proba(X)

        p_setosa = q_1
        p_versicolor = (1-q_1)* q_2
        p_virginica = (1-q_1) * (1-q_2)

        probs = np.column_stack(
            (p_setosa, p_versicolor, p_virginica)
        )

        print("Verify probs sum to 1")
        print(np.sum(probs, axis=1))
        return probs
    def predict(self,X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

### Implement Softmax
def softmax(S):
    # clip to +-500
    S = np.clip(S, -500, 500)

    #subtract each row’s maximum before exponentiating, so
    #it is stable for large inputs
    S = S - np.max(S, axis=1, keepdims = True)

    #calculate p
    P = np.exp(S) / np.sum(np.exp(S), axis=1, keepdims = True) # on axis 1 for row

    return P

### Implement softmax loss and softmax grad
def softmax_loss(W, Xb, Y, lam):
    #calculate S
    S = Xb @ W

    P = softmax(S)
    P = np.clip(P, 1e-12, 1 - 1e-12)

    first_term = -np.mean(np.sum(Y * np.log(P),axis=1))

    second_term = lam/2 * np.sum(W[1:,:]**2)

    loss = first_term + second_term
    return loss

def softmax_grad(W, Xb, Y, lam):
    gradient = 0

    S = Xb @ W

    P = softmax(S)
    P = np.clip(P, 1e-12, 1 - 1e-12)

    #set bias rows to 0
    W_hat = W.copy()
    W_hat[0,:] = 0 #set first row to 0

    first_term = np.dot(Xb.T, P - Y)
    first_term *= (1/len(Y))

    second_term = lam * W_hat

    grad = first_term + second_term
    return grad

### Implement softmax regression
class SoftmaxRegression:
    def __init__(self, iters=3000, lam=0.01, n=0.5):
        self.iters = iters
        self.lam = lam
        self.n = n
        self.W = None

    def fit(self, X, Y):
        # add bias
        Xb = add_bias(X)

        # one hot encode Y
        Y = one_hot(Y, K=3)

        # initialize weights
        self.W = np.zeros((Xb.shape[1], Y.shape[1]))

        # needs to return loss
        loss = []

        # iterate
        for i in range(self.iters):
            loss_i = softmax_loss(self.W, Xb, Y, self.lam)
            loss.append(loss_i)
            grad = softmax_grad(self.W, Xb, Y, self.lam)
            self.W = self.W - self.n*grad

        return loss

    def predict_proba(self, X):
        Xb = add_bias(X)
        S = Xb @ self.W
        P = softmax(S)
        return P

    def predict(self, X):
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return preds

### train and test
ytr_p2 = ytr.copy()
yte_p2 = yte.copy()

# train and test
two_stage_model = TwoStageClassifier()
two_stage_model.fit(Xtr, ytr_p2)
two_stage_preds = two_stage_model.predict(Xte)

softmax_model = SoftmaxRegression()
softmax_model.fit(Xtr, ytr_p2)
softmax_preds = softmax_model.predict(Xte)

#accuracies
print("Two Stage Classifier Test Accuracy : ", accuracy(yte_p2, two_stage_preds))
print("Softmax Regression Test Accuracy : ", accuracy(yte_p2, softmax_preds))

#confusion matrix
print("Two Stage Classifier Confusion Matrix : ")
print(confusion_matrix(yte_p2, two_stage_preds))

print("Softmax Regression Confusion Matrix : ")
print(confusion_matrix(yte_p2, softmax_preds))

### End Stage 2