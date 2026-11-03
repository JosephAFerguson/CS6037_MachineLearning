### Libraries - Joe
import itertools
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import brentq, linprog
import math

### Part 1 - The VC Confidence term - Joe

### a - Implement fm(h) with η = 0.05 - Joe
def fm(h,m, η = 0.05):
    # sqrt( (h*a-b) / m )
    a = np.log(2*m / h)+1
    b = np.log(η/4)

    return np.sqrt( (h*a-b) / m)

### a - Plot fm(h) against h/m for m \in {10^3,10^4,10^5} - Joe
ms= [10**3,10**4,10**5]
hs = [0,0,0]
for i in range(len(ms)):
    # grab values of h evenly from 1...m
    hs[i] = np.round(np.linspace(1,ms[i],200)).astype(int)
    #print(hs[i])

# plot the three curves, they will look very similar until zoomed in
fig,ax = plt.subplots(figsize=(10,8))
ax.plot(hs[0]/ms[0], fm(hs[0], ms[0]), label='m=10^3', color='g')
ax.plot(hs[1]/ms[1], fm(hs[1], ms[1]), label='m=10^4', color='b')
ax.plot(hs[2]/ms[2], fm(hs[2], ms[2]), label='m=10^5', color='r')
ax.set_xlabel('h/m')
ax.set_ylabel('fm(h)')
ax.set_title("Plotting fm(h) against h/m")
ax.grid()
ax.legend()
plt.show()

### b - find the ratio at which fm(h)=1 - Joe
for m in ms:
    # brentq finds func at 0, so for 1, fm -1,
    # https://pythonguides.com/python-scipy-optimize-root/
    r = brentq(lambda h: fm(h*m, m) - 1, 1/m, 1)    
    print("h/m ratio for fm(h)=1 : m={}, h/m={:.4f}".format(m, r))

### c - find largest integer VC dim h for which bound guarantees R(alpha) <= 0.1
### when Remp=0, Remp=0.05 - Joe

# we have r(a) <= remp(a) + fm(h) bounded by 0.1, remp(a) + fm(h)<=0.1
# rearrange as fm(h) <= 0.1 - remp(a)
m=10**4
remps = [0,0.05]
for remp in remps:
    # check int vals 1->m
    hs = np.arange(1,m+1)
    # list of values that satisfy the bound
    satisfies = hs[fm(hs,m) <= 0.1 - remp]

    if not len(satisfies):
        print("No h satisfies")
    else:
        print("For Remp = {}, largest h = {}".format(remp,satisfies[-1]))

### Part 2 - A shattering checker - Joe

### Implement can_shatter(X) - Joe
def separable(X,y):
    """Return True if labels y in {-1,+1} are linearly separable on X."""
    m, n = X.shape
    # variables z = (w_1..w_n, b); constraint -y_i (w.x_i + b) <= -1
    A_ub = -y[:, None] * np.hstack([X, np.ones((m, 1))])
    b_ub = -np.ones(m)
    res = linprog(c=np.zeros(n + 1), A_ub=A_ub, b_ub=b_ub,
    bounds=[(None, None)] * (n + 1), method="highs")
    return res.status == 0 # 0 = feasible solution found


def can_shatter(X):
    # loop over all possible labelings
    labelings = itertools.product([-1,1], repeat=len(X))
    for labels in labelings:
        y = np.array(labels)

        # if any labeling is not separable, X can not shattered
        if not separable(X,y):
            return False

    return True

### a - verify the three non-collinear points - Joe
x1 = np.array([[0,0],[1,0], [0,1]])
print("Non-collinear:", can_shatter(x1))

### a - verify three collinear points - Joe
x2 = np.array([[0,0],[1,0], [2,0]])
print("Collinear:", can_shatter(x2))

### b - sweep through n and sample points, report table of shattering - Joe
ns = [2,3,4,5]
fracshatter1 = []
fracshatter2 = []
for n in ns:
    print(n)
    # num shatters
    shatter1 = 0
    shatter2 = 0

    for i in range(1000):
        # sample iid standard normal# Sample iid standard normal points
        points1 = np.random.normal(size=(n+1, n))
        points2 = np.random.normal(size=(n+2, n))

        if can_shatter(points1):
            shatter1 += 1

        if can_shatter(points2):
            shatter2 += 1

    # add to lists
    fracshatter1.append(shatter1/1000)
    fracshatter2.append(shatter2/1000)

# plot table
fig,ax = plt.subplots()
ax.axis("off")
ax.set_title("Shatterability in Dimension")
table = pd.DataFrame(
    {
    "N" : ns,
    "N+1 Points": fracshatter1,
    "N+2 Points" : fracshatter2
})
axtable = ax.table(cellText=table.values, colLabels=table.columns, loc='center', cellLoc='center')
plt.show()

### c - take random set, list those not separable - Joe
ranpoints = np.random.normal(size=(4,2))
ranpointsLabels = itertools.product([-1,1], repeat=len(ranpoints))

notSep = []
# iterate through labelings and determine not separable
for labels in ranpointsLabels:
    y = np.array(labels)

    if not separable(ranpoints,y):
        notSep.append(y)

print("Random Points (4,2): {}".format(ranpoints))
print("Non-separable Labelings: {}".format(notSep))

# visualize one label that is not separable
y = notSep[0]
fig,ax = plt.subplots()
ax.scatter(
    ranpoints[:,0],
    ranpoints[:,1],
    c=y,
    cmap='bwr',
    s=100
)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("4 Points - Non Separable")
ax.grid(True)
plt.show()

### d - write an analogous checker for axis-aligned rectangles - Joe

# essentially case one - find boundingbox - check if contains negative point
def bounding_box_check(X,y):
    #grab labels
    pos = X[y==1]
    neg = X[y==-1]

    #edge check
    if not len(pos) or not len(neg):
        return True

    #get bounding box
    mins = np.min(pos, axis=0)
    maxs = np.max(pos, axis=0)

    # check if negative exists inside bounding box
    for p in neg:
        if np.all(p >=mins) and np.all(p <= maxs):
            return False

    # none found
    return True

# new can chatter using bounding box only
def can_shatter_bounding_box(X):
    labelings = itertools.product([-1,1], repeat=len(X))
    for labels in labelings:
        y = np.array(labels)

        if not bounding_box_check(X,y):
            return False

    # separable
    return True