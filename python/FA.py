import numpy as np
from sklearn import preprocessing
import pandas as pd
from scipy import linalg

X = pd.read_csv('test/itemScores.csv', index_col=0)
y = pd.read_csv('test/NEOFACscores.csv', index_col=0)
m, n = X.shape
s = 2 # no. of factors to extract

# normalize X
X = preprocessing.scale(X)
# covariance matrix
R = np.cov(X, rowvar=False)

# maximize variance of original variables
a = linalg.svd(R, full_matrices=False, compute_uv=False)
# communality estimation by coefficients of multiple correlatio
c = np.ones((n,)) - 1 / np.diag(linalg.solve(R,np.diag(np.ones((n,)))))
g = np.array([])

for i in range(75):
    U, D, Vh = linalg.svd(R - np.diag(np.diag(R)) + np.diag(c), full_matrices=False, compute_uv=True)
    V = Vh.T
    D = np.diag(D)
    N = np.dot(V, np.sqrt(D[:,:2]))
    p = c
    c = np.sum(N**2,1)
    g = np.union1d(g, np.where(c>1)[0])
    if g: c[g] = 1
    if np.max(np.abs(c-p))<0.001:
        break

print 'Factorial number of iterations:', i+1

# evaluation of factor loadings and communalities estimation
B = np.hstack((N,c[:,np.newaxis]))

# normalization of factor loadings
h = np.sqrt(c)
N = N / h[:,np.newaxis]

L = N
z = 0

# iteration cycle maximizing variance of individual columns
for l in range(35):
    A, S, M = linalg.svd(np.dot(N.T, n * L**3 - np.dot(L, np.diag(np.sum(L**2, axis=0)))), full_matrices=False, compute_uv=True)
    L = np.dot(np.dot(N,A), M)
    b = z
    z = np.sum(S)
    if np.abs(z -b) < 0.00001:
        break

print 'Rotational number of iterations:',l+1

# unnormalization of factor loadings
L = L * h[:,np.newaxis]

# factors computation by regression and variance proportions
t = sum(L**2)/n
var = np.hstack((t, sum(t)))
fac = np.dot(np.dot(X,linalg.solve(R,np.diag(np.ones((n,))))), L)

# evaluation of given factor model variance specific matrix
r = np.diag(R) - np.sum(L**2,axis=1)
E = R - np.dot(L, L.T) - np.diag(r)
