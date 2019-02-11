import numpy as np
from sklearn.utils.extmath import randomized_svd
import time
import matplotlib.pyplot as plt
import scipy.linalg

n = 10000
m = 10
k = 2

eps = 1e-8
alpha = 1e-5
lamb = 50
I = np.eye(m)

np.random.seed(0)
X = np.random.normal(size = (m, n))
X = X - np.mean(X, axis=1, keepdims=True)

W1 = np.random.normal(size = (k, m))
W2 = np.random.normal(size = (m, k))

# auxiliary function that computes the distance between two SVDs
def compute_dist(W2,k,u_svd):
    dist = 0;
    u, s, _ = np.linalg.svd(W2, full_matrices = False)
    for col in range(k):
        dist += np.min([np.linalg.norm(u_svd[:,col]-u[:,col]), np.linalg.norm(-u_svd[:,col]-u[:,col])])
    return dist

# SVD with dgesdd (divide and conquer)
def svd():
    start = time.time()
    u, s, _ = np.linalg.svd(X, full_matrices = False)
    end = time.time()
    t = end - start
    return (u,s,t)

# Randomized SVD
def randomized_svd_wrapper():
    start = time.time()
    u, s, _ = randomized_svd(X, 2)
    end = time.time()
    t = end - start
    return (u,s,t)

# LAE-PCA using gradient descent
def LAE_PCA_GD(W1,W2,u_svd = None):
    W1 = W1.copy()
    W2 = W2.copy()
    start = time.time()
    XXt = X @ X.T
    i = 0
    dist = []
    while np.linalg.norm(W1 - W2.T) > eps:
        W1 -= alpha * ((W2.T @ (W2 @ W1 - I)) @ XXt + lamb * W1)
        W2 -= alpha * (((W2 @ W1 - I) @ XXt) @ W1.T + lamb * W2)
        
        if u_svd is not None:
            dist.append( compute_dist(W2,k,u_svd) )
        i += 1
        
    u, s, _ = np.linalg.svd(W2, full_matrices = False)
    s = np.sqrt(lamb / (1 - s**2))
    end = time.time()
    t = end - start
    return (u,s,t,i,dist)

# Regularized Oja's Rule
def LAE_PCA_Oja_GD(W2, u_svd = None):
    W2 = W2.copy()
    start = time.time()
    XXt = X @ X.T
    diff = np.inf
    dist = []
    i = 0
    while diff > eps:
        update = alpha * (((W2 @ W2.T - I) @ XXt) @ W2 + lamb * W2)
        W2 -= update
        diff = np.linalg.norm(update)
        
        if u_svd is not None:
            dist.append( compute_dist(W2,k,u_svd) )
        i += 1
        
    u, s, _ = np.linalg.svd(W2, full_matrices = False)
    s = np.sqrt(lamb / (1 - s**2))
    end = time.time()
    t = end - start
    return (u,s,t,i,dist)

# iterative exact minimization
def LAE_PCA_exact(W1, W2, u_svd = None):
    W1 = W1.copy()
    W2 = W2.copy()
    start = time.time()
    XXt = X @ X.T
    i = 0
    dist = []
    while np.linalg.norm(W1 - W2.T) > eps:
        coefficient_matrix = np.kron(W2.T @ W2,XXt) # order reversed since matrices are stored per row
        np.fill_diagonal(coefficient_matrix, coefficient_matrix.diagonal() + lamb)
        W1 = scipy.linalg.solve(coefficient_matrix, (W2.T @ XXt).reshape((m*k,1)), assume_a = 'pos').reshape((k,m))
        
        right_hand_side = W1 @ XXt
        coefficient_matrix = right_hand_side @ W1.T
        np.fill_diagonal(coefficient_matrix, coefficient_matrix.diagonal() + lamb)
        W2 = scipy.linalg.solve(coefficient_matrix, right_hand_side, assume_a = 'pos').T
        
        if u_svd is not None:
            dist.append( compute_dist(W2,k,u_svd) )
        i += 1
    
    u, s, _ = np.linalg.svd(W2, full_matrices = False)
    s = np.sqrt(lamb / (1 - s**2))
    end = time.time()
    t = end - start
    return (u,s,t,i,dist)


def display(method, t, i, u, s):
    if i == None:
        print('{:s} ({:0.5f} secs)'.format(method,t))
    else:
        print('{:s}  ({:0.5f} secs, {} iterations)'.format(method, t, i))
    print(s[0:k])
    print(u[:, 0:k])


# perform timing runs
(u_svd,s,t) = svd()
display('SVD', t, None, u_svd, s)

(u,s,t) = randomized_svd_wrapper()
display('Randomized SVD', t, None, u, s)

(u,s,t,i2,_) = LAE_PCA_GD(W1, W2, None)
display('LAE-PCA (GD)', t, i2, u, s)

(u,s,t,i3,_) = LAE_PCA_Oja_GD(W2, None)
display('Regularized Oja\'s rule (GD)', t, i3, u, s)

(u,s,t,i4,_) = LAE_PCA_exact(W1, W2, None)
display('LAE-PCA (Exact)', t, i4, u, s)

# perform diagnostic runs
(_,_,_,_,dist2) = LAE_PCA_GD(W1, W2, u_svd)
(_,_,_,_,dist3) = LAE_PCA_Oja_GD(W2, u_svd)
(_,_,_,_,dist4) = LAE_PCA_exact(W1, W2, u_svd)

plt.plot(np.arange(len(dist2)),dist2)
plt.plot(np.arange(len(dist3)),dist3)
plt.plot(np.arange(len(dist4)),dist4)
plt.legend(['LAE-PCA-GD','LAE-PCA-Oja-GD','LAE-PCA-Exact'])