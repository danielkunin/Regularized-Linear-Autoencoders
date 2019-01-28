import numpy as np
from sklearn.utils.extmath import randomized_svd
import time

n = 10000
m = 10
k = 2

eps = 1e-8
alpha = 1e-5
lamb = 50
I = np.eye(m)

np.random.seed(0)
X = np.random.normal(size = (m, n))

# SVD with dgesdd (divide and conquer)
start = time.time()
u, s, _ = np.linalg.svd(X, full_matrices = False)
end = time.time()
t = end - start

# Randomized SVD
start = time.time()
u1, s1, _ = randomized_svd(X, 2)
end = time.time()
t1 = end - start

# LAE-SVD
start = time.time()
i = 0
W1 = np.random.normal(size = (k, m))
W2 = np.random.normal(size = (m, k))
XXt = X @ X.T
while np.linalg.norm(W1 - W2.T) > eps:
    W1 -= alpha * ((W2.T @ (W2 @ W1 - I)) @ XXt + lamb * W1)
    W2 -= alpha * (((W2 @ W1 - I) @ XXt) @ W1.T + lamb * W2)
    i += 1
u2, s2, _ = np.linalg.svd(W2, full_matrices = False)
s2 = np.sqrt(lamb / (1 - s2**2))
end = time.time()
t2 = end - start

print('SVD ({:0.5f} secs)'.format(t))
print(s[0:k])
print(u[:, 0:k])
print('Randomized SVD ({:0.5f} secs)'.format(t1))
print(s1[0:k])
print(u1[:, 0:k])
print('LAE-SVD ({:0.5f} secs, {} iterations)'.format(t2, i))
print(s2)
print(u2)
