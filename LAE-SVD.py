import numpy as np
import time

n = 1000
m = 10
k = 2

eps = 1e-8
alpha = 1e-5
lamb = 20
I = np.eye(m)

np.random.seed(0)
X = np.random.normal(size = (m, n))
XXt = X @ X.T

W1 = np.random.normal(size = (k, m))
W2 = np.random.normal(size = (m, k))

start = time.time()
u, s, _ = np.linalg.svd(X, full_matrices = False)
end = time.time()
t = end - start

start = time.time()
while np.linalg.norm(W1 - W2.T) > eps:
    W1 -= alpha * ((W2.T @ (W2 @ W1 - I)) @ XXt + lamb * W1)
    W2 -= alpha * (((W2 @ W1 - I) @ XXt) @ W1.T + lamb * W2)
u2, s2, _ = np.linalg.svd(W2, full_matrices = False)
s2 = np.sqrt(lamb / (1 - s2**2))
end = time.time()
t2 = end - start

print('SVD ({:0.5f} secs)'.format(t))
print(s[0:k])
print(u[:, 0:k])
print('LAE-SVD ({:0.5f} secs)'.format(t2))
print(s2)
print(u2)
