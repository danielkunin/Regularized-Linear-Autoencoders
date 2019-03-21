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

class t_timer:
    def __init__(self):
        self.times = []
        self.start = time.time()
    def lap(self):
        end = time.time()
        self.times.append(end - self.start)
    def reset(self):
        self.start = time.time()
    def total(self):
        end = time.time()
        return np.sum(self.times) + end - self.start
        
def compute_svd_error(W2, k, u_svd):
    dist = 0;
    u, s, _ = np.linalg.svd(W2, full_matrices = False)
    for col in range(k):
        dist += np.min([np.linalg.norm(u_svd[:, col] - u[:, col]), np.linalg.norm(-u_svd[:, col] - u[:, col])])
    return dist

# SVD with dgesdd (divide and conquer)
def svd():
    timer = t_timer()
    u, s, _ = np.linalg.svd(X, full_matrices = False)
    t = timer.total()
    return (u, s, t)

# Randomized SVD
def randomized_svd_wrapper():
    timer = t_timer()
    u, s, _ = randomized_svd(X, 2)
    t = timer.total()
    return (u, s, t)

# LAE-PCA using untied weights and gradient descent
def LAE_PCA_untied(W1, W2, u_svd = None):
    W1 = W1.copy()
    W2 = W2.copy()
    dist = []
    timer = t_timer()
    XXt = X @ X.T
    i = 0
    while np.linalg.norm(W1 - W2.T) > eps:
        W1 -= alpha * ((W2.T @ (W2 @ W1 - I)) @ XXt + lamb * W1)
        W2 -= alpha * (((W2 @ W1 - I) @ XXt) @ W1.T + lamb * W2)
        
        if u_svd is not None:
            timer.lap()
            dist.append( compute_svd_error(W2, k, u_svd) )
            timer.reset()
        i += 1
        
    u, s, _ = np.linalg.svd(W2, full_matrices = False)
    s = np.sqrt(lamb / (1 - s**2))
    times = timer.times
    t = timer.total()
    return (u, s, t, i, dist, times)

# LAE-PCA using synchronized weights at initialization and gradient descent
def LAE_PCA_sync(W2, u_svd = None):
    W1 = W2.copy().T
    W2 = W2.copy()
    dist = []
    timer = t_timer()
    XXt = X @ X.T
    diff = np.inf
    i = 0
    while diff > eps:
        W1_update = alpha * ((W2.T @ (W2 @ W1 - I)) @ XXt + lamb * W1)
        W2_update = alpha * (((W2 @ W1 - I) @ XXt) @ W1.T + lamb * W2)
        W1 -= W1_update
        W2 -= W2_update
        diff = np.linalg.norm(W1_update) + np.linalg.norm(W2_update)
        
        if u_svd is not None:
            timer.lap()
            dist.append( compute_svd_error(W2, k, u_svd) )
            timer.reset()
        i += 1
        
    u, s, _ = np.linalg.svd(W2, full_matrices = False)
    s = np.sqrt(lamb / (1 - s**2))
    times = timer.times
    t = timer.total()
    return (u, s, t, i, dist, times)

# LAE-PCA using single weight matrix and gradient descent (Regularized Oja's Rule)
def LAE_PCA_oja(W2, u_svd = None):
    W2 = W2.copy()
    dist = []
    timer = t_timer()
    XXt = X @ X.T
    diff = np.inf
    i = 0
    while diff > eps:
        update = alpha * (((W2 @ W2.T - I) @ XXt) @ W2 + lamb * W2)
        W2 -= update
        diff = np.linalg.norm(update)
        
        if u_svd is not None:
            timer.lap()
            dist.append( compute_svd_error(W2, k, u_svd) )
            timer.reset()
        i += 1
        
    u, s, _ = np.linalg.svd(W2, full_matrices = False)
    s = np.sqrt(lamb / (1 - s**2))
    times = timer.times
    t = timer.total()
    return (u, s, t, i, dist, times)

# LAE-PCA using untied weight matrices and alternating exact minimization
def LAE_PCA_exact(W1, W2, synced = False, u_svd = None):
    W1 = W1.copy()
    W2 = W2.copy()
    dist = []
    timer = t_timer()
    XXt = X @ X.T
    diff = np.inf
    i = 0
    while diff > eps:
        if not synced:
            coefficient_matrix = np.kron(W2.T @ W2, XXt) # order reversed since matrices are stored per row
            np.fill_diagonal(coefficient_matrix, coefficient_matrix.diagonal() + lamb)
            W1 = scipy.linalg.solve(coefficient_matrix, (W2.T @ XXt).reshape((m*k, 1)), assume_a = 'pos').reshape((k, m))
        
        right_hand_side = W1 @ XXt
        coefficient_matrix = right_hand_side @ W1.T
        np.fill_diagonal(coefficient_matrix, coefficient_matrix.diagonal() + lamb)
        W2 = scipy.linalg.solve(coefficient_matrix, right_hand_side, assume_a = 'pos').T
        
        diff = np.linalg.norm(W1 - W2.T)
        
        if synced:
            W1 = W2.T
        
        if u_svd is not None:
            timer.lap()
            dist.append( compute_svd_error(W2, k, u_svd) )
            timer.reset()
        i += 1
    
    u, s, _ = np.linalg.svd(W2, full_matrices = False)
    s = np.sqrt(lamb / (1 - s**2))
    times = timer.times
    t = timer.total()
    return (u, s, t, i, dist, times)




def display(method, t, i, u, s):
    if i == None:
        print('{:s} ({:0.5f} secs)'.format(method, t))
    else:
        print('{:s}  ({:0.5f} secs, {} iterations)'.format(method, t, i))
    print(s[0:k])
    print(u[:, 0:k])


algorithms = [('LAE-PCA (GD-untied)',          LAE_PCA_untied, [W1, W2]),
              ('LAE-PCA (GD-sync)',            LAE_PCA_sync,   [W2]),
              ('LAE-PCA (GD-oja)',             LAE_PCA_oja,    [W2]),
              ('LAE-PCA (exact-alternating)',  LAE_PCA_exact,  [W1, W2, False]),
              ('LAE-PCA (exact-sync)',         LAE_PCA_exact,  [W1, W2, True])]

# perform timing runs
(u_svd, s, t) = svd()
display('SVD', t, None, u_svd, s)

(u, s, t) = randomized_svd_wrapper()
display('Randomized SVD', t, None, u, s)

for algorithm in algorithms:
    (u, s, t, i, _, _) = algorithm[1](*algorithm[2], None)
    display(algorithm[0], t, i, u, s)

# perform diagnostic runs
distances = []
runtimes = []
for algorithm in algorithms:
    (_ ,_ ,_ ,_ ,distance,runtime) = algorithm[1](*algorithm[2], u_svd)
    distances.append(distance)
    runtimes.append(runtime)

# plot results
legend = []
for idx, algorithm in enumerate(algorithms):
    legend.append(algorithm[0])
    plt.plot(distances[idx])

plt.legend(legend)
plt.title('Rate of convergence')
plt.xlabel('Iteration')
plt.ylabel('Error in SVD factor U')
plt.show()

for idx, algorithm in enumerate(algorithms):
    plt.plot(np.cumsum(runtimes[idx]), distances[idx])

plt.legend(legend)
plt.title('Rate of convergence')
plt.xlabel('Time (sec)')
plt.ylabel('Error in SVD factor U')
plt.show()