import numpy as np
from sklearn.utils.extmath import randomized_svd
import time
import matplotlib.pyplot as plt
import scipy.linalg
from enum import Enum

n = 10000
m = 10
k = 2

eps = 1e-8
alpha = 1e-5
lamb = 50
Im = np.eye(m)
Ik = np.eye(k)

np.random.seed(0)
X = np.random.normal(size = (m, n))
X = X - np.mean(X, axis=1, keepdims=True)

W1 = np.random.normal(size = (k, m))
W2 = np.random.normal(size = (m, k))


class SyncMode(Enum):
    UNTIED = 1
    SYNC = 2
    TIED = 3

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
    
class error_metric_svd:
    def __init__(self, u_svd, k):
        self.description = 'Error in SVD factor U'
        self.u_svd = u_svd
        self.k = k
    def compute_error(self, W2, W1 = None):
        dist = 0
        u, s, _ = np.linalg.svd(W2, full_matrices = False)
        for col in range(k):
            dist += np.min([np.linalg.norm(self.u_svd[:, col] - u[:, col]), np.linalg.norm(-self.u_svd[:, col] - u[:, col])])
        return dist

class error_metric_objective:
    def __init__(self, X, lamb):
        self.X = X
        self.lamb = lamb
        self.description = 'Objective value (lambda = ' + str(lamb) + ')'
    def compute_error(self, W2, W1 = None):
        if W1 is None:
            return np.linalg.norm(X - W2 @ W2.T @ X, 'fro')**2 + 2 * lamb * np.linalg.norm(W2)**2
        else:
            return np.linalg.norm(X - W2 @ W1   @ X, 'fro')**2 +     lamb * np.linalg.norm(W2)**2 + lamb * np.linalg.norm(W1)**2


# SVD with dgesdd (divide and conquer)
def svd():
    timer = t_timer()
    u, s, _ = np.linalg.svd(X, full_matrices = False)
    t = timer.total()
    return (u, s, t)

def rsvd(k):
    timer = t_timer()
    u, s, _ = randomized_svd(X, k)
    t = timer.total()
    return (u, s, t)

# LAE-PCA using gradient descent
def LAE_PCA_GD(W1, W2, syncMode, error_metric = None):
    W1 = W1.copy()
    W2 = W2.copy()
    dist = []
    timer = t_timer()
    XXt = X @ X.T
    diff = np.inf
    i = 0
    while diff > eps:
        if syncMode == SyncMode.UNTIED:
            W1 -= alpha * ((W2.T @ (W2 @ W1 - Im)) @ XXt + lamb * W1)
            W2 -= alpha * (((W2 @ W1 - Im) @ XXt) @ W1.T + lamb * W2)
            diff = np.linalg.norm(W1 - W2.T)
        elif syncMode == SyncMode.SYNC:
            W1_update = alpha * ((W2.T @ (W2 @ W1 - Im)) @ XXt + lamb * W1)
            W2_update = alpha * (((W2 @ W1 - Im) @ XXt) @ W1.T + lamb * W2)
            W1 -= W1_update
            W2 -= W2_update
            
            diff = np.linalg.norm(W1_update) + np.linalg.norm(W2_update)
        else:
            update = alpha * (((W2 @ W2.T - Im) @ XXt) @ W2 + lamb * W2)
            W2 -= update
            W1 = W2.T
            diff = 2 * np.linalg.norm(update)
        
        if error_metric is not None:
            timer.lap()
            dist.append( error_metric(W2, W1) )
            timer.reset()
        i += 1
        
    u, s, _ = np.linalg.svd(W2, full_matrices = False)
    s = np.sqrt(lamb / (1 - s**2))
    times = timer.times
    t = timer.total()
    return (u, s, t, i, dist, times)

# LAE-PCA using exact minimization
def LAE_PCA_exact(W1, W2, syncMode, error_metric = None):
    W1 = W1.copy()
    W2 = W2.copy()
    dist = []
    timer = t_timer()
    XXt = X @ X.T
    diff = np.inf
    i = 0
    while diff > eps:
        if syncMode == SyncMode.UNTIED:
            LHS = np.kron(W2.T @ W2, XXt) # order reversed since matrices are stored per row
            np.fill_diagonal(LHS, LHS.diagonal() + lamb)
            RHS = (W2.T @ XXt).reshape((m*k, 1))
            W1 = scipy.linalg.solve(LHS, RHS, assume_a = 'pos').reshape((k, m))
        
            RHS = W1 @ XXt
            LHS = RHS @ W1.T + lamb * Ik
            W2 = scipy.linalg.solve(LHS, RHS, assume_a = 'pos').T
            
            diff = np.linalg.norm(W1 - W2.T)
        elif syncMode == SyncMode.SYNC:
            RHS = W2.T @ XXt
            LHS = RHS @ W2 + lamb * Ik
            W1 = scipy.linalg.solve(LHS, RHS, assume_a = 'pos')
    
            RHS = W1 @ XXt
            LHS = RHS @ W1.T + lamb * Ik
            W2 = scipy.linalg.solve(LHS, RHS, assume_a = 'pos').T
            
            W1 = (W1 + W2.T) / 2
            W2 = W1.T
            
            if i == 0:
                diff = eps + 1
            else:
                diff = np.linalg.norm(W1 - prev_W1)
            prev_W1 = W1
        else:
            RHS = W1 @ XXt
            LHS = RHS @ W1.T + lamb * Ik
            W2 = scipy.linalg.solve(LHS, RHS, assume_a = 'pos').T
            diff = np.linalg.norm(W1 - W2.T)
            W1 = W2.T
        
        
        if error_metric is not None:
            timer.lap()
            dist.append( error_metric(W2, W1) )
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
        print('{:s} ({:0.5f} secs, {} iterations)'.format(method, t, i))
    print(s[0:k])
    print(u[:, 0:k])


algorithms = [('GD-untied',     LAE_PCA_GD,    [W1,   W2, SyncMode.UNTIED]),
              ('GD-sync',       LAE_PCA_GD,    [W2.T, W2, SyncMode.SYNC]),
              ('GD-tied',       LAE_PCA_GD,    [W1,   W2, SyncMode.TIED]),
              ('exact-untied',  LAE_PCA_exact, [W1,   W2, SyncMode.UNTIED]),
              ('exact-sync',    LAE_PCA_exact, [W2.T, W2, SyncMode.SYNC]),
              ('exact-tied',    LAE_PCA_exact, [W1,   W2, SyncMode.TIED]),]

# perform timing runs
(u_svd, s, t) = svd()
display('SVD', t, None, u_svd, s)

(u, s, t) = rsvd(k)
display('Randomized SVD', t, None, u, s)

for algorithm in algorithms:
    (u, s, t, i, _, _) = algorithm[1](*algorithm[2], None)
    display(algorithm[0], t, i, u, s)

# perform diagnostic runs
error_metric = error_metric_svd(u_svd, k)
#error_metric = error_metric_objective(X, lamb)
distances = []
runtimes = []
for algorithm in algorithms:
    (_ ,_ ,_ ,_ ,distance,runtime) = algorithm[1](*algorithm[2], error_metric.compute_error)
    distances.append(distance)
    runtimes.append(runtime)
    
# plot results
legend = []
for idx, algorithm in enumerate(algorithms):
    legend.append(algorithm[0])
    plt.plot(distances[idx])

plt.legend(legend)

plt.title('LAE-PCA - Rate of convergence')
plt.xlabel('Iteration')
plt.ylabel(error_metric.description)
plt.show()

for idx, algorithm in enumerate(algorithms):
    plt.plot(np.cumsum(runtimes[idx]), distances[idx])

plt.legend(legend)
plt.title('Rate of convergence')
plt.xlabel('Time (sec)')
plt.ylabel(error_metric.description)
plt.show()