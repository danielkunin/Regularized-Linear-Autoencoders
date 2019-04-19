import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors

def LAE_PCA_untied(X, W1, W2, lamb, alpha, eps, bsize):
    n = X.size
    XXt = X @ X.T
    path = [[W1, W2]]
    i = 0
    while np.linalg.norm(W1 - W2) > eps:

        if bsize != None:
            batch = np.random.choice(n, bsize, False)
            XXt = X[batch] @ X[batch].T

        # W1 -= alpha * ((W2 * (W2 * W1 - 1)) *XXt + lamb * W1)
        # W2 -= alpha * (((W2 * W1 - 1) * XXt) * W1 + lamb * W2)

        W1 -= 1/np.linalg.norm(W2 * XXt * W2) * ((W2 * (W2 * W1 - 1)) *XXt + lamb * W1)
        W2 -= 1/np.linalg.norm(W1 * XXt * W1) * (((W2 * W1 - 1) * XXt) * W1 + lamb * W2)
        
        path.append([W1, W2])
        i += 1
    return np.array(path).T, i

def LAE_PCA_tied(X, W1, W2, lamb, alpha, eps, bsize):
    n = X.size
    XXt = X @ X.T
    W = W1
    path = [[W, W]]
    i = 0
    diff = np.inf
    while (diff > eps):

        if bsize != None:
            batch = np.random.choice(n, bsize, False)
            XXt = X[batch] @ X[batch].T

        W_prev = W

        coef = W * XXt * W
        grad = W * coef - XXt * W + lamb * W
        lr = 1 / np.linalg.norm(coef)
        W1 = W - lr * grad

        coef = W1 * XXt * W1
        grad = W1 * coef - XXt * W1 + lamb * W1
        lr = 1 / np.linalg.norm(coef)
        W2 = W1 - lr * grad
        
        W = (W1 + W2) / 2
        
        diff = np.linalg.norm(W - W_prev)

        path.append([W, W])
        i += 1
    return np.array(path).T, i

def LAE_PCA_exact(X, W1, W2, lamb, alpha, eps, bsize):
    n = X.size
    XXt = X @ X.T
    W = W1
    path = [[W, W]]
    i = 0
    diff = np.inf
    while diff > eps:

        if bsize != None:
            batch = np.random.choice(n, bsize, False)
            XXt = X[batch] @ X[batch].T

        prev_W = W

        RHS = W * XXt
        LHS = RHS * W + lamb * 1
        w1 = RHS / LHS

        RHS = w1 * XXt
        LHS = RHS * w1 + lamb * 1
        w2 = RHS / LHS
        
        W = (w1 + w2) / 2
        diff = np.linalg.norm(W - prev_W)

        path.append([W, W])
        i += 1
        
    return np.array(path).T, i


def visualize(data, path, lamb, interval=20, save=False):
    # get dimensions  
    m, n = path.shape
    if (m != 2):
        return print("Animiation doesn't support these dimensions.")
    
    # set up plot
    fig, ax = plt.subplots()
    lim = np.abs(path).max() * 1.1
    if np.isinf(lim) | np.isnan(lim):
        return print("Optimization didn't converge.")
    ax = plt.axes(xlim=(-lim, lim), ylim=(-lim, lim))
    
    # create line path
    line = ax.plot([], [], 'o-', lw=1, markersize=3, c='k')
    time_template = 'iteration = %i / %i'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    # create color mesh
    x, y = np.meshgrid(np.linspace(-lim, lim, 100), np.linspace(-lim, lim, 100))
    z = lamb * (x**2 + y**2)
    for d in data:
        z += (d - y * x * d)**2
    z = z[:-1, :-1]
    zmin, zmax = np.abs(z).min(), np.abs(z).max()
    c = ax.pcolormesh(x, y, z, norm=colors.LogNorm(zmin, zmax), cmap='RdBu')
    plt.colorbar(c)
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.title('Untied')

    def init():
        line[0].set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = path[0,0:i]
        thisy = path[1,0:i]
        line[0].set_data(thisx, thisy)
        time_text.set_text(time_template % (i, n))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1,n+1), init_func=init, interval=interval, repeat=False)
    if save:
        ani.save('LAE-PCA.mp4', writer='ffmpeg', fps=50)
    else:
        plt.show()


# hyperparamters
lamb = 10
alpha = 1e-3
eps = 1e-3
bsize = 100
n = 2000
# parameters
W1 = 3
W2 = 4
# data
X = np.random.rand(n)

path, _ = LAE_PCA_untied(X, W1, W2, lamb*bsize/n, alpha, eps, bsize)
path, _ = LAE_PCA_tied(X, W1, W2, lamb*bsize/n, alpha, eps, bsize)
path, _ = LAE_PCA_exact(X, W1, W2, lamb*bsize/n, alpha, eps, bsize)

visualize(X, path, lamb, interval=10)
