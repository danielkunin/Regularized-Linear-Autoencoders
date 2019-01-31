# Loss Landscapes of Regularized Linear Autoencoders
##### [Daniel Kunin](http://daniel-kunin.com/), [Jonathan M. Bloom](https://www.broadinstitute.org/bios/jonathan-bloom), [Aleksandrina Goeva](https://macoskolab.com/about-us/), [Cotton Seed](https://www.broadinstitute.org/bios/cotton-seed)

Autoencoders are a deep learning model for representation learning. When trained to minimize the Euclidean distance between the data and its reconstruction, linear autoencoders (LAEs) learn the subspace spanned by the top principal directions but cannot learn the principal directions themselves. In this paper, we prove that L2-regularized LAEs learn the principal directions as the left singular vectors of the decoder, providing an extremely simple and scalable algorithm for rank-k SVD. More generally, we consider LAEs with (i) no regularization, (ii) regularization of the composition of the encoder and decoder, and (iii) regularization of the encoder and decoder separately. We relate the minimum of (iii) to the MAP estimate of probabilistic PCA and show that for all critical points the encoder and decoder are transposes. Building on topological intuition, we smoothly parameterize the critical manifolds for all three losses via a novel unified framework and illustrate these results empirically. Overall, this work clarifies the relationship between autoencoders and Bayesian models and between regularization and orthogonality. 

[Paper](https://arxiv.org/pdf/1901.08168.pdf), [Code](https://github.com/danielkunin/Regularized-Linear-Autoencoders/blob/master/Loss%20Landscapes%20of%20Regularized%20Linear%20Autoencoders%20Code.ipynb), [Clarification and Erratum](https://github.com/danielkunin/Regularized-Linear-Autoencoders/blob/master/erratum.md).

Feedback welcome! Contact Daniel and Jon: kunin@stanford.edu, jbloom@broadinstitute.org

### LAE-PCA Algorithm

Here's a simple gradient descent algorithm for **LAE-PCA** in NumPy:

```python
XXt = X @ X.T
while np.linalg.norm(W1 - W2.T) > epsilon:
    W1 -= alpha * ((W2.T @ (W2 @ W1 - I)) @ XXt + lamb * W1)
    W2 -= alpha * (((W2 @ W1 - I) @ XXt) @ W1.T + lamb * W2)

principal_directions, s,  _ = np.linalg.svd(W2, full_matrices = False)
eigenvalues = np.sqrt(lamb / (1 - s**2))
```

This may be accelerated on frameworks like TensorFlow using a host of math, hardware, sampling, and deep learning tricks, leveraging our topological and geometric understanding of the loss landscape.

The simplest improvement is to constrain `W2 = W1.T` *a priori* (see Appendix A):

```python
XXt = X @ X.T
diff = np.inf
while diff > epsilon:
    update = alpha * (((W2 @ W2.T - I) @ XXt) @ W2 + lamb * W2)
    W2 -= update
    diff = np.linalg.norm(update)

principal_directions, s,  _ = np.linalg.svd(W2, full_matrices = False)
eigenvalues = np.sqrt(lamb / (1 - s**2))
```

We call this version **regularized Oja's rule**, since without regularization the update step is identical to that of [Oja's Rule](http://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/OJA.pca.pdf).

Runnable examples of these algorithms and others are [here](https://github.com/danielkunin/Regularized-Linear-Autoencoders/blob/master/algorithms.py).

### Project History

Daniel interned with [Hail Team](https://hail.is/about.html) at the [Broad Institute of MIT and Harvard](https://broadinstitute.org) in Summer 2018. While the four of us were exploring models of deep matrix factorization for [single-cell RNA sequencing data](https://www.wired.com/story/the-human-cell-atlas-is-biologists-latest-grand-project/), Jon came across v1 of Elad Plaut's [
From Principal Subspaces to Principal Components with Linear Autoencoders](https://arxiv.org/abs/1804.10253) and contacted the author regarding a subtle error. Plaut agreed the main theorem was false as stated, and in correspondence convincingly re-demonstrated the existance of a fascinating empirical phenomenon. We realized quickly from Plaut's code and the scalar case that regularization was the key to symmetry-breaking and rigidity; the form of the general solution was also clear and simple to verify empirically. The bulk of our time was spent banging against chalkboards and whiteboards to establish additional cases and finally a proof in full generality.
