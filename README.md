# Loss Landscapes of Regularized Linear Autoencoders
##### [Daniel Kunin](http://daniel-kunin.com/), [Jonathan M. Bloom](https://www.broadinstitute.org/bios/jonathan-bloom), [Aleksandrina Goeva](https://macoskolab.com/about-us/), [Cotton Seed](https://www.broadinstitute.org/bios/cotton-seed)

Autoencoders are a deep learning model for representation learning. When trained to minimize the Euclidean distance between the data and its reconstruction, linear autoencoders (LAEs) learn the subspace spanned by the top principal directions but cannot learn the principal directions themselves. In this paper, we prove that L2-regularized LAEs learn the principal directions as the left singular vectors of the decoder, providing an extremely simple and scalable algorithm for rank-k SVD. More generally, we consider LAEs with (i) no regularization, (ii) regularization of the composition of the encoder and decoder, and (iii) regularization of the encoder and decoder separately. We relate the minimum of (iii) to the MAP estimate of probabilistic PCA and show that for all critical points the encoder and decoder are transposes. Building on topological intuition, we smoothly parameterize the critical manifolds for all three losses via a novel unified framework and illustrate these results empirically. Overall, this work clarifies the relationship between autoencoders and Bayesian models and between regularization and orthogonality; most excitingly, the transpose result suggests a local biological mechanism by which minimizing reconstruction error and energetic expenditure could give rise to backpropogation as the *optimization algorithm* underlying efficient neural coding.

[Paper](https://arxiv.org/pdf/1901.08168.pdf), [Code](https://github.com/danielkunin/Regularized-Linear-Autoencoders/blob/master/Loss%20Landscapes%20of%20Regularized%20Linear%20Autoencoders%20Code.ipynb), [Clarification and Erratum](https://github.com/danielkunin/Regularized-Linear-Autoencoders/blob/master/erratum.md).

Feedback welcome! Contact Daniel and Jon: kunin@stanford.edu, jbloom@broadinstitute.org

### LAE-PCA Algorithms

Runnable examples of the algorithms below and others are [here](https://github.com/danielkunin/Regularized-Linear-Autoencoders/blob/master/algorithms.py).

Here's a simple gradient descent algorithm for **LAE-PCA** in NumPy:

```python
XXt = X @ X.T
while np.linalg.norm(W1 - W2.T) > epsilon:
    W1 -= alpha * ((W2.T @ (W2 @ W1 - I)) @ XXt + lamb * W1)
    W2 -= alpha * (((W2 @ W1 - I) @ XXt) @ W1.T + lamb * W2)

principal_directions, s,  _ = np.linalg.svd(W2, full_matrices = False)
eigenvalues = np.sqrt(lamb / (1 - s**2))
```

This may be accelerated on frameworks like TensorFlow using a host of math, hardware, sampling, and deep learning tricks, leveraging our topological and geometric understanding of the loss landscape. For example, one can alternate between exact (convex) minimization with respect to `W1` fixing `W2` and then with respect to `W2` fixing `W1`. Or one can constrain, or tie, `W2 = W1.T` *a priori* (see Appendix A):

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

We call this version **regularized Oja's rule**, since without regularization the update step is identical to that of [Oja's Rule](http://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/OJA.pca.pdf). Note that this formulation is not convex.

We implemented two other variants of the LAE-PCA algorithm and compared their convergence rates.

<p align="center">
    <img src="/img/time.png" alt="error per sec" width="300"/>
    <img src="/img/iteration.png" alt="error per iteration" width="300"/>
</p>

### Visualization

Here is an interactive [visualization](https://danielkunin.github.io/Regularized-Linear-Autoencoders/) of the three loss landscapes in the `m = k = 1` (scalar) case as well as the `m = 2, k = 1` case with tied weights.

<p align="center">
    <img src="/img/visualization.gif" alt="visualization demo" width="400"/>
</p>

### Project History

Daniel interned with [Hail Team](https://hail.is/about.html) at the [Broad Institute of MIT and Harvard](https://broadinstitute.org) in Summer 2018. While the four of us were exploring models of deep matrix factorization for [single-cell RNA sequencing data](https://www.wired.com/story/the-human-cell-atlas-is-biologists-latest-grand-project/), Jon came across v1 of Elad Plaut's [
From Principal Subspaces to Principal Components with Linear Autoencoders](https://arxiv.org/abs/1804.10253) and contacted the author regarding a subtle error. Plaut agreed the main theorem was false as stated, and in correspondence convincingly re-demonstrated the existence of a fascinating empirical phenomenon. We realized quickly from Plaut's code and the scalar case that regularization was the key to symmetry-breaking and rigidity; the form of the general solution was also clear and simple to verify empirically. The bulk of our time was spent banging against chalkboards and whiteboards to establish additional cases and finally a unified proof in full generality.

### Computational Neuroscience

In response to v1 of the preprint, [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/) wrote us that "the question of obtaining the transpose is actually pretty important for research on a biologically plausible version of backprop, because if you obtain approximate transposes, then several local learning rules give rise to gradient estimator analogues of backprop." He also pointed us to his wonderful papers [Towards Biologically Plausible Deep Learning](https://arxiv.org/pdf/1502.04156.pdf) and [How Important Is Weight Symmetry in Backpropagation?](https://arxiv.org/pdf/1510.05067.pdf)

Neuroscientist [Dan Bear](http://neuroailab.stanford.edu/people.html) was also kind enough to share his perspective as follows:

Deep Neural Networks (DNNs) are algorithms that apply a sequence of linear and nonlinear transformations to each input data point, such as an image represented by a matrix of RGB pixel values. DNNs are now widely used in computer vision because they make useful visual information more explicit: for instance, it's hard or impossible to tell what type of object is in an image by training a linear classifier on its pixels or even on the outputs of a shallow neural network; but DNNs can be optimized to perform this sort of "difficult" visual task. 

Remarkably, the visual system of our brain seems to use similar algorithms for performing these tasks. Like DNNs, visual cortex consists of a series of "layers" -- in this case, cortical areas -- through which visual information passes. (In fact, the architecture of DNNs was inspired by the neuroanatomy of primate visual cortex.) The action potential firing patterns of neurons in each of these cortical areas can be recorded while an animal is looking at images. Visual neuroscientists have been trying for decades to explain what computations these neurons and cortical areas are doing to allow primates to perform a rich repertoire of behaviors, such as recognizing faces and objects. While they've had some success (for instance in comparing the "early" layers of visual cortex to edge-detection algorithms), neural responses in intermediate and higher visual cortex have been particularly inscrutable -- before 2014, we had no good way of predicting how a neuron in higher visual cortex would respond to a given image. 

However, just as the development of DNNs led to a major jump in success at computer vision tasks, these algorithms turned out to be far-and-away the best at explaining
neural responses in visual cortex. When shown identical images, the artificial "neurons" in late layers, intermediate, and early layers of DNNs closely resemble the firing patterns of real neurons in late, intermediate, and early visual cortex. (This resemblance is made quantitatively precise by linearly regressing each real neuron on the activation patterns of the DNN "neurons.") This correspondence suggests a general hypothesis about how our visual system processes input: like a DNN, it transforms inputs through a sequence of linear and nonlinear computations, and it has been optimized so that its outputs can be used to perform difficult visual behaviors. 

So DNNs and visual cortex seem to share both architecture and function. But this raises another question: do they also share their mechanism of optimization, i.e. a "learning rule?" How do the synapses between neurons in visual cortex become optimized (through a combination of evolution and development) to perform behaviors? 

DNNs are typically optimized by stochastic gradient descent on the "weights" of the connections between layers of artificial neurons, which are meant to stand in for synapses between neurons in the brain. Given a set of images and an error function for a particular behavior (such as categorizing objects in each image), the parameters of a DNN are repeatedly updated to minimize the error. Mathematically, this is just the chain rule: calculating how the error depends on the value of each parameter. Since the error is computed based on the output of the network, computing gradients of the error with respect to parameters in the early layers involves "backpropagating" through the intervening layers. Hence the name "Backprop." 

Backprop has proven wildly successful at optimizing DNNs. It's therefore an obvious candidate for the learning rule that modifies the strength of synapses in visual cortex. However, running the Backprop learning algorithm requires that artificial neurons in early layers "know" the weights of connections between deeper layers (a specific example: in a two-layer network, the gradient of the error with respect to the first connection weight matrix would depend on the transpose of the second connection weight matrix). This makes neurobiologists worried, because we don't know of a way that some neurons might convey information about the synapse strength of other neurons. (This doesn't mean there isn't a way, we just don't know of one.) Theorists have tried to get around this by proposing that a second set of neurons sends learning signals from higher to lower layers of the brain's visual system. Anatomically, there are plenty of neural connections from later to earlier visual cortex. But we don't know if the signals they send are consistent with what they'd need to if they were implementing Backprop. If they were consistent, it would mean that the transpose of the forward synaptic strengths were somehow encoded in the firing patterns of these backward connections. So how would this set of neurons get access to the synaptic properties of a physically distinct set of neurons? 

I think your result offers a potential solution. Imagine that in addition to the usual neurons the compute the "forward pass" of some visual computation, there's an additional set of neurons at each "layer" of the network (i.e. in each area of visual cortex.) These neurons fire in response to inputs from higher layers. So now for every set of connections from layer L to layer L+1, you have a different set of connections from layer L+1 to layer L. In general there need not be any relationship between the forward and the backward weights. But in the special case where they are transposes of each other, this new set of neurons receiving input from layer L+1 will encode the error signal needed to perform Backprop. 

It seems like you can now set up a single neural network where this transpose condition is guaranteed to hold. First build your ordinary DNN. Then at each layer, add an equal number of new artificial neurons. These new neurons will receive connections from the original set of neurons at layer L+1. During training, you will minimize both the usual error function (which depends directly only on the output layer) and the reconstruction error, at each layer, between the old neurons and the new neurons. Now the original neurons at layer L+1 are playing the role of the latent state of an autoencoder between the original and the new neurons at layer L. If you apply an L2 regularization to the forward and backward weights then your theorem seems to guarantee that the forward and backward weights will be transposes. Now you no longer need to use the Backpropagation algorithm to update the weights at each layer: at each weight update for layer L, you apply the changes that the Backprop formula tells you to apply, but now it's only a function of the forward and backward inputs to that layer. It's local, and doesn't require neurons to "know about" the synaptic connections in other parts of the network.

This is a relatively minor thing, but L2 regularization also has a straightforward biological interpretation: in the network I just described, it penalizes synapse strength for being large. Synaptic transmission is energy intensive due to membrane vesicle fusion and production, as well as maintaining transmembrane ion gradients and pumping released neurotransmitter back into the synaptic terminal and vesicles. Stronger synapses require all of these things, so are more energy intensive. I'm not sure you need an energetics rationale for applying L2 regularization because it appears to play a computational role, but it could be both. In any case, it's standard to apply L2 regularization to all weights during DNN training.
