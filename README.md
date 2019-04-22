# Loss Landscapes of Regularized Linear Autoencoders
##### [Daniel Kunin](http://daniel-kunin.com/), [Jonathan M. Bloom](https://www.broadinstitute.org/bios/jonathan-bloom), [Aleksandrina Goeva](https://tudaga.github.io/), [Cotton Seed](https://www.broadinstitute.org/bios/cotton-seed)

Autoencoders are a deep learning model for representation learning. When trained to minimize the Euclidean distance between the data and its reconstruction, linear autoencoders (LAEs) learn the subspace spanned by the top principal directions but cannot learn the principal directions themselves. In this paper, we prove that L2-regularized LAEs learn the principal directions as the left singular vectors of the decoder, providing an extremely simple and scalable algorithm for rank-k SVD. More generally, we consider LAEs with (i) no regularization, (ii) regularization of the composition of the encoder and decoder, and (iii) regularization of the encoder and decoder separately. We relate the minimum of (iii) to the MAP estimate of probabilistic PCA and show that for all critical points the encoder and decoder are transposes. Building on topological intuition, we smoothly parameterize the critical manifolds for all three losses via a novel unified framework and illustrate these results empirically. Overall, this work clarifies the relationship between autoencoders and Bayesian models and between regularization and orthogonality. Most excitingly, it suggests a resolution to the "weight transport problem" (Grossberg 1987) of computational neuroscience, providing local learning rules by which maximizing information flow and minimizing synaptic weights give rise to less-biologically-implausible analogues of backproprogation in the brain.

[Preprint here](https://arxiv.org/pdf/1901.08168.pdf), v2 coming soon, feedback welcome! Contact Daniel and Jon: kunin@stanford.edu, jbloom@broadinstitute.org

To appear at [ICML 2019](https://icml.cc/Conferences/2019). Scroll or click these links for more!

[Talks](https://github.com/danielkunin/Regularized-Linear-Autoencoders#talks), [Code](https://github.com/danielkunin/Regularized-Linear-Autoencoders/blob/master/Loss%20Landscapes%20of%20Regularized%20Linear%20Autoencoders%20Code.ipynb), [Algorithms](https://github.com/danielkunin/Regularized-Linear-Autoencoders#lae-pca-algorithms), [Visualization](https://danielkunin.github.io/Regularized-Linear-Autoencoders/), [Neuroscience](https://github.com/danielkunin/Regularized-Linear-Autoencoders#computational-neuroscience), [Resources](https://github.com/danielkunin/Regularized-Linear-Autoencoders#useful-resources), [History](https://github.com/danielkunin/Regularized-Linear-Autoencoders#project-history), [Clarification and Erratum](https://github.com/danielkunin/Regularized-Linear-Autoencoders/blob/master/erratum.md).

### Talks

We're excited to share a primer and seminar given by all four authors at the [Models, Inference and Algorithms Meeting](https://broadinstitute.org/mia) at the [Broad Institute of MIT and Harvard](https://www.broadinstitute.org/) on February 27, 2019.

<p align="center">
    <a href="https://www.youtube.com/watch?v=1mvwCmqd6zQ&list=PLlMMtlgw6qNjROoMNTBQjAcdx53kV50cS"><img src="/img/primer.png" alt="MIA Primer" width="400"/></a>
    <a href="https://www.youtube.com/watch?v=1mvwCmqd6zQ&list=PLlMMtlgw6qNjROoMNTBQjAcdx53kV50cS&t=3480s"><img src="/img/seminar.png" alt="MIA Seminar" width="400"/></a>
</p>

We went deeper into Morse theory and ensemble learning at [Google Brain](https://ai.google/research/teams/brain) on March 6, 2019. Video below left, slides [here](https://drive.google.com/file/d/1wNghE5-0JnHOYANBQMUsPWS1n3tmRJKA/view?usp=sharing).

We discussed our latest computational neuroscience results at the [Center for Brains, Minds, and Machines](https://cbmm.mit.edu/news-events/events/brains-minds-machines-seminar-series-topology-representation-teleportation) at MIT on April 2, 2019. Video below right, slides [here](https://drive.google.com/file/d/1seI05mrQmosuOHoO2Jgn-cK1ZyCrDlnP/view?usp=sharing), and a 20m post-talk interview with zero ferns [here](https://www.youtube.com/watch?v=b0qnMVKRJVM).

<p align="center">
    <a href="https://www.youtube.com/watch?v=3aqB_n087cE&list=PLlMMtlgw6qNjROoMNTBQjAcdx53kV50cS"><img src="https://img.youtube.com/vi/3aqB_n087cE/0.jpg" alt="Google Brain" width="400"/></a>
    <a href="https://www.youtube.com/watch?v=bVlzJZIH4vs&list=PLlMMtlgw6qNjROoMNTBQjAcdx53kV50cS"><img src="https://img.youtube.com/vi/bVlzJZIH4vs/0.jpg" alt="Center for Brains Minds + Machines" width="400"/></a>
</p>

### LAE-PCA Algorithms

Here's a simple gradient descent algorithm for LAE-PCA with **untied** weights (i.e. there is no constraint on `W1` or `W2`):

```python
XXt = X @ X.T
while np.linalg.norm(W1 - W2.T) > epsilon:
    W1 -= alpha * ((W2.T @ (W2 @ W1 - I)) @ XXt + lamb * W1)
    W2 -= alpha * (((W2 @ W1 - I) @ XXt) @ W1.T + lamb * W2)

principal_directions, s,  _ = np.linalg.svd(W2, full_matrices = False)
eigenvalues = np.sqrt(lamb / (1 - s**2))
```

This may be accelerated on frameworks like TensorFlow using a host of math, hardware, sampling, and deep learning tricks, leveraging our topological and geometric understanding of the loss landscape. For example, one can alternate between **exact** convex minimization with respect to `W1` fixing `W2` and then with respect to `W2` fixing `W1`. Or one can **synchronize** `W1` and `W2` to be transposes at initialization. Or one can constrain, or tie, `W2 = W1.T` *a priori* (see Appendix A):

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

The four variants of the LAE-PCA algorithm described above and runnable examples comparing their convergence rates are [here](https://github.com/danielkunin/Regularized-Linear-Autoencoders/blob/master/algorithms.py).

<p align="center">
    <img src="/img/time.png" alt="error per sec" width="400"/>
    <img src="/img/iteration.png" alt="error per iteration" width="400"/>
</p>

We can visualize the trajectories of these algorithm in the `m = k = 1` (scalar) case with code provided [here](https://github.com/danielkunin/Regularized-Linear-Autoencoders/blob/master/algorithms_visualize.py).

<p align="center">
    <img src="/img/untied.gif" alt="untied algorithm in scalar case" width="400"/>
    <img src="/img/exact.gif" alt="exact algorithm in scalar case" width="400"/>
</p>

### Visualization

We've created an [interactive visualization](https://danielkunin.github.io/Regularized-Linear-Autoencoders/) of the three loss landscapes and more in the `m = k = 1` (scalar) case.

<p align="center">
    <a href="https://danielkunin.github.io/Regularized-Linear-Autoencoders/">
        <img src="/img/visualization.gif" alt="visualization demo" width="400"/>
    </a>
</p>

### Computational Neuroscience

See Jon's [talk](https://www.youtube.com/watch?v=bVlzJZIH4vs&list=PLlMMtlgw6qNjROoMNTBQjAcdx53kV50cS) and [interview](https://www.youtube.com/watch?v=b0qnMVKRJVM) at the Center for Brains, Minds and Machines for our progress on the computational neuroscience front as of April 2, 2019. Here's the history of how this theoretical project got us thinking about learning in the brain:

In response to v1 of the preprint, [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/) wrote us on February 10, 2019 that "the question of obtaining the transpose is actually pretty important for research on a biologically plausible version of backprop, because if you obtain approximate transposes, then several local learning rules give rise to gradient estimator analogues of backprop." He pointed us to his related papers [Towards Biologically Plausible Deep Learning](https://arxiv.org/abs/1502.04156) and [Equivalence of Equilibrium Propagation and Recurrent Backpropagation](https://arxiv.org/abs/1711.08416).

We also found [How Important Is Weight Symmetry in Backpropagation?](https://arxiv.org/abs/1510.05067) by Poggio, et. al., which calls the "weight symmetry problem [...] arguably the crux of BP’s biological implausibility," as well as the 2018 review [Theories of Error Back-Propagation in the Brain](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(19)30012-9) which lists weight symmetry as the top outstanding question, going back to [Stephen Grossberg, 1987](https://www.sciencedirect.com/science/article/abs/pii/S0364021387800253) and [Francis Crick, 1989](https://www.nature.com/articles/337129a0). Discussing the issue with neuroscientist [Dan Bear](http://neuroailab.stanford.edu/people.html), we believe our work provides a simple, scalable solution to the weight symmetry problem via emergent dynamics.

Dan was kind enough to elaborate on this perspective as follows:

Deep Neural Networks (DNNs) are algorithms that apply a sequence of linear and nonlinear transformations to each input data point, such as an image represented by a matrix of RGB pixel values. DNNs are now widely used in computer vision because they make useful visual information more explicit: for instance, it's hard or impossible to tell what type of object is in an image by training a linear classifier on its pixels or even on the outputs of a shallow neural network; but DNNs can be optimized to perform this sort of "difficult" visual task. 

Remarkably, the visual system of our brain seems to use similar algorithms for performing these tasks. Like DNNs, visual cortex consists of a series of "layers" -- in this case, cortical areas -- through which visual information passes. (In fact, the architecture of DNNs was inspired by the neuroanatomy of primate visual cortex.) The action potential firing patterns of neurons in each of these cortical areas can be recorded while an animal is looking at images. Visual neuroscientists have been trying for decades to explain what computations these neurons and cortical areas are doing to allow primates to perform a rich repertoire of behaviors, such as recognizing faces and objects. While they've had some success (for instance in comparing the "early" layers of visual cortex to edge-detection algorithms), neural responses in intermediate and higher visual cortex have been particularly inscrutable -- before 2014, we had no good way of predicting how a neuron in higher visual cortex would respond to a given image. 

However, just as the development of DNNs led to a major jump in success at computer vision tasks, these algorithms turned out to be far-and-away the best at explaining
neural responses in visual cortex. When shown identical images, the artificial "neurons" in late layers, intermediate, and early layers of DNNs closely resemble the firing patterns of real neurons in late, intermediate, and early visual cortex. (This resemblance is made quantitatively precise by linearly regressing each real neuron on the activation patterns of the DNN "neurons.") This correspondence suggests a general hypothesis about how our visual system processes input: like a DNN, it transforms inputs through a sequence of linear and nonlinear computations, and it has been optimized so that its outputs can be used to perform difficult visual behaviors. 

So DNNs and visual cortex seem to share both architecture and function. But this raises another question: do they also share their mechanism of optimization, i.e. a "learning rule?" How do the synapses between neurons in visual cortex become optimized (through a combination of evolution and development) to perform behaviors? 

DNNs are typically optimized by stochastic gradient descent on the "weights" of the connections between layers of artificial neurons, which are meant to stand in for synapses between neurons in the brain. Given a set of images and an error function for a particular behavior (such as categorizing objects in each image), the parameters of a DNN are repeatedly updated to minimize the error. Mathematically, this is just the chain rule: calculating how the error depends on the value of each parameter. Since the error is computed based on the output of the network, computing gradients of the error with respect to parameters in the early layers involves "backpropagating" through the intervening layers. Hence the name "Backprop." 

Backprop has proven wildly successful at optimizing DNNs. It's therefore an obvious candidate for the learning rule that modifies the strength of synapses in visual cortex. However, running the Backprop learning algorithm requires that artificial neurons in early layers "know" the weights of connections between deeper layers (a specific example: in a two-layer network, the gradient of the error with respect to the first connection weight matrix would depend on the transpose of the second connection weight matrix). This makes neurobiologists worried, because we don't know of a way that some neurons might convey information about the synapse strength of other neurons. (This doesn't mean there isn't a way, we just don't know of one.) Theorists have tried to get around this by proposing that a second set of neurons sends learning signals from higher to lower layers of the brain's visual system. Anatomically, there are plenty of neural connections from later to earlier visual cortex. But we don't know if the signals they send are consistent with what they'd need to if they were implementing Backprop. If they were consistent, it would mean that the transpose of the forward synaptic strengths were somehow encoded in the firing patterns of these backward connections. So how would this set of neurons get access to the synaptic properties of a physically distinct set of neurons? 

I think your result offers a potential solution. Imagine that in addition to the usual neurons the compute the "forward pass" of some visual computation, there's an additional set of neurons at each "layer" of the network (i.e. in each area of visual cortex.) These neurons fire in response to inputs from higher layers. So now for every set of connections from layer L to layer L+1, you have a different set of connections from layer L+1 to layer L. In general there need not be any relationship between the forward and the backward weights. But in the special case where they are transposes of each other, this new set of neurons receiving input from layer L+1 will encode the error signal needed to perform Backprop. 

It seems like you can now set up a single neural network where this transpose condition is guaranteed to hold. First build your ordinary DNN. Then at each layer, add an equal number of new artificial neurons. These new neurons will receive connections from the original set of neurons at layer L+1. During training, you will minimize both the usual error function (which depends directly only on the output layer) and the reconstruction error, at each layer, between the old neurons and the new neurons. Now the original neurons at layer L+1 are playing the role of the latent state of an autoencoder between the original and the new neurons at layer L. If you apply an L2 regularization to the forward and backward weights then your theorem seems to guarantee that the forward and backward weights will be transposes. Now you no longer need to use the Backpropagation algorithm to update the weights at each layer: at each weight update for layer L, you apply the changes that the Backprop formula tells you to apply, but now it's only a function of the forward and backward inputs to that layer. It's local, and doesn't require neurons to "know about" the synaptic connections in other parts of the network.

L2 regularization also has several possible biological interpretations: in the network I just described, it penalizes synapse strength for being large. Synaptic transmission is energy intensive due to membrane vesicle fusion and production, as well as maintaining transmembrane ion gradients and pumping released neurotransmitter back into the synaptic terminal and vesicles. Stronger synapses require all of these things, so are more energy intensive. Thus, the L2 penalty on synapse strength could be interpreted as an energy cost on the physical building blocks that constitute a functional synapse.

Here is a second interpretation: during training of a neural network, L2 regularization on the weights will cause them to decay exponentially to zero (unless updated due to the other error functions.) In the network I proposed above with L2-regularized linear autoencoders operating between each pair of adjacent layers, the layer-wise reconstruction errors were assumed to be minimized at the same time as the task-related error. But this doesn't need to be the case either in deep learning or in the brain. The reconstruction error could be minimized as part of a faster "inner loop" of optimization, implemented by the local recurrent circuits ubiquitous in cortex or by biophysical mechanisms operating in individual neurons. In this scenario, L2 regularization on the faster timescale could stand in for another well-known neurophysiological phenomenon: the short-term depression of synaptic strength when a neuron fires many action potentials in a short time window, which results from depletion of readily releasable synaptic vesicles and from modifications to synaptic ion channels.

Regularization may therefore play a crucial role in shaping the computations that neural systems perform. A well-known example of this idea in computational neuroscience is the result of Olshausen and Field (Nature 1996). They showed that an autoencoder with an L1 regularization penalty on the activations of the latent state could explain one of the most robust findings in visual neuroscience, the preferential response of primary visual cortical neurons to oriented gratings. The L1 regularization was crucial for producing this effect, which the authors thought of as a "sparse" encoding of natural images. Other forms of biophysical processes have been proposed to explain characteristic features of retinal neuron responses, as well (Ozuysal and Baccus, Neuron 2012.) In biology, the features of any system must be explained as the result of evolutionary constraints. Linking biophysical mechanisms to neural computation is a promising road to understanding the intricacies of the brain.

Reading on DNNs and the primate visual system:
https://papers.nips.cc/paper/7775-task-driven-convolutional-recurrent-models-of-the-visual-system.pdf
https://www.nature.com/articles/nn.4244

Clear explanation of backpropogation and why it requires the transpose of the forward weight matrix:
http://neuralnetworksanddeeplearning.com/chap2.html

Olshausen and Field:
https://www.nature.com/articles/381607a0

### Useful Resources

Wikipedia has articles on [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition), [autoencoders](https://en.wikipedia.org/wiki/Autoencoder), [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)), [orthogonality](https://en.wikipedia.org/wiki/Orthogonal_matrix), [definiteness](https://en.wikipedia.org/wiki/Definiteness_of_a_matrix), [matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus), [gradient flow](https://en.wikipedia.org/wiki/Vector_field#Gradient_field), [topology](https://en.wikipedia.org/wiki/Topology), [manifolds](https://en.wikipedia.org/wiki/Manifold), [Grassmannians](https://en.wikipedia.org/wiki/Grassmannian), [CW complexes](https://en.wikipedia.org/wiki/CW_complex), [algebraic topology](https://en.wikipedia.org/wiki/Algebraic_topology), [homology](https://en.wikipedia.org/wiki/Homology_(mathematics)), [cellular homology](https://en.wikipedia.org/wiki/Cellular_homology), [Morse theory](https://en.wikipedia.org/wiki/Morse_theory), [Morse homology](https://en.wikipedia.org/wiki/Morse_homology), and [random matrix theory](https://en.wikipedia.org/wiki/Random_matrix).

For a rigorous treatments of manifolds and Morse theory, see Jon's [review](http://www.math.harvard.edu/theses/phd/bloom/ThesisXFinal.pdf), Milnor's classic [Morse Theory](https://www.maths.ed.ac.uk/~v1ranick/papers/milnmors.pdf), Banyaga and Hurtubise's [Lectures on Morse Homology](https://www.springer.com/us/book/9781402026959), and Chen's [history](https://math.berkeley.edu/~alanw/240papers03/chen.pdf) of the field through [Morse](https://en.wikipedia.org/wiki/Marston_Morse), [Thom](https://en.wikipedia.org/wiki/Ren%C3%A9_Thom), [Smale](https://en.wikipedia.org/wiki/Stephen_Smale), [Milnor](https://en.wikipedia.org/wiki/John_Milnor) and [Witten](https://en.wikipedia.org/wiki/Edward_Witten). These works are complemented by the development of simplicial, singular, and CW homology in Hatcher's [Algebraic Topology](http://pi.math.cornell.edu/~hatcher/AT/AT.pdf). Jon fantasizes that [extending Morse homotopy](https://www.youtube.com/watch?v=9-echU1zIfI) to [manifolds with boundary](https://arxiv.org/abs/1212.6467) may one day be useful as well...

### Project History

Daniel interned with [Hail Team](https://hail.is/about.html) at the [Broad Institute of MIT and Harvard](https://broadinstitute.org) in Summer 2018. While the four of us were exploring models of deep matrix factorization for [single-cell RNA sequencing data](https://www.wired.com/story/the-human-cell-atlas-is-biologists-latest-grand-project/), Jon came across v1 of Elad Plaut's [
From Principal Subspaces to Principal Components with Linear Autoencoders](https://arxiv.org/abs/1804.10253) and contacted the author regarding a subtle error. Plaut agreed the main theorem was false as stated, and in correspondence convincingly re-demonstrated the existence of a fascinating empirical phenomenon. We realized quickly from Plaut's code and the scalar case that regularization was the key to symmetry-breaking and rigidity; the form of the general solution was also clear and simple to verify empirically. The bulk of our time was spent banging against chalkboards and whiteboards to establish additional cases and finally a unified proof in full generality.
