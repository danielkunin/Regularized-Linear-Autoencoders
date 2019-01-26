# Clarification and Erratum

### Project history

Daniel interned with [Hail Team](https://hail.is/about.html) at the Broad Institute of MIT and Harvard in Summer 2018. While exploring models of deep matrix factorization for [single-cell RNA sequencing data](https://www.wired.com/story/the-human-cell-atlas-is-biologists-latest-grand-project/), Jon came across v2 of Elad Plaut's [
From Principal Subspaces to Principal Components with Linear Autoencoders](https://arxiv.org/abs/1804.10253) and contacted the author regarding a subtle error. Plaut agreed the main theorem was false as stated, and in correspondence convincingly re-demonstrated the existance of a fascinating empirical phenomenon. We quickly realized from the scalar case that regularization was the key to symmetry-breaking and rigidity; the form of the general solution was also clear (and easy to verify empircally), as were proofs of additional special cases. The bulk of our time was spent banging against chalkboards and whiteboards to establish a proof that applied in full generality.

- In Section 1, we write "While the paper made no mention of it,
we realized by looking at the code that training was done with the
common practice of L2-regularization". While we did realize this by
looking at the code, in fact Plaut does mention in [Section IV](https://arxiv.org/abs/1804.10253) that
he found "weight decay regularization" to be beneficial. Our work
proves that it is necessary.

- In Footnote 1, we write "The principal directions of X are
the eigenvectors of the covariance of X." To clarify, we mean
the m x m covariance matrix between the rows of X.

- In Section 2.3, the first paragraph should conclude: "In fact,
L_2-regularization reduces the symmetry group from GL_k(R) to
O_k(R), making the [right] singular vectors of the encoder and
[left] singular vectors of the decoder well-defined."

- In Section 2.3, "(2)" refers to the second item in the list,
not equation (2).

- In Section 6 and Appendix B, we explain how the connectedness
of R^m guarantees that all minima are connected by gradient trajectories
through index-1 saddles. In fact, as observed empirically with [Fast Geometric Ensembling](https://arxiv.org/pdf/1802.10026.pdf),
these index-1 saddles are, with high probability, at a height close to
to that of the minima. This is due to the concentration of measure
phenomenon explored in [The Loss Landscapes of Multilayer Networds](https://arxiv.org/abs/1412.0233).

- In Appendix B, the final sentence should read: "And since R^m is
contractible, their method may in principle extend to finding
critical points of higher index that form a *contractible chain
complex*, flowing along gradient trajectories from one minimum
to all minima through index-1 saddles, from those index-1 saddles
to other index-1 saddles through index-2 saddles, and so on until
contractibility is satisfied. Note there may exist additional
critical points forming *null-homotopic chain complexes*."

- In Appendix B, the columns of the Morse index table should be labeled by the principal directions u_1, u_2, u_3, and u_4.
