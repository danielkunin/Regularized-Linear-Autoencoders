# Erratum and Clarification

- In Footnote 1, we write "The principal directions of X are
the eigenvectors of the covariance of X." To clarify, we mean
the m x m covariance matrix between the rows of X.

- In Section 2.3, the first paragraph should conclude: "In fact,
L_2-regularization reduces the symmetry group from GL_k(R) to
O_k(R), making the [right] singular vectors of the encoder and
[left] singular vectors of the decoder well-defined."

- In Section 2.3, "(2)" refers to the second item in the list,
not equation (2).

- In Section 5 and Appendix B, we explain how the connectedness
of R^m guarantees that all minima are connected by gradient trajectories
through index-1 saddles. In fact, as observed empirically with [Fast Geometric Ensembling](https://arxiv.org/pdf/1802.10026.pdf),
these index-1 saddles are, with high probability, at a height close to
to that of the minima. This is due to a concentration of measure
phenomenon explored in [The Loss Landscapes of Multilayer Networds](https://arxiv.org/abs/1412.0233).

- In Appendix B, the final sentence should read: "And since R^m is
contractible, their method may in principle extend to finding
critical points of higher index that form a *contractible chain
complex*, flowing along gradient trajectories from one minimum
to all minima through index-1 saddles, from those index-1 saddles
to other index-1 saddles through index-2 saddles, and so on until
contractibility is satisfied. Note there may exist additional
critical points forming *null-homotopic chain complexes*."
