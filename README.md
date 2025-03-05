# Knowledge Graph Completion with Mixed Geometry Tensor Factorization

Abstract: In this paper, we propose a new geometric approach for knowledge graph completion via low rank tensor approximation. We augment a pretrained and well-established Euclidean model based on a Tucker tensor decomposition with a novel hyperbolic interaction term. This correction enables more nuanced capturing of distributional properties in data better aligned with real-world knowledge graphs. By combining two geometries together, our approach improves expressivity of the resulting model achieving new state-of-the-art link prediction accuracy with a significantly lower number of parameters compared to the previous Euclidean and hyperbolic models.

We evaluate our mixed geometry model for link prediction task on three standard knowledge graphs: WN18RR, FB15k-237 and YAGO3-10. We compute metrics: HR@10, HR@3, HR@1, MRR.

Versions of python packages:

numpy $1.26.4$

torch $2.1.2+cpu$


