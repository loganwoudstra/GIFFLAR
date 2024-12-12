# GIFFLAR - Glycan-Informed Foundational Framework to Learn Abstract Representations

Glycans are the most complex biological sequence, with monosaccharides forming extended, non-linear sequences. As post-translational modifications, they modulate protein structure, function, and interactions. Due to their diversity and complexity, predictive models of glycan properties and functions are still insufficient. Graph Neural Networks (GNNs) are deep learning models designed to process and analyze graph-structured data. These architectures leverage the connectivity and relational information in graphs to learn effective representations of nodes, edges, and entire graphs. Iteratively aggregating information from neighboring nodes, GNNs capture complex patterns within graph data, making them particularly well-suited for tasks such as link prediction or graph classification across domains. This work presents a new model architecture based on combinatorial complexes and higher-order message passing to extract features from glycan structures into a latent space representation. The architecture is evaluated on an improved GlycanML benchmark suite, establishing a new state-of-the-art performance. We envision that these improvements will spur further advances in computational glycosciences and reveal the roles of glycans in biology.

[arXiv](https://arxiv.org/abs/2409.13467)

### Installation:
```bash
conda create -n gifflar -y python=3.11
conda activate gifflar
pip install -r requirements.txt
```
