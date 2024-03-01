# Adapters are end-to-end probes


Typically, most Language Learning Model (LLM) probes train a linear classifier on the LLM's residual stream or use a sparse autoencoder. However, an alternative approach is to utilize an adapter, such as LoRA. Instead of only training the hidden states, this method involves end-to-end backpropagation. The key questions are: How does this function? How well does it generalize?

Refer to the branches for details on my experiments.

Stylized Facts:

- Implementing an adapter as an importance matrix in a Sparse Autoencoder (SAE) does not seem beneficial.
- Utilizing the activations from adapters as counterfactual residual streams does not significantly improve results.
- The use of Sparse Autoencoders or VQ-VAE (tokenized autoencoders) does not noticeably enhance the outcome in this context (although the VQ-VAE interpretability project appears promising).

Future Work:

- I've been applying Phi-2 on datasets where it returns incorrect answers. To advance this, I believe a more reliable and natural method to generate and measure deception is necessary.

Related work:
- https://github.com/wassname/discovering_latent_knowledge
