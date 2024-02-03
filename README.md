# Adapters are end-to-end probes



Most LLM probes train a linear classifier on top of the LLM residual stream. Or a sparse autoencoder on the LLM's residual stream. But what if we use an adapter, such as LoRA, isntead. Instead of jus training of the hidden states, we train it using end-to-end backpropagation. How does this work? How does it generalise?

See the branches for some of my experiments. 

Styilized facts:
- using an adapter as an importance matrix in a SAE does not seem to help
- using the activations from adapters as counterfactual residual streams do not seem to help significantly
- using Sparse Autoencoders or VQ-VAE (tokenized autoencoders) does not significantly help in this case (although I think VQ-VAE interpreatility project may hold great promise)

Future work:
- I've been using Phi-2 on datasets where it gets the wrong answer. To progress this I think I need a more reliable and natural way to produce deception that I can measure
