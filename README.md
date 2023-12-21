# Adapters are end-to-end probes



Most LLM probes train a linear classifier on top of the LLM residual stream. Or a sparse autoencoder on the LLM's residual stream. But what if we use an adapter, such as LoRA, isntead. Instead of jus training of the hidden states, we train it using end-to-end backpropagation. How does this work? How does it generalise?
