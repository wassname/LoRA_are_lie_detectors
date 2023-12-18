# 2023-12-17 20:04:28

The idea is to use Lora as a probe. Only activated on the biases of the last layers or something like that.

This way we easily use backprop to chose the probe.

Then we test it for lie detection on a new dataset to check acc and generalizaion.

# 2023-12-17 20:05:09

Examples of peft


- jdp https://github.com/JD-P/minihf/blob/b54075c34ef88d9550e37fdf709e78e5a68787c4/lora_tune.py#L50


The idea is that we do something like this


```py
with torch.no_rgrad():
    probs = model(input)
    choice_probs = probs[choices]
    opposite_probs = choice_probs[::-1].detach()

    with adapter('intervention'):
        probs2 = model(input)
        choice_probs2 = probs2[choices]

        # or do we keep everything else the same?
        loss = kl_div(choice_probs2, opposite_probs)

        loss.backward()
        optimizer.step()
```


# 2023-12-18 07:30:23

I've having device problems. I probobly need to load a non quant version.

can;t install flash mem
can't train in 8bit mode?
bitsandbytes = "^0.41.3.post2" ?


and if I use 16 bit and my 25g gpu it's out of mem. I can't use a batch of one or lightnign fauils


- ... so don't use liughtning
- or use Loraq https://github.com/huggingface/peft/blob/46a84bd395f1b486b7b076acfa8f6df3dfad26b8/examples/loftq_finetuning/README.md?plain=1#L2 
- or work out how to use butsanybytes during train
- or gradient accum?
