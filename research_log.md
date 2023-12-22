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


- ... so don't use lightning
- or use Loraq https://github.com/huggingface/peft/blob/46a84bd395f1b486b7b076acfa8f6df3dfad26b8/examples/loftq_finetuning/README.md?plain=1#L2 
- or work out how to use butsanybytes during train
- or gradient accum?


# 2023-12-19 06:58:21

Oh I can have a batch of 2 is I have 1 shot and a smaller prompt length!


Change my choice code to just go 
```py
batch_size = 3
tokens = 102
logits = torch.rand(batch_size,tokens)
token_choices = torch.randint(tokens, (batch_size, 2))

# Create a batch range
batch_range = torch.arange(batch_size).unsqueeze(1)

# Select the tokens
selected_logits = logits[batch_range, token_choices]
selected_logits
```

#  2023-12-20 07:47:59

I was working.... but not with some of the changes loss is always inf... I need to find the change

Could be
- new selection logic?
- disable adapter in a diff way
- package version?

# 2023-12-21 07:42:11 

Other dapters.

see [notes[(/home/wassname/Documents/syncthing/Markdown_notes/2023/12/21/llm_adapters.md)

# 2023-12-21 08:42:37 tiny models?

which tiny models in 2023-12-20? 


| Model     | Size | boolq base | chat CommonReason (ARC_c)| instruct code (HumanEval) |  Notes |
| ---       | ---  | ---   | --- |--- |--- |
| Pythia-1B | 1.0B |  69.83  | 48.3 | 1.83 | . |
| tinyllama | 1.1B | 63.1 | 53.86| 9.5 |  https://github.com/jzhang38/TinyLlama/blob/main/EVAL.md |
| phi-1.5   | 1.3B | 75.8 | 44.9 | ~35 | https://arxiv.org/abs/2309.05463 |
| phi-2     | 2.7B | 83.3 | ~53 | ~48 | not arxiv yet https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/ |


# why inf

- larger batch? no
- that bug in choices? no
- 4bit? no... in fact it fixed it. wtf!!

# batch sizes?

| model | quant | batch | gpu ram |
| --- | --- | --- | --- |
| phi-1_5 | bnb=4bit | 3 | 21g |

note 16-true fails, 4bit helps?


lightning options:
- accelerator: gpu, or accelerate (seems to conflict with bnb)
- precision, bf16-true (fails),bf16-mixed (uses more ram, likely because it undoes bnb)


# can I modify peft to just do bias?

https://github.com/huggingface/transformers/blob/cf32c941350cb296e4c2c9e26a9274291d515e90/src/transformers/integrations/peft.py#L268
https://github.dev/huggingface/peft


Tried phi-1.5 and it got 55% acc, low.
Phi-2 52%, wtf
with 2 shot it was 68%... OK I see the problem

Now try phi-2. Maybe I need instruction tuned. Or more shots
Or filter out unkown ones?
Or try to not make a flip model, but a lying model.
Or more epochs?
