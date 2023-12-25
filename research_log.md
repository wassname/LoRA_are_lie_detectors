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

- [x] Now try phi-2. Maybe I need instruction tuned. Or more shots
- [x] Or filter out unkown ones?
- [x] Or try to not make a flip model, but a lying model.
- [x] Or more epochs?


# model acc

does the prompt format matter?
- phi 96%
- alpaca 84%
- chatml 60%

hmm
{'Walmart-the-bag/phi-2-uncensored': 0.7304964539007093,
 'Yhyu13/phi-2-sft-alpaca_gpt4_en-ep1': 0.7872340425531915,
 'wassname/phi-2-w_hidden_states': 0.7659574468085106,
 'wassname/phi-1_5-w_hidden_states': 0.6312056737588653}

for Walmart-the-bag/phi-2-uncensored:
with base model
	balance=	51.41% [N=284]
	acc    =	73.05% [N=141]      - when the model is not lying... we get this task acc
	lie_acc=	35.66% [N=143]      - when the model tries to lie... we get this acc
	known_lie_acc=	25.71% [N=70]      - when the model tries to lie and knows the answer... we get this acc
	choice_cov=	87.37%             - Our choices accounted for a mean probability of this

for Yhyu13/phi-2-sft-alpaca_gpt4_en-ep1:
with base model
	balance=	51.41% [N=284]
	acc    =	78.72% [N=141]      - when the model is not lying... we get this task acc
	lie_acc=	27.97% [N=143]      - when the model tries to lie... we get this acc
	known_lie_acc=	27.06% [N=85]      - when the model tries to lie and knows the answer... we get this acc
	choice_cov=	86.58%             - Our choices accounted for a mean probability of this

for wassname/phi-2-w_hidden_states:
with base model
	balance=	51.41% [N=284]
	acc    =	76.60% [N=141]      - when the model is not lying... we get this task acc
	lie_acc=	28.67% [N=143]      - when the model tries to lie... we get this acc
	known_lie_acc=	26.58% [N=79]      - when the model tries to lie and knows the answer... we get this acc
	choice_cov=	83.37%             - Our choices accounted for a mean probability of this
wassname/phi-1_5-w_hidden_states

for wassname/phi-1_5-w_hidden_states:
with base model
	balance=	51.41% [N=284]
	acc    =	63.12% [N=141]      - when the model is not lying... we get this task acc
	lie_acc=	39.16% [N=143]      - when the model tries to lie... we get this acc
	known_lie_acc=	36.07% [N=61]      - when the model tries to lie and knows the answer... we get this acc
	choice_cov=	75.41%             - Our choices accounted for a mean probability of this

# 2023-12-22 17:04:52

Experiments
- I tried unsensored/chat/instruction tuned models... not much difference
- I tried training the adapter to
  - always say the opposite (meh)
  - always lie (helped a bit)
  - always lie when told to/demonstrated to (pending)
  

bugs :bug: ... wait are my labels right

and am I using ans right... lets check
- well it comes from binary_ans_adapt (score)
- which is from postprocess result. 
- binary_ans = choice_probs[:, 1] / (choice_probs.sum(1) + 1e-12)
- So! it's just a boolean prediction, not correctness


So to get ranking I need to flip it...

# 2023-12-22 17:45:54 need to check intervention LR's

OK fixed the label. Still poor.
maybe training wkv? no

- [ ] Maybe I need to fix the LR? Maybe use neels 1 layer linear??
- [ ] Maybe I need soft labels?
- [x] Maybe I need to try it, in sample .e.g. val set. then I might find I  just need more daya to gen
- [x] Maybe keep it just to rows it knows. helpts a lot
- [x] maybe no scaling? sometimes helps, sometimes doesn't

# 2023-12-23 10:05:31

Hmm if I filter it to known I do get some 90%'s, promising!

Wait it was a one off, curious

Need to rerun the other 2 experiments

# 2023-12-23 12:49:27 peft


models https://github.dev/huggingface/peft/blob/993836ff90791289b94d27caa46385eec958e147/src/peft/mapping.py#L82


# 2023-12-23 15:35:05

- [x] Now try phi-2. Maybe I need instruction tuned. Or more shots
- [x] Or filter out unkown ones?
- [x] Or try to not make a flip model, but a lying model.
- [x] Or more epochs?


- [x] Maybe I need to try it, in sample .e.g. val set. then I might find I  just need more daya to gen
- [x] Maybe keep it just to rows it knows. helpts a lot
- [x] maybe no scaling? sometimes helps, sometimes doesn't
- [ ] Maybe I need to fix the LR? Maybe use neels 1 layer linear??
  - [ ] https://github.com/lingo-mit/lm-truthfulness/blob/master/lm_truthfulness_gpt-j.ipynb
- [ ] Maybe I need soft labels?
- [ ] larger batch? 

- [ ] maybe it's not understandable to a lienar probe. I've made _different_ hidden states, but maybe not understandable to a linear regression. So maybe a non linear probe on top? Can try a VAE another day.
  - [ ] make some plugin probe. linear, my conv one, bayesian
  - [ ] https://github.com/pytorch/botorch/issues/725
    - [ ] https://mapie.readthedocs.io/en/latest/examples_regression/4-tutorials/plot_cqr_tutorial.html


pytorch probe
probe with uncertainty taken from that paper


Idea: remove system prompts! it still works. few shot examples do seem better
what about if I take away the instructions. train a adapter that lies when demonstrated. ?
- hmm I get 90% auc, but I get that as baseline or probe. 


new datasets

sciq?

or uncertainty like https://github.com/google-research/gpax

https://arxiv.org/pdf/2305.18213.pdf

it would be cool to use this https://github.com/pytorch/botorch

#@

# binary uncertainty?

https://mapie.readthedocs.io/en/latest/theoretical_description_binary_classification.html

https://github.com/cornellius-gp/gpytorch/blob/master/examples/04_Variational_and_Approximate_GPs/Non_Gaussian_Likelihoods.ipynb


# 2023-12-24 19:42:37

So in this experiment 
notebooks/03_mjc_ltold_to_lie_loss.ipynb
7776d2639eb95dd89fe933983cb7839f368e7b52

I trained a probe that lied, while the base model told the truth. But it didn't extend to the OOD set!

And the intervention hidden states didn't hepl much on the train/val set. The base hidden states seemed to be enougth.

Hmm. there are even cases where the base hidden states help predict if the adapter will lie, and of course it almost always does. Confusing

Oh but the model does have really poor performance on the OOD set. Hmm so maybe it's not a good example?
Or maybe I have a bug...

The basics of my approach is that the intervention should be good. but the intervention doesn't seem to add much. Hmm

Hypothesis:
- the information isn't there because 1) the model is too small or 2) the hidden states are no the right way to look at it or 

what do I mean by adapters acting like probes. Sure we can train them, take the hidden states, and use it with logistic regression to get an answer. But the normal way is to just let the adaptor give the answer. Well do think treating them as a probe is better. But I need to prove it. Perhaps even make a linear adapter. This can be a building block even if it doesn't unlock lie detection right away.

**Hypotheis**: probe acc is greater than adapter acc. And even OOD.

wait shoulnd't auc predictive be the same for baseline and intervention if I am only using the baseline hidden states??

So experiment results:
- acc of base model 0.824639
- acc of adapter 0.62 (more lies)
- just hidden states of base: 0.886339
- hidden states of both 0.908748 (slightly better but not much)


But wait I need to train for truth telling...

Experiment notebooks/04_mjc_truth.ipynb at commit ed6343344ea63f7dc5fd35fc5958501d2ef07278

base model task (truth auroc) 0.64 val, 0.69 OOD
adapter acc 0.71 val (better), 0.65 OOD (worse)
base model residual LR 0.89 val, 0.917
adapter residual LR 0.905 (better), 0.974 (better)
non linear probe 0.96 auroc

So yes! Hypothesis confirmed


Hypothesis: Probes on adapter are better than either probes or adapters.

|model| val acc | OOD acc |
|--|--|--|
|base model  acc | 0.64  | 0.70 OOD |
|adapter acc | 0.71  (better) | 0.64 (worse)|
|base+probe model residual auroc | 0.89 | 0.917|
|adapter+probe residual auroc | **0.905** | **0.974 (better)**|

So yes! Hypothesis confirmed
mm


# 2023-12-25 19:11:42

Note there are probes of CCS here https://github.com/saprmarks/geometry-of-truth/blob/91b223224699754efe83bbd3cae04d434dda0760/probes.py#L75
