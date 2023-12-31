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

Wait why is ranking better? Does it make sense? Lets plot it

- [ ] TODO: Hmm I should be able to calculate the truth from any label I use right? Let's check that!!
  - [ ] I can just fork


~~Hmm I can undo that ranking label. But once I crunch the numbers... it'sn ot that accurate. It seems to distort the real objective. And if it doesn't help them perhaps my adapter > probe hypothesis is disproven.~~ 0.98->0.96 roc_auc... still good


# 2023-12-27 06:00:10

A nicer way to code the labels
```py
ds_train, ds_val = test_train_split(ds)
y_val = ds2proxy(ds_val)
y = proxy2label(y_val, ds_val)
```


Can i make lightning models with a sklearn like interface?
see
- https://github.com/huonfraser/scikit-TabTorch/blob/master/sklike_torch/torch_wrapper.py


TODO
- try pl sk api
- I need to flip my rankings....


Hmm wait I was filtering by known. That no longer makes sense when I am trying to exceed known. Really I need a better idea of known, but it's not possible to garuntee I'm extracting all the knowledge of a question from a model. Maybe if I use a lying adapter. But that's another exp.


finished the truth adapter vs probe experiment. 

the adapter generalized better
see
notebooks/10_compare_probes.ipynb
cfac3cdef758d66c42469bdd94c12d6cc23e1227

# PEFT modules?

UniPELTConfig The UniPELT adapter architecture proposed by Mao et al. (2022). See https://arxiv.org/pdf/2110.07577.pdf.


- https://github.com/adapter-hub/adapters/blob/main/src/adapters/configuration/adapter_config.py#L629C1-L652C32
	- # CURRENT STRINGS
	- "seq_bn": SeqBnConfig(),
	- "double_seq_bn": DoubleSeqBnConfig(),
	- "par_bn": ParBnConfig(),
	- "scaled_par_bn": ParBnConfig(scaling="learned"),
	- "seq_bn_inv": SeqBnInvConfig(),
	- "double_seq_bn_inv": DoubleSeqBnInvConfig(),
	- "compacter++": CompacterPlusPlusConfig(),
	- "compacter": CompacterConfig(),
	- "prefix_tuning": PrefixTuningConfig(),
	- "prefix_tuning_flat": PrefixTuningConfig(flat=True),
	- "prompt_tuning": PromptTuningConfig(),
	- "lora": LoRAConfig(),
	- "ia3": IA3Config(),
	- "mam": MAMConfig(),
	- "unipelt": UniPELTConfig(),
- peft https://github.dev/huggingface/peft
  - adalora - adaptive hyperparams for lora
  - ~~adaption prompt~~
  - ia3 - it's like lora 2.0
  - loha - Low-Rank Hadamard Product   https://arxiv.org/abs/2108.06098 lots of p[arams
  - lokr - Low-Rank Kronecker Product  - 
  - lora
  - mixes
  - ~~multi_task_prompt_tuning~~
  - oft - Orthogonal Finetuning mode
  - ~~p_tuning~~
  - ~~prefix_tuning~~
  - ~~prompt_tuning~~



ia3, which modules

k v not o or q
one of the mlp. in this case it was an intermediateo ne? wi_1 the middle one, not the out one

# 2023-12-29 07:28:30

IA3 seems better but it's not learning, more epochs seem to help? more params too
also I need to fix intervention code

plus I would like to use IA3 to get an lie detection importance matrix. Then I can use that with a VAE, and see if it's helpfull to a probe! I would need to configure it to work on the residual only?

out_proj + fc2 combined?
- run this, why not?
- also seems

# 2023-12-29 12:00:11 VAE + Importance

Step 1: I use IA3 configured on just the the parts the directly effect the residual 

```py
peft_config = IA3Config(
    task_type=TaskType.SEQ_CLS, target_modules=[  "fc2", "out_proj"], 
        # feedforward_modules=["fc2","out_proj", ]
)
```

Note that this is not that flexible. It can only learn to lie on 40% (up from 26) on known questions. But iit's good for an importance matrix.
Wait that was with the start of the MLP, with the end it learns nothing!
So it modifying the residual stream doesn't change it's behaviour, I can infer that the residual stream is not important for lie detection. This kind of fits with the poor results people have been getting.
Logically what's important are the activations on the default params! So I should try using them!! That's the output of Wqkv and and output of fc1 (input of fc2). these are the ia3 params

# for normal ia3 seetings see https://github.com/huggingface/peft/blob/cf04d0353f0343cbf66627228c4495f51669af34/src/peft/utils/constants.py#L81
# and https://github.com/huggingface/peft/blob/cf04d0353f0343cbf66627228c4495f51669af34/src/peft/utils/constants.py#L102


            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you shoud specify the target modules manually."
            ),


TODO:
- [ ] read anthropic [paper](https://transformer-circuits.pub/2022/toy_model/index.html) on importance matrix, 
  - [x] [maybe reply to colin](https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab)

> Consider a toy model where we train an embedding of five features of varying importanceWhere “importance” is a scalar multiplier on mean squared error loss. in two dimensions, add a ReLU afterwards for filtering, and vary the sparsity of the features. With dense features, the model learns to represent an orthogonal basis of the most important two features (similar to what Principal Component Analysis might give us), and the other three features are not represented. But if we make the features sparse, this changes:
> Where “importance” is a scalar multiplier on mean squared error loss.
>
> Features Vary in Importance: Not all features are equally useful to a given task. Some can reduce the loss more than others. For an ImageNet model, where classifying different species of dogs is a central task, a floppy ear detector might be one of the most important features it can have. In contrast, another feature might only very slightly improve performance
	> r computational reasons, we won't focus on it in this article, but we often imagine an infinite number of features with importance asymptotically approaching zero.


# data sets I would like custom prompts for

math
ScienceQA
TruthfullQA

# 2023-12-30 07:15:34

So I'm trying to do that experiment where I take the residual at the same point that IA3 intervenes. But suddenly it's not learning. Why? :bug:

- x disable bitsandbytes bnb, nope
- x needs out_project, nope still not learning
- x make some feedforward... yes. WTF it seems like in the peft implementation of ia3. only the feedforward ones are learning. Or is it due to tracedict? no that's only on during collect
- x it's not inference_mode is it, what does that do
- x is it grad accum? no
- x make sure I set taks to causal no
- x grad clip no
- turn of 16b train? 
- avoid lightning?
- a big in the peft code... not that I can see or inspect during debugging
  - it is active
  - it has grad
  - it's stil l1 after training :(
- **ranger21? YES!!** WTH

success criteria
- if train loss should go from 3-5 to 1-3 in the first epoch at least for nll


what is T-frw from the ia3 paper?

n summary, the T-Few recipe is defined as follows: We use the T0 model as a backbone. We add
(IA)3 for downstream task adaptation and use parameters initialized from pre-training (IA)3 on the
same multitask mixture for T0. As an objective, we use
- the sum of a standard language modeling loss LLM, 
- **an unlikelihood loss LUL for incorrect choices** see 3.2 https://arxiv.org/pdf/2205.05638.pdf
- and a length-normalized loss LLN. 
- We train for 1,000 steps with a batch size of 8 sequences using the Adafactor optimizer [ 49] with a learning rate of 3e−3 and a linear decay schedule with a 60-step warmup. 

# unlikelihood loss LUL

```py
# https://github.com/r-three/t-few/blob/114deced63ae722a368e06cb08ea956b20c393a6/src/models/EncoderDecoder.py#L94
lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.tokenizer.pad_token_id).long()


cand_loglikely = -F.cross_entropy(
    model_output.logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none"
).view(bs, num_choices, -1)
cand_loglikely += (lm_target < 0).view(bs, num_choices, -1) * -100
cand_loglikely[range(bs), labels] = -100
unlikely_loss = -torch.log(1 - torch.exp(cand_loglikely) + 1e-2).sum() / (cand_loglikely != -100).sum()
```





# why does ia3 only work for feedforward_modules/

trace it in peft...

is_feedforward




bitsandbytes, I have "0.41.3.post2", latest is 0.41.3.post1 ?
peft, I have 0.7.1, latest 0.7.1
transformers "4.34.0", latest Patch release: v4.36.


# try precision once again

If I want true bnb, I load and follow this https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html#quantization-via-bitsandbytes

but phi had annoying programming. It keep resetting to 32 bit. But if I used 16-mixed at least the ensures the right type.

Note without bnb I run out of mem with 1 batch

with hard 16 I get inf, so I proboly need the lroa parts to be 32.


TODO
- [x] fix IA3
- [ ] unlikelihood loss LUL
- [x] AEC
  - [x] extract importance matrix from adapter
  - [x] handle new formats


# 2023-12-31 08:47:05

Hmm I'm using a adapter that doesn't do much. I may need to use out_proj as well or instead. 
- [ ] In fact I could try a logistic regression on all activations and see which are the most usefull!!! but none are working. hmm
- [ ] LUL loss
- [ ] I could try a full bnb notebook, that uses manual training loop. See if I can get batch 2
- [x] by VAE dl is very slow, hmm. oh it's just the dataset, solved with workers>0
- [ ] IMPORTANCE MATRIX!!


without with_torch is it 3x faster


open questions:
- whats the best representation? I'm thinking no one knows, and it's deeply non linear
- 


experiment, collect EVERYTHING. Save to disc regularly. Try it ALL for interpret.
- also collect features? Like diff. Mult.


idea: importancem matrix and logistic regression? E.g. top k?

Why does my importance matrixn ot match?? in probes nb? But it seems fine in nb 11? Prob bc in 11 we decimate which makes them seem the same

importance_matrix.shape
torch.Size([16, 17920])
torch.Size([1336, 16, 17918, 2])

So it looks like we diffed the neurons instead as the layers are the same...
yup because I said n, not dim, and it did the last dim.


# 2023-12-31 14:45:10 

UPTO
- I'm trying the importance matrix experiment. Still need to work out the best way to do an importance matrix? Take the top std? Or top and bottom?
- and the VAE, but that can wait
- I just fixed a bug where I was diffing by neurons (nonsense) so I need to recollect hidden states
- I still am confused why IA3 hardly train it to lie, so I an now trying the opposite KLDiv loss. And I might try ULU loss
- 


Hypothesis: an adapters residuals and importance matrix will help interpret hidden states. Experiment: LR on residual * importance matrix

# 2024-01-01 10:00:49


UPTO
- [x] I just fixed a bug where I was diffing by neurons (nonsense) so I need to recollect hidden states
- [ ] I'm trying the importance matrix experiment. Still need to work out the best way to do an importance matrix? Take the top std? Or top and bottom?
- [ ] and the VAE, but that can wait
- [ ] I still am confused why IA3 hardly train it to lie, so I an now trying the opposite KLDiv loss. And I might try ULU loss.
- [ ] :bug: my known lie doesn't make sense anymore? Well truth is when examples of truth are given (and maybe it's asked to tell the truth?) I would need to analyze this seperatly, but getting say N tokens. And trying differen't combinatiuons of system prompt, and n-shot examples
- [ ] :bug: fix the intervention thing in the model training nb
- [ ] in PEFT/LORA they use out_project  and gc2 as well! "phi": ["Wqkv", "out_proj", "fc1", "fc2"], maybe try those to


use the importance matrix in the VAE! But first make sure the feature are usefull

# 2024-01-03 08:16:19

Cleaning probe code. Need to rerun 06b. And try diff importance matrix with LR


With no importance matrix: 17 mins, and 0.64 auroc
w matrix 10 min 0.6731
top 1%, 10 sec and  0.64
abs top 1% 11s 0.66
top 1% abs, just Wqkv 0.646
top 1% abs, just fc1 0.66

Findings 
**So fc1 better**
**importance matrix helps**
**abs** is better

top abs 20%. runtime=1.5min, auc= 0.62
top 3%, 0.64

TODO try pl bolt on lr
try my own pl methods...

# 2024-01-04 15:32:50

I need to flip during training. Easiest to just flip the last dim of y, and y=1-y


- [x] run 06b
- [x] debug 07 compare probes
  - [x] importance matrix... it helps but not much
  - [ ] experiment with diff layers?
- [ ] VAE with importance matrix?
  - [ ] how to make tractable?
- [ ] codebook
- [ ] grant


I can just repurpose other codebook code? https://lilianweng.github.io/posts/2018-08-12-vae/#vq-vae-and-vq-vae-2 prob need 1d tho
- openAI Jukebox uses them for audio  https://old.reddit.com/r/MachineLearning/comments/109yuvi/d_has_any_work_been_done_on_vqvae_language_models/
- Sounds like HuBERT and other MLMs used for ASR pretraining. Look for seq2seq work in the world of TTS and ASR.
- DALL-E additionally uses VQ-VAE[4] to operate this model, which was used only for texts, and for images as well.
- https://github.com/ML4ITS/TimeVQVAE
