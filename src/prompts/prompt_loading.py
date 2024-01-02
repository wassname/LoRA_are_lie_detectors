"""
Modified from https://github.com/EleutherAI/elk/blob/3bbe26c3858aac1b03e6f80628a5056fae44db9c/elk/extraction/prompt_loading.py#L129

Changed to record choices
"""
from collections import Counter
from random import Random
from typing import Any, Iterator, Literal, List, Dict
from pathlib import Path
# import datasets
import json
from jinja2 import TemplateError
from datasets import ClassLabel, Dataset, Value, load_dataset
import yaml
import numpy as np
from elk.promptsource.templates import env
from elk.promptsource import DatasetTemplates
from elk.utils import (
    assert_type,
    infer_label_column,
    select_split,
)
import datasets
import functools
from elk.extraction.balanced_sampler import BalancedSampler, FewShotSampler
import pandas as pd
from loguru import logger
from src.helpers.ds import shuffle_dataset_by
from src.helpers.scores import row_choice_ids
from src.config import ExtractConfig
from src.models.load import verbose_change_param, AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase


# Local path to the folder containing the templates
TEMPLATES_FOLDER_PATH = Path(__file__).parent / "templates"

def load_prompt_structure(prompt_format='llama2'):
    # for use with https://huggingface.co/docs/transformers/main/chat_templating is the tokenizer doesn't include it
    f = TEMPLATES_FOLDER_PATH  / "prompt_formats" / f"{prompt_format}.jinja2"
    if not f.exists():
        raise FileNotFoundError(f"Could not find prompt format {prompt_format} at {f}")
    return f.open().read()


def load_default_sys_instructions(path='system.yaml'):
    f = TEMPLATES_FOLDER_PATH / path
    yaml_dict = yaml.load(f.open('r'), Loader=yaml.FullLoader)
    templates = yaml_dict["templates"]["falsity"]
    return templates

default_sys_instructions = load_default_sys_instructions()


def sample_n_true_y_false_prompts(prompts, num_truth=1, num_lie=1, seed=42):
    """sample some truth and some false"""
    df = pd.DataFrame(prompts)
    
    # restrict to template where the choices are a single token
    # m = df.answer_choices.map(answer_len)<=2
    # df = df[m]
    df = pd.concat([
        df.query("instructed_to_lie==True").sample(num_truth, random_state=seed),
        df.query("instructed_to_lie==False").sample(num_lie, random_state=seed)])
    return df.to_dict(orient="records")

def load_prompts(
    ds_string: str,
    *,
    sys_instructions: Dict[bool, Dict[str, str]]= default_sys_instructions,
    binarize: bool = True,
    num_shots: int = 0,
    seed: int = 42,
    split_type: Literal["train", "val"] = "train",
    template_path: str | None = None,
    rank: int = 0,
    world_size: int = 1,
    prompt_sampler = sample_n_true_y_false_prompts,
    N=np.inf,
) -> Iterator[dict]:
    """Load a dataset full of prompts generated from the specified dataset.

    Args:
        ds_string: Name of HF dataset to use, e.g. `"super_glue:boolq"` or `"imdb"`.
        binarize: Whether to binarize the dataset labels for multi-class datasets.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot.
        seed: The seed to use for prompt randomization.
        split_type: Whether to use the train or val split of the dataset.
        template_path: Path to feed into `DatasetTemplates` for loading templates.
        rank: The rank of the current process. Defaults to 0.
        world_size: The number of processes. Defaults to 1.
        prompt_sampler: when given an unbalanced set of true and false prompts this might take one of each randomly

    Returns:
        An iterable of prompt dictionaries.
    """
    ds_name, _, config_name = ds_string.partition(":")

    ds_dict = assert_type(dict, load_dataset(ds_name, config_name or None))
    split_name = select_split(ds_dict, split_type)

    # TODO:, can I make sure it's the same shuffle regardless of length?
    ds = assert_type(Dataset, ds_dict[split_name].shuffle(seed=seed))
    if world_size > 1:
        ds = ds.shard(world_size, rank)

    if template_path is None:
        prompter = DatasetTemplates(ds_name, config_name)
    else:
        prompter = DatasetTemplates(template_path)

    # If the prompt template says to binarize, we should
    binarize = binarize or prompter.binarize
    prompter.drop_non_mc_templates()

    num_templates = len(prompter.templates)
    assert num_templates > 0
    if rank == 0:
        logger.info(f"Extracting {num_templates} variants of each prompt")

    label_column = prompter.label_column or infer_label_column(ds.features)

    label_feature = ds.features[label_column]
    if isinstance(label_feature, ClassLabel):
        label_choices = [label_feature.str2int(label) for label in label_feature.names]
    elif isinstance(label_feature, Value) and label_feature.dtype == "bool":
        label_choices = [False, True]
    else:
        # Which classes are actually present in this split of the dataset?
        # This is shockingly fast since it uses an optimized Apache Arrow primitive.
        label_choices = sorted(ds.unique(label_column))
        if rank == 0:
            logger.info(f"Using the following pseudo-labels: {label_choices}")

    rng = Random(seed)
    if num_shots > 0:
        train_name = select_split(ds_dict, "train")
        
        # TODO don't we need to binarize this?
        # FIXME: this doesn't binarize
        fewshot = FewShotSampler(
            ds_dict[train_name].shuffle(seed=seed),  # TODO: not iterator
            num_shots=num_shots,
            rng=rng,
            label_col=label_column,
        )
        fewshot_iter = iter(fewshot)
    else:
        fewshot_iter = None

    if label_column in ds.features:
        ds = BalancedSampler(
            ds.to_iterable_dataset(),
            set(label_choices),
            label_col=label_column,
        )
    else:
        if rank == 0:
            logger.info("No label column found, not balancing")
        ds = ds.to_iterable_dataset()

    j = 0
    for i, example in enumerate(ds):
        if j>N:
            break
        prompts = _convert_to_prompts(
            example,
            binarize=binarize,
            label_column=label_column,
            label_choices=label_choices,  # type: ignore[arg-type]
            prompter=prompter,
            rng=rng,
            sys_instructions=sys_instructions,
            fewshot_iter=fewshot_iter,
        )
        prompts = [{'ds_string': ds_string, 'example_i':i, **p} for p in prompts]
        
        def prompt_ok(prompt):
            """ we want answers where we can distinguish them from the first token
            we don't have access to the tokenizer here, so we just make sure the first 3 letters are differen't and there are not spaces
            """
            answer_choices = prompt['answer_choices']
            a = answer_choices[0][:3]
            b = answer_choices[1][:3]
            keep = (a != b) and ' ' not in a
            if not keep:
                logger.debug(f"removing prompt because it's answers are not unique: {prompt['ds_string']} {prompt['template_name']} {prompt['answer_choices']}")
            return keep

        prompts = list(filter(prompt_ok, prompts))
        prompts = prompt_sampler(prompts, seed=42+j)
        # TODO: make sure they are single token answers (or at least the first token is unique)
        for p in prompts:
            j += 1
            yield p


def cast_example(e):
    assert e['label']>=0
    assert e['label']<=1
    e['label']=bool(e['label'])
    return e


def _convert_to_prompts(
    example: dict[str, Any],
    prompter: DatasetTemplates,
    binarize: bool,
    label_column: str,
    label_choices: list[bool | int | str],
    rng: Random,
    sys_instructions: Dict[bool, Dict[str, str]] = default_sys_instructions,
    fewshot_iter: Iterator[list[dict]] | None = None,
) -> list:
    """Prompt-generating function to pass to `IterableDataset.map`."""
    example = cast_example(example)
    prompts = []
    templates = list(prompter.templates.values())

    # For sanity checking that prompts are unique
    prompt_counter = Counter()
    label = example[label_column]

    if binarize:
        # Replace the full list of possibilities with a randomly sampled false label
        # and the correct label, as done in the DLK paper. Note that this does add some
        # "supervision" by stacking the deck in favor of the correct answer.
        label_choices = [
            rng.choice([c for c in label_choices if c != label]),
            label,
        ]
        rng.shuffle(label_choices)

    for template in templates:
        answer_choices=template.get_fixed_answer_choices_list()
        
        # skip prompts where the responses are similar in the first token
        if answer_choices[0][:3]==answer_choices[1][:3]:
            logger.trace(f"skipping prompt because it's answers are not unique (for the first token): {template.name} {answer_choices}")
            continue
        answer_choices = [[c] for c in answer_choices]
        for instructed_to_lie in [False, True]:
            for sys_instr_name, sys_instr in sys_instructions[instructed_to_lie].items():
                instructed_example = example.copy()
                if instructed_to_lie: instructed_example['label'] = not bool(instructed_example['label'])

                q, a = template.apply(instructed_example)
                messages = [
                    
                    dict(role='user', content=q)
                ]
                prompt_counter[(sys_instr + q, a)] += 1

                if fewshot_iter is not None:
                    # Infinite iterator so we don't need to worry about StopIteration
                    fewshot_examples = next(fewshot_iter)
                    fewshot_examples = [cast_example(e).copy() for e in fewshot_examples]
                    
                    if instructed_to_lie: 
                        fewshot_examples = [{**e, 'label': not bool(e['label'])} for e in fewshot_examples]
                        for e in fewshot_examples:
                            # arg, check negation worked
                            assert e['label']>=0
                            assert e['label']<2
                            assert isinstance(e['label'], bool), 'labels should be bool'
                        
                    fewshot_texts = []
                    for q, a in map(template.apply, fewshot_examples):
                        fewshot_texts.append(dict(role='user', content=q))
                        fewshot_texts.append(dict(role='assistant', content=a.strip()))
                        # some of the answers have extra trailing text, that's OK. But extra preceeding text is not, let's check for that
                        aa = a.strip()
                        assert any([any([aa.startswith(a) for a in ac]) for ac in answer_choices]), f"fewshot response `{aa}` has extra preceeding text compared to allowed choices: {answer_choices}. template is: {template.name}"
                    messages = [dict(role='system', content=sys_instr)] + fewshot_texts + messages

                prompts.append(dict(
                    # Strip whitespace from the answer to make it easier to
                    # compare with the model's output
                    answer=a.strip(),
                    messages=messages,
                    
                    answer_choices=answer_choices,
                    template_name=template.name,
                    label_true=example['label'],
                    label_instructed=instructed_example['label'],
                    instructed_to_lie=instructed_to_lie,
                    sys_instr_name=sys_instr_name,
                ))

    # Sanity check: variants should be unique
    ((maybe_dup, dup_count),) = prompt_counter.most_common(1)
    if dup_count > 1:
        raise ValueError(f'Prompt duplicated {dup_count} times! "{maybe_dup}"')

    return prompts

def format_prompt(tokenizer, messages):
    # if not chat template is present, load it from structure.yaml onto tokenizer
    # https://huggingface.co/docs/transformers/main/chat_templating
    try:
        q = tokenizer.apply_chat_template(messages, tokenize=False)
    except TemplateError as e:
        if 'Conversation roles must alternate user/assistant/user/assistant/...' in str(e):
            system = messages[0]['content']
            q = tokenizer.apply_chat_template(messages[1:], tokenize=False)
            q = system + q
        else:
            raise e
    return q


def load_preproc_datasets(dataset_names: List[str], tokenizer: PreTrainedTokenizerBase, N:int, prompt_format:str = None, split_type:str="train", seed=42, num_shots=1, max_length=999):
    datasets2 = []
    n = N//len(dataset_names)+1
    for ds_name in dataset_names:
        ds_tokens1 = load_preproc_dataset(
            ds_name,
            tokenizer,
            N=n,
            seed=seed,
            num_shots=num_shots,
            max_length=max_length,
            prompt_format=prompt_format,
        ).with_format("torch")
        datasets2.append(ds_tokens1)
    ds_tokens = datasets.interleave_datasets(datasets2, seed=seed)

    # inds = list(range(ds_tokens))
    # Random(seed).shuffle(inds)
    # ds_tokens = ds_tokens.select(inds)

    return ds_tokens


def load_preproc_dataset(ds_name: str, tokenizer: PreTrainedTokenizerBase, N:int, prompt_format:str = None, split_type:str="train", seed=42, num_shots=1, max_length=999, sys_instructions=default_sys_instructions) -> Dataset:
    # rng = Random(seed)
    ds_prompts = Dataset.from_generator(
        load_prompts,
        gen_kwargs=dict(
            ds_string=ds_name,
            num_shots=num_shots,
            split_type=split_type,
            sys_instructions=sys_instructions,
            # template_path=template_path,
            seed=seed,
            N=N*3,
        ),
        # prefetch=False,
        # num_proc=0,
        keep_in_memory=False,
    )
    
    
    # ## Format prompts
    # The prompt is the thing we most often have to change and debug. So we do it explicitly here.
    # We do it as transforms on a huggingface dataset.
    # In this case we use multishot examples from train, and use the test set to generated the hidden states dataset. We will test generalisation on a whole new dataset.
    if tokenizer.chat_template is None:
        if prompt_format is not None:
            logger.info(f"setting tokenizer chat template to {prompt_format}")
            tokenizer.chat_template = load_prompt_structure(prompt_format=prompt_format)
        else:
            logger.error(f"tokenizer has no chat template, and you have not set one. Using default one which might lead to poor performance. Set it in the config")
            logger.debug(f"tokenizer already has a chat template: \n\n{tokenizer.chat_template}\n\n. overriding with {prompt_format}")

    ds_tokens = (
        ds_prompts
        .map(lambda ex: {'question': format_prompt(tokenizer, ex['messages'])}, desc="format_prompt")
        .map(
            lambda ex: tokenizer(
                ex["question"], padding="max_length", max_length=max_length, truncation=True, add_special_tokens=True,
                return_tensors="pt",
                return_attention_mask=True,
                # return_overflowing_tokens=True,
            ),
            batched=True,
            desc='tokenize',
        )
        .map(lambda r: {"truncated": np.sum(r["attention_mask"], 0)==max_length}, desc='truncated')
        .map(lambda r: {"length": np.sum(r["attention_mask"], 0)}, desc='truncated')
        .map(
            lambda r: {"prompt_truncated": tokenizer.batch_decode(r["input_ids"])},
            batched=True,
            desc='prompt_truncated',
        )
        .map(lambda r: {'choice_ids': row_choice_ids(r, tokenizer)}, desc='choice_ids')
    )
    
    b4 = ds_tokens.num_rows
    # print('num_rows', ds_tokens.num_rows)
    logger.info(f"median token length: {np.median(ds_tokens['length'])} for {ds_name}. max_length={max_length}")
    # print(np.histogram(ds_tokens['length']))
    truncation_rate = np.mean(ds_tokens['truncated'])
    assert truncation_rate<0.75, f"truncation rate is too high {truncation_rate}. Try a longer max_length than {max_length}"
    logger.info(f"truncation rate: {truncation_rate:2.2%} on {ds_name}")
    ds_tokens = ds_tokens.filter(lambda r: r["truncated"] == False)

    # print('num_rows', ds_tokens.num_rows)
    ds_tokens = shuffle_dataset_by(ds_tokens, target='label_true', random_state=seed, stratify_columns=[])
    # print('num_rows', ds_tokens.num_rows)
    
    # ## Filter out truncated examples
    ds_tokens = ds_tokens.filter(lambda r: not r['truncated'])
    logger.info(f'num_rows (after filtering out truncated rows) {b4}=>{ds_tokens.num_rows}')
    assert len(ds_tokens), f'No examples left after filtering out truncated rows, try a longer max_length than {max_length}'
    assert len(ds_tokens)>=N, f'Few {len(ds_tokens)}<{N} examples left after filtering out truncated rows, try a longer max_length than {max_length}'
    if len(ds_tokens)>N:
        ds_tokens = ds_tokens.select(range(N))
    return ds_tokens
