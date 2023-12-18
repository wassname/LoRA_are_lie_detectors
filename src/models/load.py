"""
This file load various open source models

When editing or updating this file check out these resources:
- [LLM-As-Chatbot](https://github.com/deep-diver/LLM-As-Chatbot/blob/main/models/falcon.py)
- [oobabooga](https://github.com/oobabooga/text-generation-webui/blob/main/modules/models.py#L134)
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerBase, PreTrainedTokenizer, GPTQConfig, BitsAndBytesConfig
import torch
from loguru import logger
from typing import Tuple

def verbose_change_param(tokenizer, path, after):
    
    if not hasattr(tokenizer, path):
        logger.info(f"tokenizer does not have {path}")
        return tokenizer
    before = getattr(tokenizer, path)
    if before!=after:
        setattr(tokenizer, path, after)
        logger.info(f"changing {path} from {before} to {after}")
    return tokenizer


def load_model(model_repo =  "microsoft/phi-2", pad_token_id=0, disable_exllama=True, device="auto",) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """
    A uncensored and large coding ones might be best for lying.
    
    """
    # see https://github.com/deep-diver/LLM-As-Chatbot/blob/main/models/starchat.py
    # gptq_config = GPTQConfig(bits=4, dataset="c4", disable_exllama=False)
    model_options = dict(
        device_map=device,
        torch_dtype=torch.float16, 
        # load_in_8bit=True, # difficult
        load_in_4bit=True,
        trust_remote_code=True,
        # disable_exllama=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4',
        ),

        ## in the azure phi-repo they use these but you need to install flashattn
        # flash_attn=True, 
        # flash_rotary=True, 
        # fused_dense=True
    )

    config = AutoConfig.from_pretrained(model_repo, trust_remote_code=True,)
    # config.quantization_config['use_exllama'] = False    

    # disable
    # quantization_config=GPTQConfig(**dict(**config.quantization_config, disable_exllama=False))
    # config.quantization_config = quantization_config

    verbose_change_param(config, 'use_cache', False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True, legacy=False)
    verbose_change_param(tokenizer, 'pad_token_id', pad_token_id)
    verbose_change_param(tokenizer, 'padding_side', 'left')
    verbose_change_param(tokenizer, 'truncation_side', 'left')
    
    model = AutoModelForCausalLM.from_pretrained(model_repo, config=config, 
                                                 **model_options)
    
    # if disable_exllama:
    #     from auto_gptq import exllama_set_max_input_length
    #     model = exllama_set_max_input_length(model, max_input_length=5000)

    return model, tokenizer

