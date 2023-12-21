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
import peft

def verbose_change_param(tokenizer, path, after):
    
    if not hasattr(tokenizer, path):
        logger.info(f"tokenizer does not have {path}")
        return tokenizer
    before = getattr(tokenizer, path)
    if before!=after:
        setattr(tokenizer, path, after)
        logger.info(f"changing {path} from {before} to {after}")
    return tokenizer


def load_model(model_repo =  "microsoft/phi-2", adaptor_path=None, device="auto", bnb=True) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """
    A uncensored and large coding ones might be best for lying.
    
    """
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # bfloat16 is recommended
        # bnb_4bit_use_double_quant=False,
        # bnb_4bit_quant_type='nf4',
    )
    if not bnb:
        quantization_config = None
    model_options = dict(
        device_map=device,
        torch_dtype=torch.float16, 
        trust_remote_code=True,
        quantization_config=quantization_config,

        ## in the azure phi-repo they use these but you need to install flash-attn
        # flash_attn=True, 
        # flash_rotary=True, 
        # fused_dense=True
    )

    config = AutoConfig.from_pretrained(model_repo, trust_remote_code=True,)
    # config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True, legacy=False)
    tokenizer.pad_token_id = 0 # tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(model_repo, config=config, 
                                                    **model_options)
    
    if adaptor_path is not None:
        model = peft.PeftModel.from_pretrained(model, adaptor_path)
    return model, tokenizer

