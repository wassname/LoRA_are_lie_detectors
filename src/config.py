from simple_parsing import Serializable, field
from dataclasses import InitVar, dataclass, replace
from typing import Literal
from pathlib import Path

root_folder = Path(__file__).parent.parent.absolute()
TEMPLATE_PATH = root_folder / "src/prompts/templates/"


@dataclass
class ExtractConfig(Serializable):
    """Config for extracting hidden states from a language model."""

    datasets: tuple[str, ...] = ("amazon_polarity",  "super_glue:boolq", "glue:qnli", "imdb")
    """Names of HF datasets to use, e.g. `"super_glue:boolq"` or `"imdb"` `"glue:qnli"""
    
    model: str = "wassname/phi-2-GPTQ_w_hidden_states"

    batch_size: int = 5


    prompt_format: str | None = 'phi'
    """if the tokenizer does not have a chat template you can set a custom one. see src/prompts/templates/prompt_formats/readme.md."""
    

    num_shots: int = 2
    """Number of examples for few-shot prompts. If zero, prompts are zero-shot."""

    max_length: int | None = 1000
    """Maximum length of the input sequence passed to the tokenize encoder function"""
    
    intervention_fit_examples: int = 200
    """how many example to use for intervention calibration"""
