from simple_parsing import Serializable, field
from dataclasses import InitVar, dataclass, replace
from typing import Literal
from pathlib import Path

root_folder = Path(__file__).parent.parent.absolute()
TEMPLATE_PATH = root_folder / "src/prompts/templates/"


@dataclass
class ExtractConfig(Serializable):
    """Config for extracting hidden states from a language model."""

    datasets: tuple[str, ...] = ("amazon_polarity",  "super_glue:boolq", "imdb" )
    """Names of HF datasets to use, e.g. `"super_glue:boolq"` or `"imdb"` `"glue:qnli"""

    datasets_oos: tuple[str, ...] = ("glue:qnli", )
    
    # model: str = "wassname/phi-2-w_hidden_states"
    # model: str = "/media/wassname/SGIronWolf/projects5/elk/sgd_probes_are_lie_detectors/phi-1_5"
    model: str = "wassname/phi-1_5-w_hidden_states"

    batch_size: int = 2

    prompt_format: str | None = 'phi'
    """if the tokenizer does not have a chat template you can set a custom one. see src/prompts/templates/prompt_formats/readme.md."""
    
    num_shots: int = 1
    """Number of examples for few-shot prompts. If zero, prompts are zero-shot."""

    max_length: int | None = 777
    """Maximum length of the input sequence passed to the tokenize encoder function"""
    
    # intervention_fit_examples: int = 200
    # """how many example to use for intervention calibration"""

    max_examples: tuple[int, int] = (1000, 200)
    """Maximum number of examples to use from each split of the dataset."""

    seed: int = 42
    """Random seed."""
