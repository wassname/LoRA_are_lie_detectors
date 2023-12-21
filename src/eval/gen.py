from IPython.display import display, HTML
import torch
# generate
# https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/text_generation#transformers.GenerationConfig


@torch.no_grad()
def gen(model, inputs, tokenizer):
    s = model.generate(
        input_ids=inputs["input_ids"][None, :].to(model.device),
        attention_mask=inputs["attention_mask"][None, :].to(model.device),
        use_cache=False,
        max_new_tokens=100,
        min_new_tokens=100,
        do_sample=False,
        early_stopping=False,
    )
    input_l = inputs["input_ids"].shape[0]
    old = tokenizer.decode(
        s[0, :input_l], clean_up_tokenization_spaces=False, skip_special_tokens=False
    )
    new = tokenizer.decode(
        s[0, input_l:], clean_up_tokenization_spaces=False, skip_special_tokens=False
    )
    display(HTML(f"<pre>{old}</pre><b><pre>{new}</pre></b>"))
