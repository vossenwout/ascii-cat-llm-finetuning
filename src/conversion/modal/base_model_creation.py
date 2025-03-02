# type: ignore

import modal
import os

app = modal.App("convert_gguf")

unsloth_image = modal.Image.from_dockerfile("src/conversion/modal/Dockerfile")


def add_ascii_tokens_to_model(model, tokenizer):
    ascii_tokens = {"additional_special_tokens": ["<ascii>", "</ascii>"]}
    tokenizer.add_special_tokens(ascii_tokens)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


@app.function(
    gpu="L4",
    image=unsloth_image,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("hugginface-secret")],
)
def add_ascii_to_tokenizer(hf_model_repo: str, save_model_repo: str) -> str:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_model_repo,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    model, tokenizer = add_ascii_tokens_to_model(model, tokenizer)

    model.push_to_hub(
        save_model_repo,
        token=os.environ["HF_ACCESS_TOKEN"],
    )
    tokenizer.push_to_hub(
        save_model_repo,
        token=os.environ["HF_ACCESS_TOKEN"],
    )
    return ""


# Creates a model with ascii tokens added
@app.local_entrypoint()
def main():
    hf_model_repo = "unsloth/Meta-Llama-3.1-8B"
    save_model_repo = "pookie3000/Meta-Llama-3.1-8B-ascii-tokenizer"
    add_ascii_to_tokenizer.remote(
        hf_model_repo=hf_model_repo,
        save_model_repo=save_model_repo,
    )


# run using
# modal run src/conversion/modal/base_model_creation.py
