# type: ignore

import modal
import os

app = modal.App("convert_gguf_2")

unsloth_image = modal.Image.from_dockerfile("src/conversion/modal/Dockerfile")


@app.function(
    gpu="T4",
    image=unsloth_image,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("hugginface-secret")],
)
def convert_base_model_gguf(hf_model_repo: str, save_model_repo: str) -> str:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_model_repo,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    model.push_to_hub_gguf(
        save_model_repo,
        tokenizer,
        quantization_method="f16",
        token=os.environ["HF_ACCESS_TOKEN"],
    )
    return ""


@app.local_entrypoint()
def main():
    hf_model_repo = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    save_model_repo = "pookie3000/Meta-Llama-3.1-8B-bnb-4bit-f16-gguf"
    convert_base_model_gguf.remote(
        hf_model_repo=hf_model_repo,
        save_model_repo=save_model_repo,
    )


# run using
# modal run src/conversion/modal/convert_gguf.py
