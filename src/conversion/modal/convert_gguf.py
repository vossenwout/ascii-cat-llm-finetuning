# type: ignore

import modal
import os

app = modal.App("convert_gguf")

unsloth_image = modal.Image.from_dockerfile("src/conversion/modal/Dockerfile")


@app.function(
    gpu="T4",
    image=unsloth_image,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("hugginface-secret")],
)
def convert_base_model_gguf(
    hf_model_repo: str,
    save_model_repo: str,
    quantization_method: str,
    load_in_4bit: bool,
) -> str:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_model_repo,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=load_in_4bit,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    # else ascii generation will get messed up
    tokenizer.clean_up_tokenization_spaces = False

    model.push_to_hub_gguf(
        save_model_repo,
        tokenizer,
        quantization_method=quantization_method,
        token=os.environ["HF_ACCESS_TOKEN"],
    )
    return ""


@app.local_entrypoint()
def main():
    hf_model_repo = "meta-llama/Llama-3.2-3B"
    save_model_repo = "pookie3000/Llama-3.2-3B-GGUF"
    quantization_method = "not_quantized"
    load_in_4bit = False
    convert_base_model_gguf.remote(
        hf_model_repo=hf_model_repo,
        save_model_repo=save_model_repo,
        quantization_method=quantization_method,
        load_in_4bit=load_in_4bit,
    )


# run using
# modal run src/conversion/modal/convert_gguf.py


# !!! DOE GEWOON https://huggingface.co/spaces/ggml-org/gguf-my-repo
