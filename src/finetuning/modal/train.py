# type: ignore

import modal
from typing import Any
from datasets import Dataset
import os

app = modal.App("ascii_finetuning")

unsloth_image = modal.Image.from_dockerfile("src/finetuning/modal/Dockerfile")

OUTPUT_ASCII_PROMPT = """
Generate ascii art that matches the following description.

### description:
{description}

### ascii visualization:
<ascii>
{ascii_art}
</ascii>
"""


def create_model(
    base_model: str, max_seq_length: int, train_embeddings: bool
) -> tuple[Any, Any]:

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    common_peft_params = {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_rslora": False,
        "loftq_config": None,
    }

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    if train_embeddings:
        target_modules.extend(["embed_tokens", "lm_head"])

    model = FastLanguageModel.get_peft_model(
        model, target_modules=target_modules, **common_peft_params
    )

    return model, tokenizer


def prepare_dataset(hf_dataset_name: str, tokenizer: Any) -> Dataset:
    from datasets import load_dataset

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        ascii_art_samples = examples["ascii"]
        training_prompts = []
        for ascii_art in ascii_art_samples:
            training_prompt = (
                OUTPUT_ASCII_PROMPT.format(description="cat", ascii_art=ascii_art)
                + EOS_TOKEN
            )
            training_prompts.append(training_prompt)
        return {
            "text": training_prompts,
        }

    dataset = load_dataset(hf_dataset_name, split="train")
    return dataset.map(formatting_prompts_func, batched=True)


@app.function(
    gpu="T4",
    image=unsloth_image,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("hugginface-secret")],
)
def train(
    base_model: str,
    hf_dataset_name: str,
    hf_save_lora_name: str,
    epochs: int,
    train_embeddings: bool,
    max_seq_length: int,
) -> str:
    from trl import SFTTrainer, SFTConfig
    import torch

    # Create model and tokenizer
    model, tokenizer = create_model(
        base_model=base_model,
        max_seq_length=max_seq_length,
        train_embeddings=train_embeddings,
    )

    # Prepare dataset
    dataset = prepare_dataset(hf_dataset_name=hf_dataset_name, tokenizer=tokenizer)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            num_train_epochs=epochs,
            warmup_steps=5,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            dataset_text_field="text",
        ),
    )

    # Train and save
    trainer_stats = trainer.train()
    model.push_to_hub(hf_save_lora_name, token=os.environ["HF_ACCESS_TOKEN"])
    tokenizer.push_to_hub(hf_save_lora_name, token=os.environ["HF_ACCESS_TOKEN"])

    return ""


@app.local_entrypoint()
def main():
    # PARAMS
    base_model = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    hf_dataset_name = "pookie3000/ascii-art-animals"
    hf_save_lora_name = "pookie3000/ascii-art-cats-lora-v1"
    epochs = 3
    train_embeddings = False
    max_seq_length = 2048

    # TRAINING
    train.remote(
        base_model=base_model,
        hf_dataset_name=hf_dataset_name,
        hf_save_lora_name=hf_save_lora_name,
        epochs=epochs,
        train_embeddings=train_embeddings,
        max_seq_length=max_seq_length,
    )


# run using
# modal run src/finetuning/modal/train.py
