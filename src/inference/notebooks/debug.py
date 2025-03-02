from llama_cpp import Llama

INFERENCE_PROMPT = """
Generate ascii art that matches the following description.

### description:
{description}

### ascii visualization:
"""
lora_path = "src/local_models/adapters/gguf/cat-ascii-overfit-2-lora-f32.gguf"
model_path = "src/local_models/base_models/gguf/Meta-Llama-3.1-8B-bnb-4bit.gguf"
merged_model_path = (
    "src/local_models/base_models/gguf/cat-ascii-overfit-2-q8_0_merged.gguf"
)
use_lora = True

if use_lora:
    llm = Llama(model_path=model_path, lora_path=lora_path, verbose=True, n_ctx=131072)
else:
    llm = Llama(model_path=merged_model_path, verbose=True)


def generate_ascii_art(description: str, max_tokens: int) -> str:
    prompt = INFERENCE_PROMPT.format(description=description)
    output = ""
    print(prompt)
    completion = llm.create_completion(prompt, max_tokens=max_tokens)
    print(completion)


for _ in range(1):
    generate_ascii_art(description="cat", max_tokens=200)
