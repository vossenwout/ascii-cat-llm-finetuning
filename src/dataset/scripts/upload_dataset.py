from pathlib import Path
import os
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv(
    "config/.env",
)


# upload dataset to huggingface
def upload_dataset(path: Path, dataset_name: str) -> None:
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN environment variable not set")
    login(token=hf_token)
    df = pd.read_parquet(path)
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(dataset_name)


if __name__ == "__main__":
    upload_dataset(
        path=Path("src/dataset/out/ascii_art_cat_1.parquet"),
        dataset_name="ascii-art-animals",
    )
