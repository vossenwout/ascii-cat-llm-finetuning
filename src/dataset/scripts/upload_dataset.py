from pathlib import Path
import os
import pandas as pd
from datasets import Dataset, load_dataset
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


# pull dataset from huggingface and save as parquet
def pull_dataset(dataset_name: str, output_path: Path) -> Dataset:
    # Ensure the output path has .parquet extension
    if not str(output_path).endswith(".parquet"):
        output_path = Path(str(output_path) + ".parquet")

    dataset = load_dataset(dataset_name)
    df = dataset["train"].to_pandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    return dataset


if __name__ == "__main__":
    # upload_dataset(
    #    path=Path("src/dataset/out/ascii_art_cat_1.parquet"),
    #    dataset_name="ascii-art-animals",
    # )
    dataset = pull_dataset(
        dataset_name="pookie3000/ascii-art-animals",
        output_path=Path("src/dataset/out/pookie3000/ascii-art-animals"),
    )
