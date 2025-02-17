from pathlib import Path

import pandas as pd


def visualize_parquet_dataset(dataset_parquet_path: Path) -> None:
    df = pd.read_parquet(dataset_parquet_path)
    ascii_art_string = ""
    for index, row in df.iterrows():
        ascii_art_string += f"index: {index}\n"
        for column in row.index:
            ascii_art_string += f"{column}:\n\n"
            ascii_art_string += f"{row[column]}\n\n"
        ascii_art_string += f"--------------------------------\n"

    with open(dataset_parquet_path.with_suffix(".txt"), "w") as f:
        f.write(ascii_art_string)

    print(f"Visulization saved to {dataset_parquet_path.with_suffix('.txt')}")


if __name__ == "__main__":
    dataset_parquet_path = Path("src/dataset/out/ascii_art_cat_2.parquet")
    visualize_parquet_dataset(dataset_parquet_path)
