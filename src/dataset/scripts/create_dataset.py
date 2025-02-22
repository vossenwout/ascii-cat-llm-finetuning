import json
from pathlib import Path
import pandas as pd


def create_dataset(path: Path, version_tag: str) -> None:
    paths = []
    for file in path.glob("**/*.txt"):
        txt_file = file
        meta_file = file.parent / "meta.json"
        paths.append({"ascii": txt_file, "meta": meta_file})
    paths.sort(
        key=lambda x: (x["ascii"].parent.parents[0], int(x["ascii"].parent.name))
    )

    dataset_items = []
    for path in paths:
        with open(path["ascii"], "r") as f:
            ascii_art = f.read()
        with open(path["meta"], "r") as f:
            meta = json.load(f)
        dataset_items.append(
            {
                "ascii": ascii_art,
                "creature": str(path["ascii"].parent.parent.name),
                "local_path": str(path["ascii"]),
                **meta,
            }
        )

    df = pd.DataFrame(dataset_items)

    output_path = Path(f"src/dataset/out/ascii_art_{version_tag}.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Dataset successfully saved to {output_path}")


if __name__ == "__main__":
    create_dataset(path=Path("src/dataset/ascii_art/animals/cat"), version_tag="cat_3")
