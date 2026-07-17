from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    test_ratio = 1.0 - train_ratio - val_ratio
    train_val, test = train_test_split(
        df, test_size=test_ratio, stratify=df["target"], random_state=seed,
    )
    relative_val = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["target"], random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_splits(dataset: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    out_dir = DATA_DIR / "processed" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_json(out_dir / "train.jsonl", orient="records", lines=True)
    val.to_json(out_dir / "val.jsonl", orient="records", lines=True)
    test.to_json(out_dir / "test.jsonl", orient="records", lines=True)
    print(f"{dataset}: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    for dataset in ["german", "loans"]:
        split_dir = DATA_DIR / "processed" / dataset
        if not split_dir.exists():
            raise FileNotFoundError(f"{split_dir} not found. Run preprocess.py first.")
        for split in ["train", "val", "test"]:
            path = split_dir / f"{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Run preprocess.py first.")
        df = pd.concat(
            [pd.read_json(split_dir / f"{s}.jsonl", lines=True) for s in ["train", "val", "test"]],
            ignore_index=True,
        )
        train, val, test = stratified_split(df)
        save_splits(dataset, train, val, test)
