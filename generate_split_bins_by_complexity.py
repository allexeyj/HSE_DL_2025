#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default="embeddings/train_img_embeddings_stage1.npz")
    parser.add_argument("--complexity-csv", type=Path, default="csv_files/train_complexity_score.csv")
    parser.add_argument("--output", type=Path, default="splits/complexity_based_split.csv")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=20)
    args = parser.parse_args()

    # Загружаем IDs из эмбеддингов
    data = np.load(args.embeddings)
    model_ids = sorted(set(f.split('_')[0] for f in data["filenames"]))
    model_ids = [int(mid) for mid in model_ids]

    # Загружаем complexity scores
    complexity_df = pd.read_csv(args.complexity_csv)
    complexity_map = dict(zip(complexity_df['model_id_int'], complexity_df['complexity_score']))

    # Матчим
    valid_ids = [mid for mid in model_ids if mid in complexity_map]
    complexities = np.array([complexity_map[mid] for mid in valid_ids])

    print(f"Total models: {len(valid_ids)}")
    print(
        f"Complexity stats: min={complexities.min():.6f}, median={np.median(complexities):.2f}, max={complexities.max():.2f}")

    # Логарифмическое биннинг для сильно скошенного распределения
    log_complexities = np.log(complexities)
    bins = pd.qcut(log_complexities, q=args.n_bins, labels=False, duplicates='drop')

    print(f"Bins created: {len(np.unique(bins))}")

    # Сплит
    train_ids, val_ids, train_comp, val_comp = train_test_split(
        valid_ids,
        complexities,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=bins
    )

    print(f"Train: {len(train_ids)} models, complexity median={np.median(train_comp):.2f}")
    print(f"Val: {len(val_ids)} models, complexity median={np.median(val_comp):.2f}")

    # Сохраняем
    df = pd.concat([
        pd.DataFrame({'model_id': train_ids, 'split': 'train'}),
        pd.DataFrame({'model_id': val_ids, 'split': 'validation'})
    ]).sort_values('model_id').reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()