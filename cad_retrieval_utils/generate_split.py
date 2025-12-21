import pandas as pd
import numpy as np
from pathlib import Path


def generate_train_val_split(n_models=525, val_ratio=0.2, seed=42):
    """
    Генерирует train/val split для модели IDs от 0000 до 0524
    """
    np.random.seed(seed)

    # Создаем список всех model_ids
    model_ids = [f"{i:04d}" for i in range(n_models)]

    # Перемешиваем и делим
    indices = np.random.permutation(n_models)
    n_val = int(n_models * val_ratio)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Создаем DataFrame
    data = []
    for idx in train_indices:
        data.append({"model_id": model_ids[idx], "split": "train"})
    for idx in val_indices:
        data.append({"model_id": model_ids[idx], "split": "validation"})

    df = pd.DataFrame(data)
    df = df.sort_values("model_id").reset_index(drop=True)

    # Сохраняем
    output_path = Path("../splits/text_split.csv")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Split сохранен в {output_path}")
    print(f"   Train: {len(train_indices)} samples")
    print(f"   Val: {len(val_indices)} samples")

    return df


if __name__ == "__main__":
    generate_train_val_split()