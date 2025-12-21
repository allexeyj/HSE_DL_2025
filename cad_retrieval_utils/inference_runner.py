import argparse
from pathlib import Path

from cad_retrieval_utils.utils import init_environment, load_config


def parse_model_specs(model_spec_args: list[str]) -> list[dict[str, str]]:
    """
    Парсит аргументы модели в формате moe_path:pc_encoder_path

    Args:
        model_spec_args: Список строк в формате "moe_path:pc_encoder_path"

    Returns:
        Список словарей с ключами "moe" и "pc_encoder"
    """
    model_specs = []
    for spec in model_spec_args:
        if ":" not in spec:
            raise ValueError(
                f"Неверный формат спецификации модели: {spec}. "
                "Ожидается формат: moe_path:pc_encoder_path"
            )
        moe_path, pc_encoder_path = spec.split(":", 1)

        if not Path(moe_path).exists():
            raise FileNotFoundError(f"MoE checkpoint не найден: {moe_path}")
        if not Path(pc_encoder_path).exists():
            raise FileNotFoundError(
                f"PC encoder checkpoint не найден: {pc_encoder_path}"
            )

        model_specs.append({"moe": moe_path, "pc_encoder": pc_encoder_path})

    return model_specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Путь к .py-файлу, в котором объявлен CONFIG",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        dest="models",
        help="Пара путей к чекпоинтам в формате moe_path:pc_encoder_path. "
        "Можно указать несколько раз для ансамбля.",
    )
    parser.add_argument(
        "--output",
        help="Путь для сохранения submission файла",
        required=True,
    )

    args = parser.parse_args()

    CONFIG = load_config(args.config)

    print(f"Using config: {args.config}")
    print(f"Device: {CONFIG.device}")

    # Парсим и добавляем спецификации моделей в CONFIG
    try:
        model_specs = parse_model_specs(args.models)
        CONFIG.paths.ensemble_model_specs = model_specs

        print(f"\n📊 Загружено спецификаций моделей: {len(model_specs)}")
        for i, spec in enumerate(model_specs, 1):
            print(f"  Модель {i}:")
            print(f"    MoE: {spec['moe']}")
            print(f"    PC Encoder: {spec['pc_encoder']}")
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Ошибка: {e}")
        exit(1)

    # Если указан output путь, перезаписываем конфиг
    if args.output:
        CONFIG.paths.ensemble_save_file = Path(args.output)
        print(f"\n📁 Результат будет сохранен в: {CONFIG.paths.ensemble_save_file}")

    init_environment(CONFIG)

    from cad_retrieval_utils import make_submission

    make_submission(CONFIG)


if __name__ == "__main__":
    main()
