import argparse
from pathlib import Path

from cad_retrieval_utils.utils import init_environment, load_config


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
        "--text_proj",
        required=True,
        help="Путь к чекпоинту text projection",
    )
    parser.add_argument(
        "--text_encoder",
        default=None,
        help="Путь к чекпоинту text encoder (если были разморожены слои)",
    )
    parser.add_argument(
        "--pc_encoder",
        required=True,
        help="Путь к чекпоинту PC encoder",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Путь для сохранения submission файла",
    )

    args = parser.parse_args()

    CONFIG = load_config(args.config)

    print(f"Using config: {args.config}")
    print(f"Device: {CONFIG.device}")

    # Добавляем пути моделей в конфиг
    CONFIG.paths.model_spec = {
        "text_proj": args.text_proj,
        "text_encoder": args.text_encoder,  # Может быть None
        "pc_encoder": args.pc_encoder,
    }
    CONFIG.paths.submission_save_file = Path(args.output)

    # Проверяем существование обязательных файлов
    if not Path(args.text_proj).exists():
        raise FileNotFoundError(f"Text projection checkpoint not found: {args.text_proj}")
    if not Path(args.pc_encoder).exists():
        raise FileNotFoundError(f"PC encoder checkpoint not found: {args.pc_encoder}")

    # Проверяем text_encoder если указан
    if args.text_encoder:
        if not Path(args.text_encoder).exists():
            raise FileNotFoundError(f"Text encoder checkpoint not found: {args.text_encoder}")
        print(f"✅ Will load unfrozen text encoder weights from: {args.text_encoder}")
    else:
        print(f"ℹ️  No text encoder weights specified (using frozen pretrained)")

    init_environment(CONFIG)

    from cad_retrieval_utils import make_submission

    make_submission(CONFIG)


if __name__ == "__main__":
    main()