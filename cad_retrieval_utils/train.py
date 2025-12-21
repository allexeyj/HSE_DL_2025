import argparse

from cad_retrieval_utils.utils import init_environment, load_config, wandb_init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train runner")
    parser.add_argument(
        "--config",
        required=True,
        help="Путь к .py-файлу, в котором объявлен CONFIG",
    )
    args = parser.parse_args()

    CONFIG = load_config(args.config)

    print(f"Using config: {args.config}")
    print(f"Device: {CONFIG.device}")

    init_environment(CONFIG)
    wandb_init(CONFIG)

    from cad_retrieval_utils import train_loop

    train_loop(CONFIG)


if __name__ == "__main__":
    main()
