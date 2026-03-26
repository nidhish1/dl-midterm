import argparse
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a LoRA adapter/checkpoint tar.gz.")
    parser.add_argument("--tar_path", type=Path, required=True, help="Path to checkpoint tar.gz")
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory where files should be extracted",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.tar_path, "r:gz") as tf:
        tf.extractall(args.out_dir)

    print(f"Extracted {args.tar_path} -> {args.out_dir}")


if __name__ == "__main__":
    main()
