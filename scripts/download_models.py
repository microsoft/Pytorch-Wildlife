"""Download models from the Sparrow Model Zoo.

    python scripts/download_models.py MDV6-yolov10-c AI4G-Amazon-V2
    python scripts/download_models.py --list
"""

import argparse

import sparrow_engine as spe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", help="zoo model ids (default: the default set)")
    parser.add_argument("--list", action="store_true", help="list zoo models and exit")
    parser.add_argument("--dest", help="model directory (default: the engine model dir)")
    parser.add_argument("--force", action="store_true", help="re-download installed models")
    args = parser.parse_args()

    if args.list:
        for m in spe.zoo_models():
            print(m.id)
        return

    paths = spe.download_models(args.models or None, dest=args.dest, force=args.force)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
