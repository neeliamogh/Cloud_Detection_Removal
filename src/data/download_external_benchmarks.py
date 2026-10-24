from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cloudsen12", "sen12mscr", "sparcs"], required=True)
    parser.parse_args()
    print(
        "External benchmark download is intentionally credential-aware. "
        "Use the dataset owners' terms, Kaggle/Earthdata credentials where required, "
        "and place converted patches under data/external/<dataset>. "
        "The evaluation code can be extended to read those patches with the same NPZ schema."
    )


if __name__ == "__main__":
    main()

