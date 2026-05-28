from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one exported NPZ state file."
    )

    parser.add_argument(
        "state_file",
        type=str,
        help="Path to .npz state file.",
    )

    args = parser.parse_args()

    state_file = Path(args.state_file)

    if not state_file.exists():
        raise FileNotFoundError(f"State file not found: {state_file}")

    data = np.load(state_file, allow_pickle=False)

    print("=" * 100)
    print("NPZ state check")
    print("=" * 100)

    print(f"File: {state_file.resolve()}")

    for key in data.files:
        value = data[key]

        if key.endswith("_json"):
            print(f"\n{key}:")
            print(json.loads(str(value)))
        else:
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")

    print("\nDone.")


if __name__ == "__main__":
    main()