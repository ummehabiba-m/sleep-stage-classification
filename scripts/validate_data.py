"""
scripts/validate_data.py
Validate that the preprocessing pipeline produces well-formed outputs.
"""
import argparse
import numpy as np
import sys


def validate_epochs(epochs: list, labels: list) -> list[str]:
    errors = []

    if len(epochs) == 0:
        errors.append("all_epochs is empty — preprocessing produced no data")
        return errors

    if len(epochs) != len(labels):
        errors.append(
            f"Mismatch: {len(epochs)} epoch arrays vs {len(labels)} label arrays"
        )

    for i, (ep, lb) in enumerate(zip(epochs, labels)):
        if ep is None or lb is None:
            errors.append(f"Subject {i}: got None output from preprocessor")
            continue
        if ep.ndim != 3:
            errors.append(f"Subject {i}: expected 3-D epoch array, got shape {ep.shape}")
        if len(ep) != len(lb):
            errors.append(
                f"Subject {i}: {len(ep)} epochs but {len(lb)} labels"
            )
        if np.any(np.isnan(ep)):
            errors.append(f"Subject {i}: NaNs found in epoch data")
        if np.any(np.isinf(ep)):
            errors.append(f"Subject {i}: Infs found in epoch data")
        valid_stages = {0, 1, 2, 3, 4}
        unexpected = set(lb.tolist()) - valid_stages
        if unexpected:
            errors.append(f"Subject {i}: unexpected label values {unexpected}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline on synthetic data (no real EEG files needed)",
    )
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    if args.dry_run:
        print("Running data validation on synthetic data...")
        # Synthetic: 3 subjects, 100 epochs each, 30s @ 100 Hz, 2 channels
        epochs = [np.random.randn(100, 3000, 2).astype(np.float32) for _ in range(3)]
        labels = [np.random.randint(0, 5, 100) for _ in range(3)]
    else:
        import pickle, os
        path = os.path.join(args.data_dir, "processed_data.pkl")
        print(f"Loading data from {path}...")
        with open(path, "rb") as f:
            data = pickle.load(f)
        epochs = data["all_epochs"]
        labels = data["all_labels"]

    errors = validate_epochs(epochs, labels)

    if errors:
        print("\n❌ Data validation FAILED:")
        for e in errors:
            print(f"  • {e}")
        sys.exit(1)
    else:
        print(f"✓ Data validation passed ({len(epochs)} subjects)")


if __name__ == "__main__":
    main()
