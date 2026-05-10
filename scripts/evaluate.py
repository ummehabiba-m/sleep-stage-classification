"""
scripts/evaluate.py
Load results.json and fail the pipeline if metrics are below thresholds.
"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--min-kappa", type=float, default=0.70)
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    kappa = results["test_kappa"]
    accuracy = results["test_accuracy"]
    stage_names = results.get("class_names", ["Wake", "N1", "N2", "N3", "REM"])
    f1_scores = results.get("per_class_f1", [])

    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Test Accuracy : {accuracy:.4f}  (threshold ≥ {args.min_accuracy})")
    print(f"Cohen's Kappa : {kappa:.4f}  (threshold ≥ {args.min_kappa})")

    if f1_scores:
        print("\nPer-Class F1 Scores:")
        for name, f1 in zip(stage_names, f1_scores):
            flag = "✓" if f1 >= 0.60 else "⚠"
            print(f"  {flag} {name:5s}: {f1:.4f}")

    print("=" * 60)

    failed = []
    if accuracy < args.min_accuracy:
        failed.append(
            f"Accuracy {accuracy:.4f} < threshold {args.min_accuracy}"
        )
    if kappa < args.min_kappa:
        failed.append(
            f"Kappa {kappa:.4f} < threshold {args.min_kappa}"
        )

    if failed:
        print("\n❌ EVALUATION FAILED — model does not meet quality gates:")
        for msg in failed:
            print(f"  • {msg}")
        sys.exit(1)
    else:
        print("\n✅ EVALUATION PASSED — model meets all quality gates")

    # Write markdown report for upload
    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/evaluation.md", "w") as f:
        f.write(f"# Evaluation Report\n\n")
        f.write(f"| Metric | Value | Threshold | Status |\n")
        f.write(f"|--------|-------|-----------|--------|\n")
        f.write(f"| Accuracy | {accuracy:.4f} | {args.min_accuracy} | {'✅' if accuracy >= args.min_accuracy else '❌'} |\n")
        f.write(f"| Kappa | {kappa:.4f} | {args.min_kappa} | {'✅' if kappa >= args.min_kappa else '❌'} |\n")


if __name__ == "__main__":
    main()
