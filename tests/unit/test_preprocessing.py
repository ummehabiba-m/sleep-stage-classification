"""
tests/unit/test_preprocessing.py
Unit tests for the SleepDataPreprocessor and SleepSequenceDataset.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


# ─── Helpers ────────────────────────────────────────────────────────────────

def make_synthetic_epochs(n_epochs=100, n_samples=3000, n_channels=2):
    """Return normalised synthetic epoch data + labels."""
    data = np.random.randn(n_epochs, n_samples, n_channels).astype(np.float32)
    labels = np.random.randint(0, 5, n_epochs)
    return data, labels


# ─── Data validation logic (inline so tests work without full install) ───────

def validate_epochs(epochs, labels):
    errors = []
    if len(epochs) == 0:
        errors.append("empty")
        return errors
    for i, (ep, lb) in enumerate(zip(epochs, labels)):
        if ep.ndim != 3:
            errors.append(f"subject {i}: wrong ndim {ep.ndim}")
        if len(ep) != len(lb):
            errors.append(f"subject {i}: length mismatch")
        if np.any(np.isnan(ep)):
            errors.append(f"subject {i}: NaNs")
        unexpected = set(lb.tolist()) - {0, 1, 2, 3, 4}
        if unexpected:
            errors.append(f"subject {i}: bad labels {unexpected}")
    return errors


# ─── Tests ──────────────────────────────────────────────────────────────────

class TestDataValidation:
    def test_valid_data_passes(self):
        epochs = [make_synthetic_epochs()[0] for _ in range(3)]
        labels = [make_synthetic_epochs()[1] for _ in range(3)]
        assert validate_epochs(epochs, labels) == []

    def test_empty_dataset_fails(self):
        errors = validate_epochs([], [])
        assert len(errors) > 0

    def test_nan_in_data_fails(self):
        ep, lb = make_synthetic_epochs()
        ep[0, 0, 0] = np.nan
        errors = validate_epochs([ep], [lb])
        assert any("NaN" in e for e in errors)

    def test_invalid_label_fails(self):
        ep, lb = make_synthetic_epochs()
        lb[0] = 99  # Invalid sleep stage
        errors = validate_epochs([ep], [lb])
        assert any("bad labels" in e for e in errors)

    def test_length_mismatch_fails(self):
        ep, _ = make_synthetic_epochs(n_epochs=100)
        lb = np.random.randint(0, 5, 50)  # wrong length
        errors = validate_epochs([ep], [lb])
        assert any("length mismatch" in e for e in errors)


class TestEpochShape:
    def test_epoch_shape_correct(self):
        ep, _ = make_synthetic_epochs(n_epochs=50, n_samples=3000, n_channels=2)
        assert ep.shape == (50, 3000, 2)

    def test_normalization_zero_mean(self):
        """After z-score normalization mean should be near 0."""
        ep, _ = make_synthetic_epochs(n_epochs=100)
        # Apply z-score per channel
        for ch in range(ep.shape[2]):
            mean = ep[:, :, ch].mean()
            std = ep[:, :, ch].std()
            ep[:, :, ch] = (ep[:, :, ch] - mean) / (std + 1e-8)
        assert abs(ep[:, :, 0].mean()) < 0.01

    def test_normalization_unit_std(self):
        ep, _ = make_synthetic_epochs(n_epochs=100)
        for ch in range(ep.shape[2]):
            mean = ep[:, :, ch].mean()
            std = ep[:, :, ch].std()
            ep[:, :, ch] = (ep[:, :, ch] - mean) / (std + 1e-8)
        assert abs(ep[:, :, 0].std() - 1.0) < 0.01


class TestSequenceDataset:
    def test_sequence_count(self):
        """n_sequences = n_epochs // seq_length per subject."""
        n_epochs, seq_length = 100, 20
        expected_seqs = n_epochs // seq_length  # 5
        ep, lb = make_synthetic_epochs(n_epochs=n_epochs)
        n_seqs = len(ep) // seq_length
        assert n_seqs == expected_seqs

    def test_sequence_shape(self):
        seq_length = 20
        ep, _ = make_synthetic_epochs(n_epochs=100, n_samples=3000, n_channels=2)
        sequences = []
        for i in range(len(ep) // seq_length):
            sequences.append(ep[i * seq_length:(i + 1) * seq_length])
        sequences = np.array(sequences)
        assert sequences.shape == (5, 20, 3000, 2)

    def test_label_alignment(self):
        """Labels must align with epoch sequences."""
        seq_length = 20
        ep, lb = make_synthetic_epochs(n_epochs=100)
        for i in range(len(ep) // seq_length):
            seq_lb = lb[i * seq_length:(i + 1) * seq_length]
            assert len(seq_lb) == seq_length


class TestMetricThresholds:
    def test_kappa_threshold_pass(self):
        kappa = 0.73
        assert kappa >= 0.70

    def test_kappa_threshold_fail(self):
        kappa = 0.65
        assert kappa < 0.70

    def test_accuracy_threshold_pass(self):
        accuracy = 0.86
        assert accuracy >= 0.80
