"""
Unit coverage for app/utils/sampling.py — the shared helper introduced to
cap interactive-chart payloads (Visualization performance work). These are
pure-function tests with no HTTP/DB involved, checking the properties that
matter for correctness: no-op under the cap, index-aligned sampling across
paired series, and clear failure on mismatched lengths.
"""
import pytest

from app.utils.sampling import sample_paired_series, thin_curve


class TestSamplePairedSeries:
    def test_no_op_when_under_cap(self):
        actual = [1.0, 2.0, 3.0]
        predicted = [1.1, 2.1, 3.1]
        result = sample_paired_series(max_points=10, actual=actual, predicted=predicted)
        assert result["actual"] == actual
        assert result["predicted"] == predicted
        assert result["sampled"] is False
        assert result["sample_size"] == 3
        assert result["total_size"] == 3

    def test_caps_and_reports_sizes_when_over_cap(self):
        n = 500
        actual = list(range(n))
        predicted = [v * 2 for v in actual]
        result = sample_paired_series(max_points=50, actual=actual, predicted=predicted)
        assert result["sampled"] is True
        assert result["sample_size"] == 50
        assert result["total_size"] == n
        assert len(result["actual"]) == 50
        assert len(result["predicted"]) == 50

    def test_paired_series_stay_index_aligned(self):
        # predicted[i] must always equal actual[i] * 2 for every sampled
        # point — proves sampling picks the SAME indices across series
        # rather than sampling each independently.
        n = 1000
        actual = list(range(n))
        predicted = [v * 2 for v in actual]
        result = sample_paired_series(max_points=37, actual=actual, predicted=predicted)
        for a, p in zip(result["actual"], result["predicted"]):
            assert p == a * 2

    def test_deterministic_across_calls(self):
        actual = list(range(200))
        predicted = list(range(200))
        r1 = sample_paired_series(max_points=20, actual=actual, predicted=predicted)
        r2 = sample_paired_series(max_points=20, actual=actual, predicted=predicted)
        assert r1["actual"] == r2["actual"]

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            sample_paired_series(actual=[1, 2, 3], predicted=[1, 2])

    def test_empty_series(self):
        result = sample_paired_series(actual=[], predicted=[])
        assert result == {"actual": [], "predicted": [], "sampled": False, "sample_size": 0, "total_size": 0}


class TestThinCurve:
    def test_no_op_when_under_cap(self):
        fpr = [0.0, 0.5, 1.0]
        tpr = [0.0, 0.6, 1.0]
        result = thin_curve(max_points=10, fpr=fpr, tpr=tpr)
        assert result["fpr"] == fpr
        assert result["tpr"] == tpr
        assert result["sampled"] is False

    def test_thins_and_keeps_endpoints(self):
        n = 5000
        fpr = [i / (n - 1) for i in range(n)]
        tpr = [i / (n - 1) for i in range(n)]
        result = thin_curve(max_points=100, fpr=fpr, tpr=tpr)
        assert result["sampled"] is True
        assert result["total_size"] == n
        assert len(result["fpr"]) <= 101  # endpoint de-dupe/append may add at most one
        assert result["fpr"][0] == pytest.approx(0.0)
        assert result["fpr"][-1] == pytest.approx(1.0)
        # Curve stays monotonic non-decreasing after thinning.
        assert result["fpr"] == sorted(result["fpr"])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            thin_curve(fpr=[0.0, 1.0], tpr=[0.0])
