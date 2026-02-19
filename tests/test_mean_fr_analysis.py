import os
import sys

import numpy as np
import pytest

base_dir = os.path.dirname(os.path.dirname(
    globals().get("__file__", os.getcwd())
))
sys.path.insert(0, base_dir)

from mean_fr_analysis import (
    load_series,
    compute_descriptive_stats,
    compute_normality_tests,
    compute_stationarity_tests,
    compute_autocorrelation,
    compute_trend,
    compute_outliers,
    compute_volatility,
    analyse,
    print_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
SAMPLE_SERIES = RNG.standard_normal(100).astype(np.float64)


# ---------------------------------------------------------------------------
# load_series
# ---------------------------------------------------------------------------

def test_load_series_reads_file(tmp_path):
    p = tmp_path / "test.txt"
    data = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
    np.savetxt(str(p), data, fmt="%.6f")

    result = load_series(str(p))
    np.testing.assert_allclose(result, data, atol=1e-5)


def test_load_series_raises_for_missing_file():
    with pytest.raises(Exception):
        load_series("/nonexistent/path/mean_fr.txt")


def test_load_series_raises_for_short_series(tmp_path):
    p = tmp_path / "short.txt"
    np.savetxt(str(p), np.array([1.0, 2.0, 3.0]), fmt="%.2f")
    with pytest.raises(ValueError, match="too short"):
        load_series(str(p))


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def test_descriptive_stats_basic():
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_descriptive_stats(series)

    assert stats.n == 5
    assert stats.mean == pytest.approx(3.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.minimum == pytest.approx(1.0)
    assert stats.maximum == pytest.approx(5.0)
    assert stats.q1 < stats.median < stats.q3


def test_descriptive_stats_std_positive():
    stats = compute_descriptive_stats(SAMPLE_SERIES)
    assert stats.std > 0
    assert stats.variance > 0


def test_descriptive_stats_iqr():
    stats = compute_descriptive_stats(SAMPLE_SERIES)
    assert stats.iqr == pytest.approx(stats.q3 - stats.q1)


def test_descriptive_stats_skewness_symmetric():
    series = np.linspace(-1, 1, 100)
    stats = compute_descriptive_stats(series)
    assert abs(stats.skewness) < 0.1


# ---------------------------------------------------------------------------
# Normality tests
# ---------------------------------------------------------------------------

def test_normality_returns_none_values_when_scipy_absent(monkeypatch):
    import mean_fr_analysis as mod
    monkeypatch.setattr(mod, "_SCIPY_AVAILABLE", False)
    result = mod.compute_normality_tests(SAMPLE_SERIES)
    assert result.shapiro_stat is None
    assert result.jb_stat is None
    assert result.dagostino_stat is None


def test_normality_detects_normal_distribution():
    rng = np.random.default_rng(42)
    normal_series = rng.standard_normal(200)
    result = compute_normality_tests(normal_series)
    if result.shapiro_p is not None:
        assert result.shapiro_p > 0.01


def test_normality_detects_non_normal():
    rng = np.random.default_rng(42)
    # Heavily skewed distribution
    skewed = rng.exponential(scale=1.0, size=200)
    result = compute_normality_tests(skewed)
    if result.jb_p is not None:
        assert result.jb_p < 0.05


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

def test_stationarity_returns_none_when_statsmodels_absent(monkeypatch):
    import mean_fr_analysis as mod
    monkeypatch.setattr(mod, "_STATSMODELS_AVAILABLE", False)
    result = mod.compute_stationarity_tests(SAMPLE_SERIES)
    assert result.adf_stat is None
    assert result.kpss_stat is None


def test_stationarity_white_noise_is_stationary():
    rng = np.random.default_rng(42)
    series = rng.standard_normal(200)
    result = compute_stationarity_tests(series)
    if result.adf_is_stationary is not None:
        assert result.adf_is_stationary is True


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def test_autocorrelation_lag0_is_one():
    result = compute_autocorrelation(SAMPLE_SERIES, nlags=10)
    if result.acf_values is not None:
        assert result.acf_values[0] == pytest.approx(1.0, abs=1e-6)


def test_autocorrelation_returns_none_when_statsmodels_absent(monkeypatch):
    import mean_fr_analysis as mod
    monkeypatch.setattr(mod, "_STATSMODELS_AVAILABLE", False)
    result = mod.compute_autocorrelation(SAMPLE_SERIES)
    assert result.acf_values is None
    assert result.pacf_values is None


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def test_trend_flat_series_slope_near_zero():
    series = np.ones(50)
    result = compute_trend(series)
    assert abs(result.slope) < 1e-10


def test_trend_increasing_series():
    series = np.arange(100, dtype=np.float64)
    result = compute_trend(series)
    assert result.slope == pytest.approx(1.0, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


def test_trend_p_value_significant_for_strong_trend():
    series = np.arange(200, dtype=np.float64) + RNG.standard_normal(200) * 0.01
    result = compute_trend(series)
    if result.p_value is not None:
        assert result.p_value < 0.05


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def test_outliers_detected_iqr():
    series = np.concatenate([np.zeros(50), [100.0]])
    result = compute_outliers(series)
    assert 50 in result.iqr_outlier_indices


def test_outliers_detected_zscore():
    series = np.concatenate([np.zeros(50), [100.0]])
    result = compute_outliers(series, zscore_threshold=3.0)
    assert 50 in result.zscore_outlier_indices


def test_no_outliers_clean_series():
    series = np.linspace(0.0, 1.0, 100)
    result = compute_outliers(series)
    assert len(result.iqr_outlier_indices) == 0


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def test_volatility_rolling_std_length():
    result = compute_volatility(SAMPLE_SERIES, window=10)
    assert len(result.rolling_std) == len(SAMPLE_SERIES)


def test_volatility_rolling_std_positive():
    result = compute_volatility(SAMPLE_SERIES, window=10)
    assert np.all(result.rolling_std >= 0)


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def test_analyse_returns_report():
    report = analyse(SAMPLE_SERIES)
    assert report.descriptive.n == len(SAMPLE_SERIES)
    assert report.descriptive.std > 0


def test_analyse_with_mean_fr_file():
    path = os.path.join(base_dir, "mean_fr.txt")
    if not os.path.exists(path):
        pytest.skip("mean_fr.txt not present")
    series = load_series(path)
    report = analyse(series)
    assert report.descriptive.n == 252


# ---------------------------------------------------------------------------
# print_report smoke test
# ---------------------------------------------------------------------------

def test_print_report_runs_without_error(capsys):
    report = analyse(SAMPLE_SERIES)
    print_report(report)
    captured = capsys.readouterr()
    assert "DESCRIPTIVE STATISTICS" in captured.out
    assert "NORMALITY TESTS" in captured.out
    assert "STATIONARITY TESTS" in captured.out
    assert "OUTLIER DETECTION" in captured.out
    assert "VOLATILITY" in captured.out
