"""
Detailed statistical analysis of the mean_fr.txt time series.

The file is expected to contain one floating-point value per line representing
daily mean financial returns.

Usage:
    python mean_fr_analysis.py [--path mean_fr.txt] [--plot]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

try:
    from scipy import stats as scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.stats.diagnostic import het_arch
    _STATSMODELS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STATSMODELS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_series(path: str) -> npt.NDArray[np.float64]:
    """Load a time series from a plain-text file (one value per line)."""
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(
            f"Expected a 1-D time series in '{path}', got shape {data.shape}"
        )
    if len(data) < 4:
        raise ValueError(
            f"Time series is too short for statistical analysis (got {len(data)} points)"
        )
    return data


# ---------------------------------------------------------------------------
# Analysis results container
# ---------------------------------------------------------------------------

@dataclass
class DescriptiveStats:
    n: int
    mean: float
    median: float
    std: float
    variance: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    cv: float  # coefficient of variation


@dataclass
class NormalityResults:
    """Results from normality tests (require scipy)."""
    shapiro_stat: Optional[float] = None
    shapiro_p: Optional[float] = None
    jb_stat: Optional[float] = None
    jb_p: Optional[float] = None
    dagostino_stat: Optional[float] = None
    dagostino_p: Optional[float] = None


@dataclass
class StationarityResults:
    """Results from stationarity tests (require statsmodels)."""
    adf_stat: Optional[float] = None
    adf_p: Optional[float] = None
    adf_is_stationary: Optional[bool] = None
    kpss_stat: Optional[float] = None
    kpss_p: Optional[float] = None
    kpss_is_stationary: Optional[bool] = None


@dataclass
class AutocorrelationResults:
    """Autocorrelation function values (require statsmodels)."""
    acf_values: Optional[npt.NDArray[np.float64]] = None
    pacf_values: Optional[npt.NDArray[np.float64]] = None
    nlags: int = 20


@dataclass
class TrendResults:
    """Results from a simple linear trend fit."""
    slope: float
    intercept: float
    r_squared: float
    p_value: Optional[float] = None


@dataclass
class OutlierResults:
    """Outlier detection using the IQR and Z-score methods."""
    iqr_outlier_indices: npt.NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    iqr_outlier_values: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    zscore_outlier_indices: npt.NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    zscore_outlier_values: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    zscore_threshold: float = 3.0


@dataclass
class VolatilityResults:
    """Rolling volatility statistics and ARCH-effect test (require statsmodels)."""
    rolling_std: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    rolling_window: int = 20
    arch_stat: Optional[float] = None
    arch_p: Optional[float] = None
    arch_effect_present: Optional[bool] = None


@dataclass
class AnalysisReport:
    descriptive: DescriptiveStats
    normality: NormalityResults
    stationarity: StationarityResults
    autocorrelation: AutocorrelationResults
    trend: TrendResults
    outliers: OutlierResults
    volatility: VolatilityResults


# ---------------------------------------------------------------------------
# Individual analysis functions
# ---------------------------------------------------------------------------

def compute_descriptive_stats(series: npt.NDArray[np.float64]) -> DescriptiveStats:
    """Compute basic descriptive statistics for the time series."""
    n = len(series)
    mean = float(np.mean(series))
    median = float(np.median(series))
    std = float(np.std(series, ddof=1))
    variance = float(np.var(series, ddof=1))
    minimum = float(np.min(series))
    maximum = float(np.max(series))
    q1 = float(np.percentile(series, 25))
    q3 = float(np.percentile(series, 75))
    iqr = q3 - q1

    # Skewness and excess kurtosis (Fisher)
    skewness = float(_skewness(series))
    kurtosis = float(_kurtosis(series))

    cv = (std / abs(mean)) if mean != 0 else float("nan")

    return DescriptiveStats(
        n=n,
        mean=mean,
        median=median,
        std=std,
        variance=variance,
        minimum=minimum,
        maximum=maximum,
        q1=q1,
        q3=q3,
        iqr=iqr,
        skewness=skewness,
        kurtosis=kurtosis,
        cv=cv,
    )


def _skewness(x: npt.NDArray[np.float64]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    m2 = np.mean((x - np.mean(x)) ** 2)
    m3 = np.mean((x - np.mean(x)) ** 3)
    if m2 == 0:
        return float("nan")
    return float(m3 / (m2 ** 1.5))


def _kurtosis(x: npt.NDArray[np.float64]) -> float:
    """Return excess kurtosis (Fisher definition, normal = 0)."""
    n = len(x)
    if n < 4:
        return float("nan")
    m2 = np.mean((x - np.mean(x)) ** 2)
    m4 = np.mean((x - np.mean(x)) ** 4)
    if m2 == 0:
        return float("nan")
    return float(m4 / (m2 ** 2) - 3)


def compute_normality_tests(series: npt.NDArray[np.float64]) -> NormalityResults:
    """Run Shapiro-Wilk, Jarque-Bera, and D'Agostino normality tests."""
    result = NormalityResults()
    if not _SCIPY_AVAILABLE:
        return result

    # Shapiro-Wilk (works best for n <= 5000)
    sw_stat, sw_p = scipy_stats.shapiro(series)
    result.shapiro_stat = float(sw_stat)
    result.shapiro_p = float(sw_p)

    # Jarque-Bera
    jb_stat, jb_p = scipy_stats.jarque_bera(series)
    result.jb_stat = float(jb_stat)
    result.jb_p = float(jb_p)

    # D'Agostino K^2
    dag_stat, dag_p = scipy_stats.normaltest(series)
    result.dagostino_stat = float(dag_stat)
    result.dagostino_p = float(dag_p)

    return result


def compute_stationarity_tests(series: npt.NDArray[np.float64]) -> StationarityResults:
    """Run ADF and KPSS stationarity tests."""
    result = StationarityResults()
    if not _STATSMODELS_AVAILABLE:
        return result

    # Augmented Dickey-Fuller
    adf_out = adfuller(series, autolag="AIC")
    result.adf_stat = float(adf_out[0])
    result.adf_p = float(adf_out[1])
    result.adf_is_stationary = result.adf_p < 0.05

    # KPSS (null hypothesis: series is stationary)
    kpss_out = kpss(series, regression="c", nlags="auto")
    result.kpss_stat = float(kpss_out[0])
    result.kpss_p = float(kpss_out[1])
    # p-value < 0.05 → reject H0 (not stationary)
    result.kpss_is_stationary = result.kpss_p >= 0.05

    return result


def compute_autocorrelation(
    series: npt.NDArray[np.float64],
    nlags: int = 20,
) -> AutocorrelationResults:
    """Compute ACF and PACF up to *nlags* lags."""
    result = AutocorrelationResults(nlags=nlags)
    if not _STATSMODELS_AVAILABLE:
        return result

    max_lags = min(nlags, len(series) // 2 - 1)
    result.acf_values = acf(series, nlags=max_lags, fft=True)
    result.pacf_values = pacf(series, nlags=max_lags, method="ywm")
    return result


def compute_trend(series: npt.NDArray[np.float64]) -> TrendResults:
    """Fit a linear trend and return slope, intercept, and R²."""
    n = len(series)
    t = np.arange(n, dtype=np.float64)
    # Least-squares fit
    slope, intercept = np.polyfit(t, series, 1)

    # R-squared
    fitted = slope * t + intercept
    ss_res = np.sum((series - fitted) ** 2)
    ss_tot = np.sum((series - np.mean(series)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

    p_value: Optional[float] = None
    if _SCIPY_AVAILABLE:
        _, _, _, p_val, _ = scipy_stats.linregress(t, series)
        p_value = float(p_val)

    return TrendResults(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        p_value=p_value,
    )


def compute_outliers(
    series: npt.NDArray[np.float64],
    zscore_threshold: float = 3.0,
) -> OutlierResults:
    """Detect outliers using IQR fence and Z-score methods."""
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_mask = (series < lower) | (series > upper)
    iqr_idx = np.where(iqr_mask)[0]

    std = np.std(series, ddof=1)
    mean = np.mean(series)
    if std > 0:
        zscores = np.abs((series - mean) / std)
        z_mask = zscores > zscore_threshold
    else:
        z_mask = np.zeros(len(series), dtype=bool)
    z_idx = np.where(z_mask)[0]

    return OutlierResults(
        iqr_outlier_indices=iqr_idx,
        iqr_outlier_values=series[iqr_idx],
        zscore_outlier_indices=z_idx,
        zscore_outlier_values=series[z_idx],
        zscore_threshold=zscore_threshold,
    )


def compute_volatility(
    series: npt.NDArray[np.float64],
    window: int = 20,
) -> VolatilityResults:
    """Compute rolling standard deviation and test for ARCH effects."""
    n = len(series)
    result = VolatilityResults(rolling_window=window)

    # Rolling std (numpy implementation for no-pandas dependency)
    rolling_stds = []
    for i in range(n):
        start = max(0, i - window + 1)
        rolling_stds.append(float(np.std(series[start : i + 1], ddof=min(1, i - start))))
    result.rolling_std = np.array(rolling_stds, dtype=np.float64)

    if _STATSMODELS_AVAILABLE:
        try:
            arch_lm, arch_p, _, _ = het_arch(series, nlags=5)
            result.arch_stat = float(arch_lm)
            result.arch_p = float(arch_p)
            result.arch_effect_present = arch_p < 0.05
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyse(
    series: npt.NDArray[np.float64],
    nlags: int = 20,
    zscore_threshold: float = 3.0,
    volatility_window: int = 20,
) -> AnalysisReport:
    """Run all statistical analyses on *series* and return an :class:`AnalysisReport`."""
    return AnalysisReport(
        descriptive=compute_descriptive_stats(series),
        normality=compute_normality_tests(series),
        stationarity=compute_stationarity_tests(series),
        autocorrelation=compute_autocorrelation(series, nlags=nlags),
        trend=compute_trend(series),
        outliers=compute_outliers(series, zscore_threshold=zscore_threshold),
        volatility=compute_volatility(series, window=volatility_window),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "N/A (library not installed)"
    return f"{value:.{digits}f}"


def print_report(report: AnalysisReport) -> None:
    """Print a human-readable summary of the analysis report."""
    d = report.descriptive
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(f"  Observations : {d.n}")
    print(f"  Mean         : {_fmt(d.mean)}")
    print(f"  Median       : {_fmt(d.median)}")
    print(f"  Std Dev      : {_fmt(d.std)}")
    print(f"  Variance     : {_fmt(d.variance)}")
    print(f"  Min          : {_fmt(d.minimum)}")
    print(f"  Max          : {_fmt(d.maximum)}")
    print(f"  Q1           : {_fmt(d.q1)}")
    print(f"  Q3           : {_fmt(d.q3)}")
    print(f"  IQR          : {_fmt(d.iqr)}")
    print(f"  Skewness     : {_fmt(d.skewness)}")
    print(f"  Kurtosis     : {_fmt(d.kurtosis)} (excess)")
    print(f"  CV           : {_fmt(d.cv)}")

    n = report.normality
    print()
    print("=" * 60)
    print("NORMALITY TESTS")
    print("=" * 60)
    print(f"  Shapiro-Wilk    stat={_fmt(n.shapiro_stat, 4)}  p={_fmt(n.shapiro_p, 4)}")
    print(f"  Jarque-Bera     stat={_fmt(n.jb_stat, 4)}  p={_fmt(n.jb_p, 4)}")
    print(f"  D'Agostino K²   stat={_fmt(n.dagostino_stat, 4)}  p={_fmt(n.dagostino_p, 4)}")
    alpha = 0.05
    if n.shapiro_p is not None:
        normal = n.shapiro_p > alpha and n.jb_p > alpha and n.dagostino_p > alpha
        print(f"  → Distribution appears {'NORMAL' if normal else 'NON-NORMAL'} at α=0.05")

    s = report.stationarity
    print()
    print("=" * 60)
    print("STATIONARITY TESTS")
    print("=" * 60)
    print(f"  ADF  stat={_fmt(s.adf_stat, 4)}  p={_fmt(s.adf_p, 4)}")
    if s.adf_is_stationary is not None:
        print(f"       → Series is {'STATIONARY' if s.adf_is_stationary else 'NON-STATIONARY'} (ADF, α=0.05)")
    print(f"  KPSS stat={_fmt(s.kpss_stat, 4)}  p={_fmt(s.kpss_p, 4)}")
    if s.kpss_is_stationary is not None:
        print(f"       → Series is {'STATIONARY' if s.kpss_is_stationary else 'NON-STATIONARY'} (KPSS, α=0.05)")

    ac = report.autocorrelation
    print()
    print("=" * 60)
    print("AUTOCORRELATION (ACF / PACF)")
    print("=" * 60)
    if ac.acf_values is not None:
        print(f"  ACF  lags 1-5 : {ac.acf_values[1:6]}")
        print(f"  PACF lags 1-5 : {ac.pacf_values[1:6]}")
    else:
        print("  N/A (statsmodels not installed)")

    tr = report.trend
    print()
    print("=" * 60)
    print("TREND ANALYSIS (OLS)")
    print("=" * 60)
    print(f"  Slope     : {_fmt(tr.slope)}")
    print(f"  Intercept : {_fmt(tr.intercept)}")
    print(f"  R²        : {_fmt(tr.r_squared, 4)}")
    print(f"  p-value   : {_fmt(tr.p_value, 4)}")
    if tr.p_value is not None:
        sig = tr.p_value < 0.05
        print(f"  → Trend is {'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'} at α=0.05")

    ol = report.outliers
    print()
    print("=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)
    print(f"  IQR method  : {len(ol.iqr_outlier_indices)} outlier(s)")
    if len(ol.iqr_outlier_indices) > 0:
        for idx, val in zip(ol.iqr_outlier_indices, ol.iqr_outlier_values):
            print(f"    index={idx}  value={val:.6f}")
    print(f"  Z-score (>{ol.zscore_threshold}) : {len(ol.zscore_outlier_indices)} outlier(s)")
    if len(ol.zscore_outlier_indices) > 0:
        for idx, val in zip(ol.zscore_outlier_indices, ol.zscore_outlier_values):
            print(f"    index={idx}  value={val:.6f}")

    vol = report.volatility
    print()
    print("=" * 60)
    print(f"VOLATILITY (rolling window={vol.rolling_window})")
    print("=" * 60)
    if len(vol.rolling_std) > 0:
        print(f"  Mean rolling std : {np.mean(vol.rolling_std):.6f}")
        print(f"  Max  rolling std : {np.max(vol.rolling_std):.6f}")
        print(f"  Min  rolling std : {np.min(vol.rolling_std):.6f}")
    print(f"  ARCH LM stat={_fmt(vol.arch_stat, 4)}  p={_fmt(vol.arch_p, 4)}")
    if vol.arch_effect_present is not None:
        print(f"  → ARCH effect: {'PRESENT' if vol.arch_effect_present else 'NOT PRESENT'} at α=0.05")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------

def plot_report(series: npt.NDArray[np.float64], report: AnalysisReport, output_path: str = "mean_fr_report.png") -> None:
    """Save a multi-panel diagnostic figure to *output_path*."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib not available – skipping plot")
        return

    t = np.arange(len(series))
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Time Series Analysis: mean_fr", fontsize=14, fontweight="bold")

    # 1. Raw series
    ax = axes[0, 0]
    ax.plot(t, series, linewidth=0.8)
    trend_line = report.trend.slope * t + report.trend.intercept
    ax.plot(t, trend_line, color="red", linestyle="--", linewidth=1.2, label="Trend")
    ax.set_title("Time Series with Linear Trend")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Value")
    ax.legend()

    # 2. Histogram + KDE
    ax = axes[0, 1]
    ax.hist(series, bins=30, density=True, alpha=0.6, color="steelblue", label="Histogram")
    if _SCIPY_AVAILABLE:
        kde = scipy_stats.gaussian_kde(series)
        xs = np.linspace(series.min(), series.max(), 200)
        ax.plot(xs, kde(xs), color="darkblue", linewidth=1.5, label="KDE")
        ax.plot(xs, scipy_stats.norm.pdf(xs, np.mean(series), np.std(series, ddof=1)),
                color="red", linestyle="--", linewidth=1.2, label="Normal")
    ax.set_title("Distribution")
    ax.set_xlabel("Value")
    ax.legend()

    # 3. Rolling volatility
    ax = axes[1, 0]
    vol = report.volatility
    if len(vol.rolling_std) > 0:
        ax.plot(t, vol.rolling_std, linewidth=0.8, color="darkorange")
    ax.set_title(f"Rolling Std Dev (window={vol.rolling_window})")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Std Dev")

    # 4. Q-Q plot
    ax = axes[1, 1]
    if _SCIPY_AVAILABLE:
        (osm, osr), (slope, intercept, _) = scipy_stats.probplot(series)
        ax.scatter(osm, osr, s=10, color="steelblue", alpha=0.6)
        ax.plot(osm, slope * np.array(osm) + intercept, color="red", linewidth=1.2)
    ax.set_title("Q-Q Plot (Normal)")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")

    # 5. ACF
    ax = axes[2, 0]
    ac = report.autocorrelation
    if ac.acf_values is not None:
        lags = np.arange(len(ac.acf_values))
        ax.bar(lags, ac.acf_values, width=0.4, color="steelblue")
        n = report.descriptive.n
        ci = 1.96 / np.sqrt(n)
        ax.axhline(ci, linestyle="--", color="red", linewidth=0.8)
        ax.axhline(-ci, linestyle="--", color="red", linewidth=0.8)
    ax.set_title("Autocorrelation Function (ACF)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")

    # 6. PACF
    ax = axes[2, 1]
    if ac.pacf_values is not None:
        lags = np.arange(len(ac.pacf_values))
        ax.bar(lags, ac.pacf_values, width=0.4, color="darkorange")
        ax.axhline(ci, linestyle="--", color="red", linewidth=0.8)
        ax.axhline(-ci, linestyle="--", color="red", linewidth=0.8)
    ax.set_title("Partial Autocorrelation Function (PACF)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("PACF")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to '{output_path}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed statistical analysis of mean_fr.txt")
    parser.add_argument("--path", default="mean_fr.txt", help="Path to the time series file")
    parser.add_argument("--plot", action="store_true", help="Save a diagnostic plot")
    parser.add_argument("--plot-output", default="mean_fr_report.png", help="Output path for the plot")
    parser.add_argument("--nlags", type=int, default=20, help="Number of ACF/PACF lags")
    parser.add_argument("--zscore", type=float, default=3.0, help="Z-score threshold for outlier detection")
    parser.add_argument("--window", type=int, default=20, help="Rolling window size for volatility")
    args = parser.parse_args()

    series = load_series(args.path)
    report = analyse(series, nlags=args.nlags, zscore_threshold=args.zscore, volatility_window=args.window)
    print_report(report)

    if args.plot:
        plot_report(series, report, output_path=args.plot_output)


if __name__ == "__main__":
    main()
