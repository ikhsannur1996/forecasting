"""Metrik & uji statistik untuk forecasting.

Metrik utama untuk kebutuhan kas ATM:
- WAPE  = Σ|ramalan − aktual| / Σ aktual  (error tertimbang nominal; tidak meledak saat aktual kecil)
- Bias  = Σ(ramalan − aktual) / Σ aktual  (negatif = under-forecast → risiko ATM kehabisan uang)
- MASE  = MAE / MAE seasonal-naive in-sample (< 1 = lebih baik dari “sama dengan minggu lalu”)
"""

import numpy as np
import pandas as pd
from scipy import stats


def forecast_metrics(df: pd.DataFrame) -> dict:
    """df berisi kolom y, yhat (dan opsional naive_mae untuk MASE)."""
    e = df["yhat"] - df["y"]
    out = {
        "wape": float(e.abs().sum() / df["y"].sum()),
        "bias": float(e.sum() / df["y"].sum()),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "smape": float(np.mean(2 * e.abs() / (df["y"].abs() + df["yhat"].abs()))),
    }
    if "naive_mae" in df:
        out["mase"] = float((e.abs() / df["naive_mae"]).mean())
    return out


def metrics_by(df: pd.DataFrame, by) -> pd.DataFrame:
    return df.groupby(by, observed=True).apply(lambda g: pd.Series(forecast_metrics(g)), include_groups=False)


def cluster_bootstrap_ci(df: pd.DataFrame, cluster_cols=("origin", "atm_id"), n_boot: int = 1000,
                         alpha: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """CI metrik dengan bootstrap per CLUSTER (1 cluster = 1 lintasan forecast origin×ATM).
    Error di dalam satu lintasan saling berkorelasi, jadi yang di-resample adalah lintasannya, bukan baris."""
    rng = np.random.default_rng(seed)
    keys = df.groupby(list(cluster_cols)).ngroup().to_numpy()
    n = keys.max() + 1
    idx_by_cluster = [np.where(keys == k)[0] for k in range(n)]
    point = forecast_metrics(df)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        rows = np.concatenate([idx_by_cluster[k] for k in pick])
        boot.append(forecast_metrics(df.iloc[rows]))
    boot = pd.DataFrame(boot)
    return pd.DataFrame({"estimate": pd.Series(point), "ci_lower": boot.quantile(alpha / 2),
                         "ci_upper": boot.quantile(1 - alpha / 2), "std_error": boot.std()})


def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict:
    """Uji Diebold-Mariano (1995) + koreksi sampel kecil Harvey-Leybourne-Newbold (1997).

    d_t = loss_a − loss_b (deret waktu). Variance dihitung dengan Newey-West (lag h−1) karena
    error forecast multi-step saling berkorelasi. H0: akurasi kedua model sama. d̄ < 0 → A lebih baik.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[: n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))) / n
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    return {"mean_diff": float(dbar), "DM": float(dm), "DM_HLN": float(stat), "p_value": float(2 * stats.t.sf(abs(stat), n - 1)), "n": n}


def interval_quantiles(log_resid: pd.Series, horizon: pd.Series, qs=(0.05, 0.10, 0.50, 0.90, 0.95)) -> pd.DataFrame:
    """Kuantil empiris residual log (aktual − ramalan) per horizon → interval multiplikatif."""
    return pd.DataFrame({h: np.quantile(r, qs) for h, r in log_resid.groupby(horizon)}, index=[f"q{int(q * 100):02d}" for q in qs]).T


def coverage_test(inside: np.ndarray, nominal: float) -> dict:
    inside = np.asarray(inside, bool)
    t = stats.binomtest(int(inside.sum()), len(inside), nominal)
    ci = t.proportion_ci(method="wilson")
    return {"coverage": float(inside.mean()), "ci_lower": ci.low, "ci_upper": ci.high, "n": len(inside), "p_value": float(t.pvalue)}


def ljung_box(residuals, lags=(7, 14)) -> pd.DataFrame:
    """Ljung-Box: H0 = residual tidak berautokorelasi (informasi waktu sudah terserap model)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    return acorr_ljungbox(np.asarray(residuals), lags=list(lags), return_df=True)


def holm_correction(p: pd.Series) -> pd.Series:
    s = p.sort_values()
    m = len(s)
    adj = np.maximum.accumulate([min(1, (m - i) * v) for i, v in enumerate(s.values)])
    return pd.Series(adj, index=s.index).reindex(p.index)


def segment_bias(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Bias per segmen + t-test (H0: rata-rata error relatif = 0) pada rata-rata per lintasan (origin×ATM)
    agar observasi yang diuji kurang lebih independen."""
    rows = []
    for g, part in df.groupby(by, observed=True):
        path = part.groupby(["origin", "atm_id"]).apply(lambda x: (x["yhat"] - x["y"]).sum() / x["y"].sum(), include_groups=False)
        t = stats.ttest_1samp(path, 0) if len(path) > 2 else None
        rows.append({by: g, "n_hari": len(part), "wape": (part["yhat"] - part["y"]).abs().sum() / part["y"].sum(),
                     "bias": (part["yhat"] - part["y"]).sum() / part["y"].sum(), "p_value": t.pvalue if t else np.nan})
    out = pd.DataFrame(rows).set_index(by)
    out["p_holm"] = holm_correction(out["p_value"])
    return out
