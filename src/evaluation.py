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


# ============================================================================ model statistik (inferensi)
def reference_coding(Z: pd.DataFrame, prefixes: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Ubah one-hot penuh menjadi dummy coding: buang 1 kolom acuan per grup (kategori TERBANYAK).

    Tanpa ini, jumlah dummy satu grup = 1 = intercept → kolinear sempurna (dummy variable trap).
    Return: matriks tanpa kolom acuan, {grup: [kolom]}, {grup: kolom acuan}.
    """
    groups, refs = {}, {}
    for name, prefix in prefixes.items():
        cols = [c for c in Z.columns if c.startswith(prefix)]
        if len(cols) > 1:
            refs[name] = Z[cols].sum().idxmax()
            cols = [c for c in cols if c != refs[name]]
        groups[name] = cols
    return Z.drop(columns=list(refs.values())), groups, refs


def _rank_with_const(D: pd.DataFrame) -> int:
    return int(np.linalg.matrix_rank(np.column_stack([np.ones(len(D)), D.to_numpy(float)])))


def resolve_perfect_collinearity(D: pd.DataFrame, groups: dict) -> tuple[pd.DataFrame, dict, list]:
    """Selama matriks (dengan intercept) tidak full-rank, buang GRUP fitur yang paling sedikit kolomnya
    yang penghapusannya paling mengurangi kekurangan rank. Return: matriks, grup tersisa, grup yang dibuang."""
    D, groups, dropped = D.copy(), dict(groups), []
    while _rank_with_const(D) < D.shape[1] + 1:
        cands = []
        for g, cols in groups.items():
            Dg = D.drop(columns=cols)
            cands.append(((Dg.shape[1] + 1) - _rank_with_const(Dg), len(cols), g))
        _, _, g = min(cands)
        D = D.drop(columns=groups.pop(g))
        dropped.append(g)
    return D, groups, dropped


def gvif(D: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Generalized VIF (Fox & Monette, 1992) per GRUP fitur (mis. semua dummy satu kategorikal).

    GVIF = det(R_grup) · det(R_lain) / det(R). Agar bisa dibandingkan antar grup dengan jumlah kolom berbeda,
    dipakai GVIF^(1/(2·df)), yang setara dengan √VIF untuk fitur 1 kolom (ambang √10 ≈ 3,16 ≈ VIF 10).
    """
    R = np.corrcoef(D.to_numpy(float), rowvar=False)
    cols = list(D.columns)
    rows = {}
    # errstate: BLAS di sebagian platform (mis. macOS Accelerate) menyalakan flag floating-point palsu saat
    # dekomposisi LU walaupun hasilnya benar (sudah dicek dengan metode eigenvalue di notebook/test)
    with np.errstate(all="ignore"):
        ld = np.linalg.slogdet(R)[1]
        for g, cs in groups.items():
            idx = [cols.index(c) for c in cs]
            rest = [i for i in range(len(cols)) if i not in idx]
            val = float(np.exp(np.linalg.slogdet(R[np.ix_(idx, idx)])[1] + np.linalg.slogdet(R[np.ix_(rest, rest)])[1] - ld))
            rows[g] = {"df": len(cs), "GVIF": val, "GVIF_adj": val ** (1 / (2 * len(cs)))}
    return pd.DataFrame(rows).T.sort_values("GVIF_adj", ascending=False)
