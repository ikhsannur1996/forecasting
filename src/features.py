"""Feature engineering untuk forecasting penarikan ATM (pure pandas/numpy, tanpa scikit-learn).

Fungsi yang SAMA dipakai untuk training, forecast rekursif di notebook, dan inferensi di API,
sehingga tidak ada perbedaan perlakuan data antara pengembangan dan produksi.

Data disimpan dalam format "wide": index = tanggal (harian, tanpa lubang), kolom = atm_id,
nilai = log1p(penarikan). Semua fitur lag/rolling untuk tanggal t hanya memakai data ≤ t−1.
"""

import numpy as np
import pandas as pd

from src.calendar_id import DOM_BINS, LEBARAN_BINS, calendar_frame

LAGS = [1, 2, 3, 7, 14, 21, 28]
MAX_LOOKBACK = 28
LOCATION_TYPES = ["mall", "office", "residential", "transport_hub", "university", "market"]
FLAG_COLS = ["is_holiday", "is_pre_holiday", "is_post_holiday", "is_december_peak", "is_semester_break"]
CATEGORIES = {
    "dow": list(range(7)),
    "month": list(range(1, 13)),
    "dom_bin": DOM_BINS["labels"],
    "lebaran_bin": [b for b in LEBARAN_BINS["labels"] if b != "normal_setelah"],
    "location_type": LOCATION_TYPES,
}
LAG_COLS = [f"lag_{k}" for k in LAGS] + ["roll_mean_7", "roll_mean_28", "roll_std_7", "same_dow_mean_4"]
FEATURE_COLUMNS = LAG_COLS + FLAG_COLS + [f"{c}_{v}" for c, vals in CATEGORIES.items() for v in vals]


def to_wide(df: pd.DataFrame, value_col: str = "withdrawal") -> pd.DataFrame:
    """Long (date, atm_id, nilai) → wide log1p, dengan index tanggal harian lengkap."""
    wide = df.pivot(index="date", columns="atm_id", values=value_col).sort_index()
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    return np.log1p(wide)


def build_features(wide_log: pd.DataFrame, dates, atm_types: dict) -> pd.DataFrame:
    """Matriks fitur untuk setiap (tanggal ∈ dates) × (ATM di wide_log).

    wide_log harus mencakup setidaknya 28 hari sebelum tanggal pertama di `dates`
    (nilai untuk tanggal target sendiri boleh NaN, karena tidak dipakai).
    """
    dates = pd.DatetimeIndex(dates)
    w = wide_log.reindex(pd.date_range(min(wide_log.index.min(), dates.min()), dates.max(), freq="D"))
    prev = w.shift(1)
    parts = {f"lag_{k}": w.shift(k) for k in LAGS}
    parts["roll_mean_7"] = prev.rolling(7, min_periods=7).mean()
    parts["roll_mean_28"] = prev.rolling(28, min_periods=28).mean()
    parts["roll_std_7"] = prev.rolling(7, min_periods=7).std()
    parts["same_dow_mean_4"] = (w.shift(7) + w.shift(14) + w.shift(21) + w.shift(28)) / 4

    atms = list(w.columns)
    long = pd.concat({name: frame.loc[dates].stack(future_stack=True) for name, frame in parts.items()}, axis=1)
    long.index.names = ["date", "atm_id"]
    long = long.reset_index()

    cal = calendar_frame(dates).set_index("date")
    long = long.join(cal, on="date")
    long["location_type"] = long["atm_id"].map(atm_types)
    for col, vals in CATEGORIES.items():
        for v in vals:
            long[f"{col}_{v}"] = (long[col] == v).astype(float)
    long[FLAG_COLS] = long[FLAG_COLS].astype(float)
    assert set(long["atm_id"]) <= set(atms)
    return long[["date", "atm_id"] + FEATURE_COLUMNS]


def training_frame(wide_log: pd.DataFrame, atm_types: dict, start=None, end=None) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) untuk training: semua tanggal yang punya 28 hari riwayat & target terisi."""
    dates = wide_log.index[MAX_LOOKBACK:]
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    feats = build_features(wide_log, dates, atm_types)
    target = wide_log.loc[dates].stack(future_stack=True).rename("y").reset_index()
    target.columns = ["date", "atm_id", "y"]
    frame = feats.merge(target, on=["date", "atm_id"]).dropna()
    return frame, frame.pop("y")


def recursive_forecast(predict_fn, wide_log: pd.DataFrame, origin, horizon: int, atm_types: dict) -> pd.DataFrame:
    """Forecast rekursif h = 1..horizon dari tanggal `origin` (data terakhir yang diketahui).

    Pada langkah h, lag yang jatuh setelah origin diisi PREDIKSI langkah sebelumnya.
    predict_fn: matriks fitur (n × len(FEATURE_COLUMNS)) → prediksi log1p.
    """
    origin = pd.Timestamp(origin)
    hist = wide_log.loc[origin - pd.Timedelta(days=MAX_LOOKBACK + 7): origin].copy()
    out = []
    for h in range(1, horizon + 1):
        day = origin + pd.Timedelta(days=h)
        hist.loc[day] = np.nan
        feats = build_features(hist, [day], atm_types)
        pred = predict_fn(feats[FEATURE_COLUMNS].to_numpy(dtype=float))
        hist.loc[day, feats["atm_id"].to_numpy()] = pred
        out.append(pd.DataFrame({"atm_id": feats["atm_id"].to_numpy(), "date": day, "h": h, "yhat_log": pred}))
    return pd.concat(out, ignore_index=True)
