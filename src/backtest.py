"""Rolling-origin backtest (time-series cross-validation).

Untuk setiap origin: model dilatih HANYA dengan data ≤ origin, lalu meramal origin+1 … origin+horizon.
Hasilnya dibandingkan dengan aktual. Origin diproses paralel.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone


def _run_origin(factories: dict, wide_log: pd.DataFrame, origin, horizon: int) -> pd.DataFrame:
    out = []
    for name, factory in factories.items():
        model = factory().fit(wide_log, end=origin)
        fc = model.forecast(wide_log, origin, horizon).assign(model=name, origin=pd.Timestamp(origin))
        out.append(fc)
    return pd.concat(out, ignore_index=True)


def rolling_origin(factories: dict, wide_log: pd.DataFrame, origins, horizon: int, n_jobs: int = -1) -> pd.DataFrame:
    """factories: {nama: fungsi tanpa argumen yang membuat model baru}."""
    parts = Parallel(n_jobs=n_jobs)(delayed(_run_origin)(factories, wide_log, o, horizon) for o in origins)
    fc = pd.concat(parts, ignore_index=True)
    actual = wide_log.stack(future_stack=True).rename("y_log").reset_index()
    actual.columns = ["date", "atm_id", "y_log"]
    fc = fc.merge(actual, on=["date", "atm_id"], how="left")
    fc["y"] = np.expm1(fc["y_log"])
    fc["yhat"] = np.expm1(fc["yhat_log"])
    fc["error"] = fc["yhat"] - fc["y"]
    fc["log_resid"] = fc["y_log"] - fc["yhat_log"]
    return fc


def naive_scale(wide_log: pd.DataFrame, origins, window: int = 365) -> pd.DataFrame:
    """Skala MASE: MAE seasonal-naive (lag 7) in-sample per ATM pada `window` hari sebelum origin."""
    y = np.expm1(wide_log)
    rows = []
    for o in origins:
        seg = y.loc[:o].iloc[-window:]
        mae = (seg - seg.shift(7)).abs().mean()
        rows.append(pd.DataFrame({"origin": pd.Timestamp(o), "atm_id": mae.index, "naive_mae": mae.to_numpy()}))
    return pd.concat(rows, ignore_index=True)
