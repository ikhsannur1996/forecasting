"""Tiga metode forecasting dengan antarmuka sama:

    model.fit(wide_log, end=origin)            # latih memakai data s.d. origin
    model.forecast(wide_log, origin, horizon)  # DataFrame: atm_id, date, h, yhat_log

- SeasonalNaive    : nilai hari yang sama minggu lalu (patokan dasar / benchmark MASE)
- ETSForecaster    : Holt-Winters (tren aditif teredam + musiman mingguan) per ATM, skala log
- GBRForecaster    : satu Gradient Boosting global untuk semua ATM, fitur lag + kalender, rekursif
"""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor

from src.features import FEATURE_COLUMNS, recursive_forecast, training_frame


class SeasonalNaive:
    name = "seasonal_naive"

    def fit(self, wide_log, end=None, **_):
        return self

    def forecast(self, wide_log, origin, horizon):
        origin = pd.Timestamp(origin)
        rows = []
        for h in range(1, horizon + 1):
            src = origin + pd.Timedelta(days=h - 7 * int(np.ceil(h / 7)))
            day = origin + pd.Timedelta(days=h)
            rows.append(pd.DataFrame({"atm_id": wide_log.columns, "date": day, "h": h, "yhat_log": wide_log.loc[src].to_numpy()}))
        return pd.concat(rows, ignore_index=True)


def _fit_ets(series: pd.Series, damped: bool):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(series.to_numpy(), trend="add", damped_trend=damped, seasonal="add",
                                     seasonal_periods=7, initialization_method="estimated")
        return model.fit(optimized=True)


class ETSForecaster:
    """Holt-Winters per ATM. Jika damped='auto', pilih teredam/tidak per ATM dengan AIC."""

    name = "ets"

    def __init__(self, damped="auto", train_days: int | None = 730, n_jobs: int = -1):
        self.damped, self.train_days, self.n_jobs = damped, train_days, n_jobs

    def _fit_one(self, s):
        options = [True, False] if self.damped == "auto" else [bool(self.damped)]
        fits = [_fit_ets(s, d) for d in options]
        return min(fits, key=lambda f: f.aic)

    def fit(self, wide_log, end=None, **_):
        data = wide_log.loc[:end] if end is not None else wide_log
        if self.train_days:
            data = data.iloc[-self.train_days:]
        self.atms_ = list(data.columns)
        self.fits_ = dict(zip(self.atms_, Parallel(n_jobs=self.n_jobs)(delayed(self._fit_one)(data[a].dropna()) for a in self.atms_)))
        self.origin_ = data.index.max()
        return self

    def forecast(self, wide_log, origin, horizon):
        assert pd.Timestamp(origin) == self.origin_, "ETS hanya bisa forecast dari akhir data latihnya"
        rows = []
        for a, f in self.fits_.items():
            fc = f.forecast(horizon)
            rows.append(pd.DataFrame({"atm_id": a, "date": pd.date_range(self.origin_ + pd.Timedelta(days=1), periods=horizon),
                                      "h": np.arange(1, horizon + 1), "yhat_log": fc}))
        return pd.concat(rows, ignore_index=True)

    def state(self, atm_id) -> dict:
        """Parameter & state akhir untuk ekspor JSON."""
        f = self.fits_[atm_id]
        p = f.params
        phi = float(p["damping_trend"]) if f.model.damped_trend else 1.0
        level, trend = float(f.level[-1]), float(f.trend[-1])
        # komponen musiman diturunkan dari forecast statsmodels sendiri (bukan dari array `season`, yang
        # untuk model tanpa damping tidak selalu sama dengan nilai yang dipakai saat forecast)
        h = np.arange(1, 8)
        season = f.forecast(7) - level - np.cumsum(phi**h) * trend
        return {"alpha": float(p["smoothing_level"]), "beta": float(p["smoothing_trend"]), "gamma": float(p["smoothing_seasonal"]),
                "phi": phi, "level": level, "trend": trend, "season": [float(x) for x in season]}


class GBRForecaster:
    name = "gbr"

    def __init__(self, atm_types: dict, estimator: GradientBoostingRegressor | None = None, train_start=None):
        self.atm_types = atm_types
        self.estimator = GradientBoostingRegressor(random_state=42) if estimator is None else estimator
        self.train_start = train_start

    def fit(self, wide_log, end=None, **_):
        X, y = training_frame(wide_log.loc[:end] if end is not None else wide_log, self.atm_types, start=self.train_start)
        self.model_ = clone(self.estimator).fit(X[FEATURE_COLUMNS].to_numpy(dtype=float), y.to_numpy())
        self.n_train_ = len(y)
        return self

    def predict_features(self, F: np.ndarray) -> np.ndarray:
        return self.model_.predict(F)

    def forecast(self, wide_log, origin, horizon):
        return recursive_forecast(self.predict_features, wide_log, origin, horizon, self.atm_types)
