"""Serialisasi model forecasting ke JSON + inferensi tanpa scikit-learn/statsmodels.

JSON berisi: model (pohon Gradient Boosting ATAU state Holt-Winters per ATM), riwayat 35 hari
terakhir per ATM (untuk lag), profil ATM, kuantil interval per horizon, dan rekomendasi kuantil
pengisian kas. Fitur dibangun oleh `src/features.py` yang sama dengan saat training.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, MAX_LOOKBACK, recursive_forecast

FORMAT_NAME = "atm-cash-forecast-json-model"
FORMAT_VERSION = 1
HISTORY_DAYS = MAX_LOOKBACK + 7


def _l(a):
    return np.asarray(a).tolist()


def export_model(forecaster, wide_log: pd.DataFrame, metadata: dict, intervals: pd.DataFrame) -> dict:
    origin = wide_log.index.max()
    hist = wide_log.iloc[-HISTORY_DAYS:]
    if forecaster.name == "gbr":
        m = forecaster.model_
        trees = [{"left": _l(t.tree_.children_left), "right": _l(t.tree_.children_right), "feature": _l(t.tree_.feature),
                  "threshold": _l(t.tree_.threshold), "value": _l(t.tree_.value[:, 0, 0])} for t in m.estimators_[:, 0]]
        model = {"type": "gbr", "init": float(m.init_.constant_.ravel()[0]), "learning_rate": float(m.learning_rate),
                 "feature_columns": FEATURE_COLUMNS, "trees": trees}
    elif forecaster.name == "ets":
        model = {"type": "ets", "states": {a: forecaster.state(a) for a in forecaster.atms_}}
    else:
        raise ValueError(forecaster.name)
    return {
        "format": FORMAT_NAME, "format_version": FORMAT_VERSION, "metadata": metadata, "model": model,
        "origin": origin.strftime("%Y-%m-%d"),
        "history": {"dates": [d.strftime("%Y-%m-%d") for d in hist.index], "log1p_values": {a: _l(hist[a]) for a in hist.columns}},
        "intervals": {str(h): row.to_dict() for h, row in intervals.iterrows()},
    }


def save_json(spec: dict, path) -> None:
    Path(path).write_text(json.dumps(spec, allow_nan=False, ensure_ascii=False, separators=(",", ":")))


class JsonForecaster:
    def __init__(self, spec: dict):
        if spec.get("format") != FORMAT_NAME or spec.get("format_version") != FORMAT_VERSION:
            raise ValueError("Format model JSON tidak dikenali")
        self.spec, self.metadata, self.model = spec, spec["metadata"], spec["model"]
        self.origin = pd.Timestamp(spec["origin"])
        self.atm_types = {a["atm_id"]: a["location_type"] for a in self.metadata["atms"]}
        h = spec["history"]
        self.history = pd.DataFrame(h["log1p_values"], index=pd.to_datetime(h["dates"]))
        self.intervals = {int(k): v for k, v in spec["intervals"].items()}
        if self.model["type"] == "gbr":
            self._trees = [{k: np.asarray(v) for k, v in t.items()} for t in self.model["trees"]]

    @classmethod
    def load(cls, path) -> "JsonForecaster":
        return cls(json.loads(Path(path).read_text()))

    # --------------------------------------------------------------- GBR
    def _tree(self, t, Z):
        node = np.zeros(len(Z), dtype=np.int64)
        rows = np.arange(len(Z))
        while True:
            active = t["left"][node] != -1
            if not active.any():
                return t["value"][node]
            r, n = rows[active], node[active]
            go_left = Z[r, t["feature"][n]] <= t["threshold"][n]
            node[r] = np.where(go_left, t["left"][n], t["right"][n])

    def predict_features(self, F: np.ndarray) -> np.ndarray:
        Z = np.asarray(F, float).astype(np.float32).astype(np.float64)  # pohon sklearn membandingkan dalam float32
        return self.model["init"] + self.model["learning_rate"] * np.sum([self._tree(t, Z) for t in self._trees], axis=0)

    # --------------------------------------------------------------- ETS
    def _ets_forecast(self, horizon: int) -> pd.DataFrame:
        rows = []
        for a, s in self.model["states"].items():
            h = np.arange(1, horizon + 1)
            damp = np.cumsum(s["phi"] ** h)
            season = np.asarray(s["season"])[(h - 1) % 7]
            rows.append(pd.DataFrame({"atm_id": a, "date": self.origin + pd.to_timedelta(h, unit="D"), "h": h,
                                      "yhat_log": s["level"] + damp * s["trend"] + season}))
        return pd.concat(rows, ignore_index=True)

    # --------------------------------------------------------------- publik
    def forecast(self, horizon: int, recent_actuals: dict | None = None) -> pd.DataFrame:
        """Ramalan h = 1..horizon. recent_actuals = {atm_id: {tanggal: nilai juta Rp}} untuk hari-hari
        SETELAH origin model (harus berurutan tanpa lubang); hanya didukung model GBR."""
        hist, origin = self.history.copy(), self.origin
        if recent_actuals:
            if self.model["type"] != "gbr":
                raise ValueError("Update data terbaru hanya didukung model GBR; lakukan refit untuk model ETS")
            new = pd.DataFrame(recent_actuals)
            new.index = pd.to_datetime(new.index)
            new = new.sort_index()
            expected = pd.date_range(origin + pd.Timedelta(days=1), new.index.max(), freq="D")
            if not new.index.equals(expected) or new.isna().any().any() or set(new.columns) != set(hist.columns):
                raise ValueError(f"recent_actuals harus berisi SEMUA ATM untuk setiap tanggal {expected[0].date()} s.d. "
                                 f"{expected[-1].date()} tanpa lubang")
            hist = pd.concat([hist, np.log1p(new[hist.columns])])
            origin = hist.index.max()
        if self.model["type"] == "gbr":
            fc = recursive_forecast(self.predict_features, hist, origin, horizon, self.atm_types)
        else:
            fc = self._ets_forecast(horizon)
        return self._with_intervals(fc)

    def _with_intervals(self, fc: pd.DataFrame) -> pd.DataFrame:
        max_h = max(self.intervals)
        q = pd.DataFrame([self.intervals[min(h, max_h)] for h in fc["h"]], index=fc.index)
        out = fc.copy()
        out["forecast"] = np.expm1(out["yhat_log"])
        for col in q.columns:
            out[col] = np.expm1(out["yhat_log"] + q[col])
        return out
