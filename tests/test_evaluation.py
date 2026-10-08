import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from src.calendar_id import calendar_frame, days_to_lebaran
from src.evaluation import cluster_bootstrap_ci, coverage_test, diebold_mariano, forecast_metrics, interval_quantiles
from src.features import FEATURE_COLUMNS, build_features, to_wide
from src.json_model import JsonForecaster, export_model
from src.models import ETSForecaster, GBRForecaster, SeasonalNaive

rng = np.random.default_rng(0)


@pytest.fixture(scope="module")
def data():
    from scripts.generate_data import generate
    master, tx = generate(seed=3)
    tx = tx[tx["date"] <= "2023-06-30"]
    return master, to_wide(tx), dict(zip(master["atm_id"], master["location_type"]))


def test_calendar_bins():
    cal = calendar_frame(pd.to_datetime(["2025-03-25", "2025-03-31", "2025-04-03", "2025-06-15", "2025-06-27"])).set_index("date")
    assert list(cal["lebaran_bin"]) == ["H-7_s.d._H-1", "H0_s.d._H+3", "H0_s.d._H+3", "normal", "normal"]
    assert cal.loc["2025-06-27", "dom_bin"] == "gajian_25-31" and cal.loc["2025-03-31", "is_holiday"] == 1
    assert days_to_lebaran(["2025-03-21"])[0] == -10


def test_metrics_and_bias_sign():
    df = pd.DataFrame({"y": [100.0, 200.0], "yhat": [90.0, 180.0], "naive_mae": [20.0, 20.0]})
    m = forecast_metrics(df)
    assert m["wape"] == pytest.approx(30 / 300) and m["bias"] == pytest.approx(-0.1) and m["mase"] == pytest.approx(0.75)


def test_diebold_mariano_detects_difference_and_handles_equal():
    base = rng.gamma(2, 1, 400)
    assert diebold_mariano(base * 0.7, base, h=14)["p_value"] < 1e-4
    noisy = base + rng.normal(0, 0.01, 400)
    assert diebold_mariano(noisy, base + rng.normal(0, 0.01, 400), h=14)["p_value"] > 0.01


def test_interval_quantiles_and_coverage():
    r = pd.Series(rng.normal(0, 0.1, 5000))
    h = pd.Series(np.tile(np.arange(1, 11), 500))
    q = interval_quantiles(r, h)
    inside = (r.to_numpy() >= q.loc[h, "q10"].to_numpy()) & (r.to_numpy() <= q.loc[h, "q90"].to_numpy())
    assert coverage_test(inside, 0.8)["coverage"] == pytest.approx(0.8, abs=0.02)


def test_cluster_bootstrap_contains_estimate():
    df = pd.DataFrame({"origin": np.repeat(range(20), 10), "atm_id": "A", "y": rng.uniform(50, 150, 200)})
    df["yhat"] = df["y"] * rng.normal(1, 0.1, 200)
    ci = cluster_bootstrap_ci(df, n_boot=100)
    assert ((ci["ci_lower"] <= ci["estimate"]) & (ci["estimate"] <= ci["ci_upper"])).all()


def test_no_leakage(data):
    _, W, types = data
    day = W.index[60]
    W2 = W.copy()
    W2.loc[day] += 3
    assert np.allclose(build_features(W, [day], types)[FEATURE_COLUMNS].to_numpy(), build_features(W2, [day], types)[FEATURE_COLUMNS].to_numpy())


def test_seasonal_naive_copies_last_week(data):
    _, W, _ = data
    origin = W.index[100]
    fc = SeasonalNaive().forecast(W, origin, 9)
    row = fc[(fc["h"] == 9) & (fc["atm_id"] == W.columns[0])]
    # h = 9 → hari yang sama dua minggu sebelumnya yang sudah diketahui: origin + 9 − 14 = origin − 5
    assert row["yhat_log"].iloc[0] == pytest.approx(W.loc[origin - pd.Timedelta(days=5), W.columns[0]])


def test_json_matches_gbr_and_ets(data):
    master, W, types = data
    meta = {"atms": master.to_dict("records")}
    iv = pd.DataFrame({"q10": [-0.1] * 14, "q90": [0.1] * 14}, index=range(1, 15))
    g = GBRForecaster(types, GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=0)).fit(W)
    jg = JsonForecaster(export_model(g, W, meta, iv))
    a, b = g.forecast(W, W.index.max(), 10), jg.forecast(10)
    assert np.abs(a["yhat_log"].to_numpy() - b["yhat_log"].to_numpy()).max() < 1e-9
    e = ETSForecaster(train_days=365, n_jobs=1).fit(W)
    je = JsonForecaster(export_model(e, W, meta, iv))
    x = e.forecast(W, W.index.max(), 10).sort_values(["atm_id", "h"])
    y = je.forecast(10).sort_values(["atm_id", "h"])
    assert np.abs(x["yhat_log"].to_numpy() - y["yhat_log"].to_numpy()).max() < 1e-9
