"""FastAPI service: forecasting kebutuhan uang tunai ATM + rekomendasi pengisian kas."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import AtmForecast, DayForecast, ForecastRequest, ForecastResponse, HealthResponse
from src.calendar_id import calendar_frame
from src.json_model import JsonForecaster

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "model.json"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("atm-forecast-api")
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model = JsonForecaster.load(MODEL_PATH)
    state["model"], state["metadata"] = model, model.metadata
    state["atms"] = {a["atm_id"]: a for a in model.metadata["atms"]}
    logger.info("Model forecasting '%s' versi %s dimuat (data s.d. %s, %d ATM)", model.metadata["model_name"],
                model.metadata["model_version"], model.origin.date(), len(state["atms"]))
    yield
    state.clear()


app = FastAPI(
    title="ATM Cash Demand Forecasting API",
    description="Ramalan penarikan tunai harian per ATM (1–28 hari) + interval prediksi + rekomendasi isi kas.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return {"message": "ATM Cash Demand Forecasting API. Buka /docs untuk dokumentasi."}


@app.get("/health", response_model=HealthResponse)
def health():
    meta = state.get("metadata")
    return HealthResponse(status="ok", model_loaded="model" in state, model_version=meta["model_version"] if meta else None)


@app.get("/model-info")
def model_info():
    m = state["model"]
    keys = ("model_name", "model_label", "model_version", "trained_at", "training_period", "validated_horizon_days",
            "max_horizon_days", "calendar_valid_until", "params", "load_quantile", "evaluation")
    return {**{k: m.metadata[k] for k in keys}, "data_until": m.origin.date(), "intervals": m.intervals}


@app.get("/atms")
def atms():
    return list(state["atms"].values())


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    model, meta = state["model"], state["metadata"]
    atm_ids = req.atm_ids or list(state["atms"])
    unknown = sorted(set(atm_ids) - set(state["atms"]))
    if unknown:
        raise HTTPException(status_code=422, detail=f"ATM tidak dikenal: {unknown}")

    recent = None
    if req.recent_actuals:
        recent = {a: {d.isoformat(): v for d, v in s.items()} for a, s in req.recent_actuals.items()}
    try:
        fc = model.forecast(req.horizon, recent_actuals=recent)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inferensi gagal")
        raise HTTPException(status_code=500, detail="Inferensi model gagal") from exc

    data_until = fc["date"].min() - pd.Timedelta(days=1)
    fc = fc[fc["atm_id"].isin(atm_ids)].copy()
    load_q = {int(k): v for k, v in meta["load_quantile_by_h"].items()}
    max_q = max(load_q)
    fc["load_raw"] = np.expm1(fc["yhat_log"] + fc["h"].map(lambda h: load_q[min(h, max_q)]))
    fc["capacity"] = fc["atm_id"].map(lambda a: state["atms"][a]["cassette_capacity"])
    fc["recommended_load"] = np.minimum(fc["load_raw"], fc["capacity"])
    cal = calendar_frame(sorted(fc["date"].unique())).set_index("date")

    warnings = []
    if req.horizon > meta["validated_horizon_days"]:
        warnings.append(f"Horizon > {meta['validated_horizon_days']} hari belum divalidasi backtest; interval hari ke-"
                        f"{meta['validated_horizon_days'] + 1}+ memakai lebar interval hari ke-{meta['validated_horizon_days']}.")
    if fc["date"].max() > pd.Timestamp(meta["calendar_valid_until"]):
        warnings.append(f"Ramalan melewati {meta['calendar_valid_until']}: perbarui kalender libur di src/calendar_id.py.")
    gap = (pd.Timestamp.today().normalize() - data_until).days
    if gap > 7 and not req.recent_actuals:
        warnings.append(f"Data terakhir model {data_until.date()} ({gap} hari lalu). Kirim recent_actuals atau refit model.")
    n_cap = int((fc["load_raw"] > fc["capacity"]).sum())
    if n_cap:
        warnings.append(f"{n_cap} hari-ATM: rekomendasi isi melebihi kapasitas kaset → dibatasi kapasitas; pertimbangkan pengisian lebih sering.")

    results = []
    for atm, g in fc.groupby("atm_id", sort=False):
        days = None
        if req.include_daily:
            days = [DayForecast(date=r.date.date(), h=int(r.h), forecast=round(r.forecast, 2), p10=round(r.q10, 2), p90=round(r.q90, 2),
                                p05=round(r.q05, 2), p95=round(r.q95, 2), recommended_load=round(r.recommended_load, 2),
                                capped_by_capacity=bool(r.load_raw > r.capacity), is_holiday=bool(cal.loc[r.date, "is_holiday"]),
                                payday_window=cal.loc[r.date, "dom_bin"], lebaran_phase=cal.loc[r.date, "lebaran_bin"])
                    for r in g.itertuples()]
        info = state["atms"][atm]
        results.append(AtmForecast(atm_id=atm, location_type=info["location_type"], cassette_capacity=info["cassette_capacity"],
                                   total_forecast=round(g["forecast"].sum(), 2), total_p10=round(g["q10"].sum(), 2),
                                   total_p90=round(g["q90"].sum(), 2), total_recommended_load=round(g["recommended_load"].sum(), 2), days=days))
    order = {a: i for i, a in enumerate(atm_ids)}
    results.sort(key=lambda r: order[r.atm_id])
    return ForecastResponse(model_version=meta["model_version"], model_name=meta["model_label"], data_until=data_until.date(),
                            horizon=req.horizon, load_quantile=meta["load_quantile"], warnings=warnings, atms=results)
