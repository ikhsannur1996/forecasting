"""Skema request/response API forecasting kebutuhan uang tunai ATM."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {"atm_ids": ["ATM09", "ATM01"], "horizon": 14}})

    atm_ids: list[str] | None = Field(None, min_length=1, max_length=500, description="Default: semua ATM")
    horizon: int = Field(14, ge=1, le=28, description="Jumlah hari ke depan (tervalidasi s.d. 14)")
    recent_actuals: dict[str, dict[date, float]] | None = Field(
        None, description="Penarikan aktual (juta Rp) setelah tanggal data terakhir model: {atm_id: {tanggal: nilai}} "
                          "untuk SEMUA ATM, berurutan tanpa lubang. Ramalan lalu dimulai dari tanggal terakhir ini.")
    include_daily: bool = Field(True, description="False = hanya total per ATM")

    @field_validator("recent_actuals")
    @classmethod
    def non_negative(cls, v):
        if v:
            for atm, series in v.items():
                if any(x < 0 for x in series.values()):
                    raise ValueError(f"nilai penarikan {atm} tidak boleh negatif")
        return v


class DayForecast(BaseModel):
    date: date
    h: int
    forecast: float = Field(..., description="Ramalan median (juta Rp)")
    p10: float
    p90: float
    p05: float
    p95: float
    recommended_load: float = Field(..., description="Rekomendasi isi kas (kuantil optimal, dibatasi kapasitas)")
    capped_by_capacity: bool
    is_holiday: bool
    payday_window: str
    lebaran_phase: str


class AtmForecast(BaseModel):
    atm_id: str
    location_type: str
    cassette_capacity: float
    total_forecast: float
    total_p10: float
    total_p90: float
    total_recommended_load: float
    days: list[DayForecast] | None = None


class ForecastResponse(BaseModel):
    model_version: str
    model_name: str
    data_until: date = Field(..., description="Tanggal data aktual terakhir yang dipakai (origin ramalan)")
    horizon: int
    load_quantile: float
    warnings: list[str]
    atms: list[AtmForecast]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
