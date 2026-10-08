"""Kalender Indonesia untuk forecasting kebutuhan uang tunai ATM.

Modul ini dipakai oleh generator data, feature engineering, notebook, DAN API,
sehingga hanya bergantung pada pandas/numpy. Tanggal libur keagamaan (Lebaran,
Idul Adha, Imlek) mengikuti penetapan pemerintah; perbarui setiap tahun.
"""

import numpy as np
import pandas as pd

LEBARAN = pd.to_datetime(["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31", "2026-03-20", "2027-03-10"])
IDUL_ADHA = pd.to_datetime(["2022-07-10", "2023-06-29", "2024-06-17", "2025-06-06", "2026-05-27", "2027-05-17"])
IMLEK = pd.to_datetime(["2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17", "2027-02-06"])
FIXED_HOLIDAYS = ["01-01", "05-01", "06-01", "08-17", "12-25"]  # Tahun Baru, Buruh, Pancasila, HUT RI, Natal
CALENDAR_VALID_UNTIL = pd.Timestamp("2027-12-31")


def national_holidays(start="2021-12-01", end="2028-01-31") -> pd.DatetimeIndex:
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    fixed = [pd.Timestamp(f"{y}-{md}") for y in years for md in FIXED_HOLIDAYS]
    lebaran_2days = list(LEBARAN) + list(LEBARAN + pd.Timedelta(days=1))
    return pd.DatetimeIndex(sorted(set(fixed + lebaran_2days + list(IDUL_ADHA) + list(IMLEK))))


HOLIDAYS = national_holidays()

# ----------------------------------------------------------------------------- binning kalender
# tanggal dalam bulan → jendela gajian (batas berdasarkan kebiasaan gaji di Indonesia: tgl 25–1)
DOM_BINS = {"edges": [0, 2, 9, 19, 24, 31], "labels": ["gajian_1-2", "awal_3-9", "tengah_10-19", "menjelang_20-24", "gajian_25-31"]}
# jarak ke Lebaran (hari) → fase mudik/THR
LEBARAN_BINS = {
    "edges": [-10_000, -22, -15, -8, -1, 3, 7, 14, 10_000],
    "labels": ["normal", "H-21_s.d._H-15", "H-14_s.d._H-8", "H-7_s.d._H-1", "H0_s.d._H+3", "H+4_s.d._H+7", "H+8_s.d._H+14", "normal_setelah"],
}


def days_to_lebaran(dates) -> np.ndarray:
    """Selisih hari terhadap Lebaran terdekat (negatif = sebelum Lebaran)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates)).values.astype("datetime64[D]").astype(np.int64)
    leb = LEBARAN.values.astype("datetime64[D]").astype(np.int64)
    diff = d[:, None] - leb[None, :]
    return diff[np.arange(len(d)), np.abs(diff).argmin(axis=1)]


def calendar_frame(dates) -> pd.DataFrame:
    """Fitur kalender mentah + hasil binning untuk setiap tanggal."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    dtl = days_to_lebaran(dates)
    lebaran_bin = pd.cut(dtl, LEBARAN_BINS["edges"], labels=LEBARAN_BINS["labels"]).astype(str)
    lebaran_bin = np.where(lebaran_bin == "normal_setelah", "normal", lebaran_bin)
    is_hol = dates.isin(HOLIDAYS)
    return pd.DataFrame({
        "date": dates,
        "dow": dates.dayofweek,                      # 0 = Senin
        "month": dates.month,
        "dom": dates.day,
        "dom_bin": pd.cut(dates.day, DOM_BINS["edges"], labels=DOM_BINS["labels"]).astype(str),
        "days_to_lebaran": dtl,
        "lebaran_bin": lebaran_bin,
        "is_holiday": is_hol.astype(int),
        "is_pre_holiday": dates.shift(1, freq="D").isin(HOLIDAYS).astype(int),
        "is_post_holiday": dates.shift(-1, freq="D").isin(HOLIDAYS).astype(int),
        "is_december_peak": ((dates.month == 12) & (dates.day >= 15)).astype(int),
        "is_semester_break": (dates.month.isin([1, 7, 8])).astype(int),
    })
