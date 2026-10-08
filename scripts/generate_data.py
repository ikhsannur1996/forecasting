"""Generate data sintetis penarikan tunai harian 20 ATM (2022–2025).

Output:
  data/atm_master.csv        : profil ATM (lokasi, kota, kapasitas kaset)
  data/atm_withdrawals.csv   : penarikan harian per ATM (juta Rp)

Pola yang dibangkitkan (berbeda per tipe lokasi): mingguan, tanggal gajian, Lebaran
(THR & mudik), libur nasional, puncak Desember, libur semester kampus, tren turun
karena pembayaran digital, dan noise AR(1) agar ada autokorelasi seperti data nyata.

    python scripts/generate_data.py --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.calendar_id import calendar_frame  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
START, END = "2022-01-01", "2025-12-31"

# tipe lokasi: (jumlah ATM, level dasar juta/hari, pola Senin..Minggu, gajian, puncak pra-Lebaran, pasca-Lebaran, libur)
LOCATION_TYPES = {
    "mall":          (4, 170, [0.85, 0.85, 0.9, 0.9, 1.05, 1.35, 1.40], 1.20, 1.45, 0.90, 1.30),
    "office":        (4, 150, [1.15, 1.10, 1.10, 1.10, 1.20, 0.45, 0.35], 1.50, 1.30, 0.30, 0.40),
    "residential":   (5, 110, [1.00, 0.95, 0.95, 0.95, 1.05, 1.10, 1.00], 1.40, 1.60, 0.70, 0.95),
    "transport_hub": (3, 200, [1.05, 0.95, 0.95, 1.00, 1.20, 0.95, 1.10], 1.15, 1.55, 1.15, 0.90),
    "university":    (2, 70,  [1.10, 1.10, 1.10, 1.05, 1.00, 0.55, 0.40], 1.10, 1.10, 0.30, 0.50),
    "market":        (2, 160, [1.00, 0.95, 0.95, 1.00, 1.05, 1.10, 1.05], 1.20, 2.00, 0.60, 0.90),
}
CITIES = ["jakarta", "jakarta", "bandung", "surabaya", "semarang", "medan", "makassar"]


def generate(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cal = calendar_frame(pd.date_range(START, END, freq="D"))
    n_days = len(cal)
    t_years = np.arange(n_days) / 365.25

    atms, rows = [], []
    i = 0
    for loc, (count, base, dow_pat, payday, pre_leb, post_leb, holiday) in LOCATION_TYPES.items():
        for _ in range(count):
            i += 1
            atm_id = f"ATM{i:02d}"
            level = base * rng.lognormal(0, 0.25)
            trend = -0.04 + rng.normal(0, 0.02)             # tren tahunan (penurunan karena digital)
            capacity = float(np.ceil(level * 2.5 / 50) * 50)  # kapasitas kaset ≈ 2,5 hari rata-rata
            atms.append({"atm_id": atm_id, "location_type": loc, "city": rng.choice(CITIES), "cassette_capacity": capacity})

            dow = np.array(dow_pat)[cal["dow"].to_numpy()]
            dom = cal["dom"].to_numpy()
            pay = np.where((dom >= 25) | (dom <= 2), payday, np.where(dom <= 9, 1 + (payday - 1) * 0.3, 1.0))
            d = cal["days_to_lebaran"].to_numpy()
            pre = np.where((d >= -21) & (d <= -1), 1 + (pre_leb - 1) * (1 - (np.abs(d) - 1) / 21), 1.0)
            post = np.where((d >= 0) & (d <= 6), post_leb, np.where((d >= 7) & (d <= 14), 1 + (post_leb - 1) * (14 - d) / 8, 1.0))
            hol = np.where(cal["is_holiday"].to_numpy() == 1, holiday, 1.0)
            dec = np.where(cal["is_december_peak"].to_numpy() == 1, 1.25 if loc == "mall" else 1.08, 1.0)
            sem = np.where((cal["is_semester_break"].to_numpy() == 1) & (loc == "university"), 0.55, 1.0)
            tr = (1 + trend) ** t_years

            ar = np.zeros(n_days)                             # noise AR(1) di skala log
            eps = rng.normal(0, 0.11, n_days)
            for k in range(1, n_days):
                ar[k] = 0.45 * ar[k - 1] + eps[k]
            y = level * dow * pay * pre * post * hol * dec * sem * tr * np.exp(ar)
            rows.append(pd.DataFrame({"date": cal["date"], "atm_id": atm_id, "withdrawal": np.round(y, 2)}))

    return pd.DataFrame(atms), pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    master, tx = generate(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(DATA_DIR / "atm_master.csv", index=False)
    tx.to_csv(DATA_DIR / "atm_withdrawals.csv", index=False)
    print(master.groupby("location_type").size().to_dict(), tx.shape)
    print(tx.groupby(tx["date"].dt.year)["withdrawal"].sum().round(0).to_dict())
