# --- 파일: mock_data.py ---
import pandas as pd
import numpy as np
import uuid
import random

RNG = np.random.default_rng(42)

FUNCTION_IDS = [f"F{str(i).zfill(2)}" for i in range(1, 17)]
AGE_GROUPS = ["20s","30s","40s","60s","70s"]  # 50s 빠졌으면 추가하세요
DEVICE_TYPES = ["Android","iOS"]
ACCESS_TIME_CLUSTERS = ["morning","afternoon","evening","night"]

def is_senior(age_group: str) -> bool:
    return age_group in ["60s","70s"]

def gen_user_row():
    age_group = random.choice(AGE_GROUPS)
    row = {
        "user_id": str(uuid.uuid4()),
        "age_group": age_group,
        "is_senior": is_senior(age_group),
        "device_type": random.choice(DEVICE_TYPES),
        "access_time_cluster": random.choice(ACCESS_TIME_CLUSTERS),
        "session_count": int(RNG.integers(1, 12)),
        "tap_path_entropy": float(np.round(RNG.uniform(0.8, 3.8), 2)),

        "task_time_sec": float(np.round(RNG.gamma(shape=3.0, scale=8.0), 1)),
        "misclicks": int(RNG.poisson(1.2)),
        "first_click_sec": float(np.round(RNG.uniform(0.3, 5.0), 2)),
        "scroll_depth": float(np.round(RNG.uniform(0.0, 1.0), 2))
    }
    if row["is_senior"]:
        row["task_time_sec"] *= float(RNG.uniform(1.15, 1.4))
        row["misclicks"] += int(RNG.integers(0, 2))
        row["first_click_sec"] *= float(RNG.uniform(1.05, 1.4))

    for fid in FUNCTION_IDS:
        entries = int(RNG.integers(0, 20))
        row[f"entry_count_{fid}"] = entries
        row[f"last_access_days_{fid}"] = int(RNG.integers(0, 45))
        succ_rate = 0.0 if entries == 0 else float(np.clip(RNG.normal(0.7, 0.15), 0.0, 1.0))
        row[f"success_rate_{fid}"] = float(np.round(succ_rate, 3))
    return row

def _standardize(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)

def make_dataset(n_users=1000, out_csv="ui_behavior_mock.csv"):
    rows = [gen_user_row() for _ in range(n_users)]
    df = pd.DataFrame(rows)

    # 난이도 점수: 높을수록 "어려운 사용자" → LOW 밀도 권장
    z_time  = _standardize(df["task_time_sec"])
    z_first = _standardize(df["first_click_sec"])
    z_mis   = _standardize(df["misclicks"])
    z_scroll= _standardize(df["scroll_depth"])

    # 시니어는 같은 지표라도 난이도를 약간 더 높게 가정(편의상 +0.2)
    senior_bias = df["is_senior"].astype(float) * 0.2

    noise = RNG.normal(0, 0.15, size=len(df))  # 경계 근처 랜덤성
    df["difficulty"] = (
            0.45 * z_time + 0.25 * z_first + 0.2 * z_mis + 0.1 * z_scroll
            + senior_bias + noise
    )

    # 분위수 기반 라벨 (상위 1/3 = LOW, 하위 1/3 = HIGH)
    q1, q2 = df["difficulty"].quantile([0.33, 0.66])
    df["density_level"] = np.where(
        df["difficulty"] >= q2, "LOW",
        np.where(df["difficulty"] <= q1, "HIGH", "MID")
    )
    df = df.drop(columns=["difficulty"])

    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved: {out_csv} shape={df.shape}")

if __name__ == "__main__":
    make_dataset(n_users=1000)
