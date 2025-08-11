import pandas as pd
import numpy as np
import uuid
import random

# 고정된 난수 생성기를 사용해 실행 시마다 동일한 패턴을 재현 가능하게 함
RNG = np.random.default_rng(42)

# 기능 ID 목록 생성: F01 ~ F16
FUNCTION_IDS = [f"F{str(i).zfill(2)}" for i in range(1, 17)]

# 연령대 목록 (분석 시 시니어 여부 판단에 사용)
AGE_GROUPS = ["20s", "30s", "40s", "60s", "70s"]  # 필요 시 "50s"도 추가 가능

# 사용자 기기 종류
DEVICE_TYPES = ["Android", "iOS"]

# 접속 시간대 클러스터
ACCESS_TIME_CLUSTERS = ["morning", "afternoon", "evening", "night"]


def is_senior(age_group: str) -> bool:
    """
    주어진 연령대(age_group)가 시니어 그룹(60대 이상)에 해당하는지 판단.
    """
    return age_group in ["60s", "70s"]


def gen_user_row():
    """
    개별 사용자 데이터 1행(row) 생성
    - 기본 사용자 속성 (연령대, 기기, 접속시간대 등)
    - UI 행동 지표 (작업 시간, 오클릭, 스크롤 깊이 등)
    - 기능별 사용 기록 (진입 횟수, 마지막 접속일, 성공률)
    """
    age_group = random.choice(AGE_GROUPS)

    row = {
        # 고유 사용자 식별자
        "user_id": str(uuid.uuid4()),

        # 연령대
        "age_group": age_group,

        # 시니어 여부 Boolean
        "is_senior": is_senior(age_group),

        # 기기 종류
        "device_type": random.choice(DEVICE_TYPES),

        # 접속 시간대
        "access_time_cluster": random.choice(ACCESS_TIME_CLUSTERS),

        # 세션 수: 1~11회 사이
        "session_count": int(RNG.integers(1, 12)),

        # 화면 이동 다양성 지표 (0.8~3.8 범위)
        "tap_path_entropy": float(np.round(RNG.uniform(0.8, 3.8), 2)),

        # 태스크 완료까지 걸린 시간(초), 감마분포로 생성
        "task_time_sec": float(np.round(RNG.gamma(shape=3.0, scale=8.0), 1)),

        # 오클릭(잘못 누른) 횟수, 포아송 분포로 생성
        "misclicks": int(RNG.poisson(1.2)),

        # 첫 클릭까지 걸린 시간(초)
        "first_click_sec": float(np.round(RNG.uniform(0.3, 5.0), 2)),

        # 스크롤 깊이 (0.0~1.0 비율)
        "scroll_depth": float(np.round(RNG.uniform(0.0, 1.0), 2))
    }

    # 시니어 사용자는 평균적으로 작업 시간이 더 길고 오클릭이 많으며,
    # 첫 클릭까지의 시간이 길다는 가정으로 보정
    if row["is_senior"]:
        row["task_time_sec"] *= float(RNG.uniform(1.15, 1.4))  # 시간 증가
        row["misclicks"] += int(RNG.integers(0, 2))            # 오클릭 추가
        row["first_click_sec"] *= float(RNG.uniform(1.05, 1.4)) # 첫 클릭 지연

    # 각 기능별 사용 데이터 생성
    for fid in FUNCTION_IDS:
        # 기능 진입 횟수 (0~19회)
        entries = int(RNG.integers(0, 20))
        row[f"entry_count_{fid}"] = entries

        # 마지막 접속 후 경과일 (0~44일)
        row[f"last_access_days_{fid}"] = int(RNG.integers(0, 45))

        # 성공률: 기능 사용 성공 비율 (entries=0이면 0.0)
        succ_rate = 0.0 if entries == 0 else float(
            np.clip(RNG.normal(0.7, 0.15), 0.0, 1.0)
        )
        row[f"success_rate_{fid}"] = float(np.round(succ_rate, 3))

    return row


def _standardize(s: pd.Series) -> pd.Series:
    """
    표준화 함수: (값 - 평균) / 표준편차
    0에 가까울수록 평균에 근접, 양수/음수는 평균보다 높거나 낮음을 의미
    """
    return (s - s.mean()) / (s.std() + 1e-9)


def make_dataset(n_users=1000, out_csv="ui_behavior_mock.csv"):
    """
    전체 모의 사용자 데이터셋 생성
    1) n_users 수만큼 사용자 row 생성
    2) 난이도 점수(difficulty) 계산
    3) 난이도에 따라 UI 밀도 레벨(HIGH/MID/LOW) 부여
    4) CSV 파일로 저장
    """
    # 사용자 데이터 생성
    rows = [gen_user_row() for _ in range(n_users)]
    df = pd.DataFrame(rows)

    # 난이도 계산을 위한 각 지표 표준화
    z_time = _standardize(df["task_time_sec"])
    z_first = _standardize(df["first_click_sec"])
    z_mis = _standardize(df["misclicks"])
    z_scroll = _standardize(df["scroll_depth"])

    # 시니어는 동일 지표에서도 난이도 가중치 +0.2
    senior_bias = df["is_senior"].astype(float) * 0.2

    # 무작위 노이즈(±0.15)로 경계값 부근의 변동성 부여
    noise = RNG.normal(0, 0.15, size=len(df))

    # 난이도 점수 계산 (가중합)
    df["difficulty"] = (
            0.45 * z_time +      # 작업 시간 영향
            0.25 * z_first +     # 첫 클릭 지연 영향
            0.2 * z_mis +        # 오클릭 영향
            0.1 * z_scroll +     # 스크롤 깊이 영향
            senior_bias + noise  # 시니어 보정 + 노이즈
    )

    # 분위수 기준 난이도 라벨링 (상위 1/3 = LOW 밀도 권장, 하위 1/3 = HIGH 밀도 권장)
    q1, q2 = df["difficulty"].quantile([0.33, 0.66])
    df["density_level"] = np.where(
        df["difficulty"] >= q2, "LOW",
        np.where(df["difficulty"] <= q1, "HIGH", "MID")
    )

    # 중간 계산 컬럼 제거
    df = df.drop(columns=["difficulty"])

    # CSV 저장
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved: {out_csv} shape={df.shape}")


if __name__ == "__main__":
    # 기본 1000명 데이터 생성
    make_dataset(n_users=1000)
