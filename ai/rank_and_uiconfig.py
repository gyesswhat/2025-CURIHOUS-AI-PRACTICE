# rank_and_uiconfig.py
import json
import numpy as np
import pandas as pd

FUNCTION_IDS = [f"F{str(i).zfill(2)}" for i in range(1, 17)]

def rfs_score(row: pd.Series, is_senior=False):
    """Recency-Frequency-Success 기반 점수."""
    scores = {}
    wR, wF, wS = (0.35, 0.35, 0.30) if not is_senior else (0.30, 0.30, 0.40)
    for fid in FUNCTION_IDS:
        F = row[f"entry_count_{fid}"]
        R_days = row[f"last_access_days_{fid}"]
        S = row[f"success_rate_{fid}"]

        R = 1.0 - np.tanh(R_days / 30.0)  # 최근일 작은게 유리
        Fn = np.tanh(F / 10.0)            # 빈도 축약
        Sn = S                             # 이미 0~1

        scores[fid] = float(wR*R + wF*Fn + wS*Sn)
    return scores

def density_to_style(density: str):
    if density == "LOW":
        return {"fontScale": 1.25, "buttonSize": "xl", "maxTiles": 4, "contrast": "high"}
    if density == "MID":
        return {"fontScale": 1.10, "buttonSize": "lg", "maxTiles": 6, "contrast": "normal"}
    return {"fontScale": 1.00, "buttonSize": "md", "maxTiles": 8, "contrast": "normal"}

def build_uiconfig(row: pd.Series, predicted_density: str, pinned_fids=None, is_senior=False):
    pinned_fids = pinned_fids or []
    scores = rfs_score(row, is_senior=is_senior)

    candidates = [fid for fid in FUNCTION_IDS if fid not in pinned_fids]
    ranked = sorted(candidates, key=lambda fid: scores[fid], reverse=True)

    style = density_to_style(predicted_density)
    topN = style["maxTiles"]
    layout = pinned_fids + ranked[:max(0, topN - len(pinned_fids))]

    return {
        "density": predicted_density,
        "style": style,
        "layout_order": layout,
        "scores": {fid: round(scores[fid], 3) for fid in layout},
        "pinned": pinned_fids,
    }

if __name__ == "__main__":
    # 단일 레코드 샘플 출력
    from train_density_lgbm import load_data
    df = load_data("ui_behavior_mock.csv")
    row = df.iloc[0]
    cfg = build_uiconfig(row, predicted_density="LOW", pinned_fids=["F06"], is_senior=bool(row["is_senior"]))
    print(json.dumps(cfg, ensure_ascii=False, indent=2))