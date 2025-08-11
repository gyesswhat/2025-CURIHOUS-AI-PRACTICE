# ai/predict_and_generate_config.py
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ai.train_density_lgbm import add_aggregates, build_feature_lists
from ai.rank_and_uiconfig import build_uiconfig

# ColumnTransformer 단계 이름과 원본 카테고리형 피처들 (train_density_lgbm.py와 일치)
CAT_XFORM_NAME = "cat"
NUM_XFORM_NAME = "num"
CAT_FEATURES = ["age_group", "device_type", "access_time_cluster", "is_senior"]

def _to_dense(X):
    try:
        return X.toarray()
    except AttributeError:
        return np.asarray(X)

def _nice_name(raw: str) -> str:
    """ColumnTransformer가 붙이는 접두사 제거 + 발표용 보기 좋게"""
    name = raw
    if name.startswith(f"{CAT_XFORM_NAME}__"):
        name = name.split("__", 1)[1]
        for base in CAT_FEATURES:
            pref = base + "_"
            if name.startswith(pref):
                val = name[len(pref):]
                label = {
                    "age_group": "Age group",
                    "device_type": "Device",
                    "access_time_cluster": "Access time",
                    "is_senior": "Senior",
                }[base]
                return f"{label}: {val}"
        return name.replace("_", " ").title()
    if name.startswith(f"{NUM_XFORM_NAME}__"):
        name = name.split("__", 1)[1]
    # 숫자 피처는 snake_case -> Title Case
    name = name.replace("_", " ").title()
    # 소소한 가독성 보정
    name = (name.replace("Sec", "sec")
            .replace("Mean", "avg")
            .replace("Sum", "sum")
            .replace("Max", "max"))
    return name

def _get_feature_names_strict(preproc, cat_feats, num_feats):
    """전처리 후 피처명 생성 (가능하면 get_feature_names_out, 아니면 수동 구성)"""
    try:
        return list(preproc.get_feature_names_out())
    except Exception:
        pass

    names = []
    # cat
    try:
        ohe = preproc.named_transformers_[CAT_XFORM_NAME]
        if hasattr(ohe, "get_feature_names_out"):
            names.extend([f"{CAT_XFORM_NAME}__{n}" for n in ohe.get_feature_names_out(cat_feats)])
        else:
            for base, cats in zip(cat_feats, ohe.categories_):
                for c in cats:
                    names.append(f"{CAT_XFORM_NAME}__{base}_{c}")
    except Exception:
        pass
    # num (passthrough)
    names.extend([f"{NUM_XFORM_NAME}__{c}" for c in num_feats])
    return names

def _difficulty_to_ui_density(pred_density: str) -> str:
    """난이도 → UI 밀도 역매핑 (HIGH 난이도면 LOW 밀도로)"""
    mapping = {"HIGH": "LOW", "LOW": "HIGH", "MID": "MID"}
    return mapping.get(pred_density, "MID")

def predict_density_for_user(row: pd.Series, preproc, model, label_encoder):
    """단일 사용자: 전처리→예측(난이도 클래스), 확률, 변환피처, 피처명, 예측클래스인덱스 반환"""
    base = pd.DataFrame([row]).copy()
    base = add_aggregates(base)
    cat_feats, num_feats = build_feature_lists()

    X = base[cat_feats + num_feats]
    Xt = preproc.transform(X)
    Xt = _to_dense(Xt)

    proba = model.predict_proba(Xt)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]  # 난이도(HIGH/MID/LOW)
    proba_dict = {cls: float(proba[i]) for i, cls in enumerate(label_encoder.classes_)}

    feat_names = _get_feature_names_strict(preproc, cat_feats, num_feats)
    if len(feat_names) != Xt.shape[1]:
        # 길이 불일치 시 안전한 대체
        feat_names = [f"f{i}" for i in range(Xt.shape[1])]

    return pred_label, proba_dict, Xt, feat_names, pred_idx

def _extract_contrib_vector(model, Xt, n_features, pred_class_idx):
    """
    LightGBM pred_contrib 모양 전부 처리:
      A) (n, f+1, C)
      B) (n, C, f+1)
      C) (n, f+1)             ← 클래스 축 없는 희귀 케이스(공통 기여로 간주)
      D) (n, (f+1)*C)         ← 평탄화된 다중 클래스 → (n, C, f+1)로 복원
    반환: (sv: (n_features,), layout: str)
    """
    contrib = model.predict(Xt, pred_contrib=True, raw_score=True)
    contrib = np.asarray(contrib)

    if contrib.ndim == 3 and contrib.shape[1] == n_features + 1:               # A
        sv = contrib[0, :-1, pred_class_idx]
        return np.asarray(sv).ravel(), "n_f+1_C"
    if contrib.ndim == 3 and contrib.shape[2] == n_features + 1:               # B
        sv = contrib[0, pred_class_idx, :-1]
        return np.asarray(sv).ravel(), "n_C_f+1"
    if contrib.ndim == 2 and contrib.shape[1] == n_features + 1:               # C
        sv = contrib[0, :-1]
        return np.asarray(sv).ravel(), "n_f+1"
    if contrib.ndim == 2 and contrib.shape[1] % (n_features + 1) == 0:         # D
        C = contrib.shape[1] // (n_features + 1)
        reshaped = contrib.reshape(contrib.shape[0], C, n_features + 1)  # (n, C, f+1)
        sv = reshaped[0, pred_class_idx, :-1]
        return np.asarray(sv).ravel(), f"n_(f+1)*C -> reshaped(C={C})"

    return np.array([]), f"unexpected:{contrib.shape}"

def _save_debug(artifacts_dir: str, user_idx: int, info: dict):
    dbg = Path(artifacts_dir) / f"debug_user{user_idx}.json"
    dbg.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

def explain_with_contrib(model, Xt, feat_names, pred_class_idx, artifacts_dir, user_idx):
    n_features = len(feat_names)
    sv, layout = _extract_contrib_vector(model, Xt, n_features, pred_class_idx)

    _save_debug(artifacts_dir, user_idx, {
        "Xt_shape": list(Xt.shape),
        "n_features": n_features,
        "layout": layout,
        "sv_len": int(sv.size),
    })

    if sv.size == 0:
        return [], None

    # 절댓값 기준 Top-3
    top_idx = np.argsort(np.abs(sv))[-3:][::-1]
    top_features = [(_nice_name(feat_names[i]), float(sv[i])) for i in top_idx]

    # 발표용 바차트
    labels = [f for f, _ in top_features][::-1]
    vals = [v for _, v in top_features][::-1]
    plt.figure(figsize=(7, 3))
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), labels)
    plt.title(f"Top-3 contributions (class idx {pred_class_idx})")
    plt.tight_layout()
    out_png = f"{artifacts_dir}/contrib_user{user_idx}.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    return top_features, out_png

def run_all(n_users=10, csv_path="ui_behavior_mock.csv", artifacts_dir="artifacts", start_index=0):
    Path(artifacts_dir).mkdir(exist_ok=True, parents=True)

    preproc = joblib.load(f"{artifacts_dir}/preproc.joblib")
    model   = joblib.load(f"{artifacts_dir}/density_model_lgbm.joblib")
    le      = joblib.load(f"{artifacts_dir}/label_encoder.joblib")

    df = pd.read_csv(csv_path)
    end = min(start_index + n_users, len(df))

    for i in range(start_index, end):
        row = df.iloc[i]
        # 1) 난이도 예측
        difficulty_pred, proba_dict, Xt, feat_names, pred_idx = predict_density_for_user(row, preproc, model, le)
        # 2) UI 밀도 역매핑
        ui_density = _difficulty_to_ui_density(difficulty_pred)

        # 3) UI 구성 생성 (UI 밀도 적용!)
        cfg = build_uiconfig(
            row,
            predicted_density=ui_density,
            pinned_fids=["F06"],
            is_senior=bool(row["is_senior"])
        )
        # 메타 정보 추가
        cfg["difficulty_pred"] = difficulty_pred   # 모델의 원래 예측(난이도)
        cfg["ui_density"] = ui_density            # 실제 적용된 UI 밀도
        cfg["probabilities"] = proba_dict

        # 4) 기여도 Top-3
        top_feats, png = explain_with_contrib(model, Xt, feat_names, pred_idx, artifacts_dir, i)
        cfg["top_features"] = [{"feature": f, "contribution": v} for f, v in top_feats]
        if png:
            cfg["contrib_image"] = png

        # 5) 저장
        out_path = f"{artifacts_dir}/uiconfig_user{i}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        print(
            f"[USER {i}] difficulty={difficulty_pred}  → ui_density={ui_density}  "
            f"prob={proba_dict}  top3={cfg['top_features']}  → {out_path}"
        )

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--csv", type=str, default="ui_behavior_mock.csv")
    p.add_argument("--artifacts", type=str, default="artifacts")
    p.add_argument("--start", type=int, default=0)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_all(n_users=args.n, csv_path=args.csv, artifacts_dir=args.artifacts, start_index=args.start)