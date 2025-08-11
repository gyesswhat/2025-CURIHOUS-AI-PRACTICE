# train_density_lgbm.py
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from lightgbm.callback import early_stopping, log_evaluation


FUNCTION_IDS = [f"F{str(i).zfill(2)}" for i in range(1, 17)]

def add_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    entry_cols = [f"entry_count_{fid}" for fid in FUNCTION_IDS]
    succ_cols  = [f"success_rate_{fid}" for fid in FUNCTION_IDS]
    rec_cols   = [f"last_access_days_{fid}" for fid in FUNCTION_IDS]

    df["entry_sum"]  = df[entry_cols].sum(axis=1)
    df["entry_mean"] = df[entry_cols].replace(0, np.nan).mean(axis=1).fillna(0)
    df["entry_max"]  = df[entry_cols].max(axis=1)

    df["success_mean"] = df[succ_cols].replace(0, np.nan).mean(axis=1).fillna(0)
    df["success_sum"]  = df[succ_cols].sum(axis=1)
    df["success_max"]  = df[succ_cols].max(axis=1)

    df["recency_mean"] = df[rec_cols].mean(axis=1)
    return df

def build_feature_lists():
    cat_feats = ["age_group","device_type","access_time_cluster","is_senior"]
    num_feats = [
        "session_count","tap_path_entropy",
        "task_time_sec","misclicks","first_click_sec","scroll_depth",
        "entry_sum","entry_mean","entry_max",
        "success_mean","success_sum","success_max",
        "recency_mean"
    ]
    return cat_feats, num_feats

def load_data(csv_path="ui_behavior_mock.csv"):
    df = pd.read_csv(csv_path)
    df = add_aggregates(df)
    return df

def train(csv_path="ui_behavior_mock.csv", artifacts_dir="artifacts"):
    df = load_data(csv_path)
    y = df["density_level"].values
    X = df.drop(columns=["density_level","user_id"])

    cat_feats, num_feats = build_feature_lists()
    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
            ("num", "passthrough", num_feats),
        ]
    )

    # label enc
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X[cat_feats + num_feats], y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # fit preprocessor, transform both train/valid
    preproc.fit(X_train)
    X_train_t = preproc.transform(X_train)
    X_valid_t = preproc.transform(X_valid)

    # LightGBM with early stopping (now in transformed space)
    clf = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=40,
        min_split_gain=0.01,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(
        X_train_t, y_train,
        eval_set=[(X_valid_t, y_valid)],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=0)  # 로그 숨김, 원하면 50 등으로
        ]
    )

    # eval
    y_pred = clf.predict(X_valid_t)
    report = classification_report(y_valid, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_valid, y_pred)

    # save artifacts: preproc, model, label encoder, meta, reports
    Path(artifacts_dir).mkdir(exist_ok=True)
    joblib.dump(preproc, f"{artifacts_dir}/preproc.joblib")
    joblib.dump(clf,     f"{artifacts_dir}/density_model_lgbm.joblib")
    joblib.dump(le,      f"{artifacts_dir}/label_encoder.joblib")

    with open(f"{artifacts_dir}/report.json","w",encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    np.savetxt(f"{artifacts_dir}/confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    meta = {
        "model_type": "lightgbm",
        "features_categorical": cat_feats,
        "features_numeric": num_feats,
        "classes": list(le.classes_),
        "best_iteration": int(getattr(clf, "best_iteration_", -1))
    }
    with open(f"{artifacts_dir}/meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] trained=lightgbm  macro_f1={report['macro avg']['f1-score']:.3f}  acc={report['accuracy']:.3f}  best_iter={meta['best_iteration']}")
    return preproc, clf, le

if __name__ == "__main__":
    train()
