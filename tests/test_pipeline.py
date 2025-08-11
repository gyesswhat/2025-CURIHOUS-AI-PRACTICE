# tests/test_pipeline.py
import json
import sys
from pathlib import Path
import pandas as pd

# 프로젝트 루트를 import 경로에 추가 (pytest 실행 위치 상관없이 동작)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def test_mock_data_exists(tmp_path):
    from ai.mock_data import make_dataset
    out = tmp_path / "mock.csv"
    make_dataset(n_users=200, out_csv=str(out))
    assert out.exists()
    df = pd.read_csv(out)
    assert "density_level" in df.columns
    assert len(df) == 200

def test_train_and_predict(tmp_path):
    from ai.mock_data import make_dataset
    from ai.train_density_lgbm import train
    from ai.predict_and_generate_config import run

    csv_path = tmp_path / "mock.csv"
    make_dataset(n_users=400, out_csv=str(csv_path))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)

    # train (수정된 반환값: preproc, model, le)
    preproc, model, le = train(csv_path=str(csv_path), artifacts_dir=str(artifacts))
    assert (artifacts / "preproc.joblib").exists()
    assert (artifacts / "density_model_lgbm.joblib").exists()
    assert (artifacts / "label_encoder.joblib").exists()
    assert (artifacts / "report.json").exists()
    assert (artifacts / "confusion_matrix.csv").exists()

    # predict + uiconfig
    run(user_index=0, csv_path=str(csv_path), artifacts_dir=str(artifacts))

    # 산출물 확인
    uiconfig = artifacts / "uiconfig_user0.json"
    assert uiconfig.exists()
    cfg = json.loads(uiconfig.read_text(encoding="utf-8"))
    assert cfg["density"] in {"LOW", "MID", "HIGH"}
    assert "style" in cfg and "maxTiles" in cfg["style"]
    assert "layout_order" in cfg and 1 <= len(cfg["layout_order"]) <= 8