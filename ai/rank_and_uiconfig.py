# ai/rank_and_uiconfig.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import hashlib
import math

FUNCTION_IDS = [f"F{str(i).zfill(2)}" for i in range(1, 17)]

FUNCTION_META: Dict[str, Dict[str, str]] = {
    # 가입상품관리
    "F01": {"category": "가입상품관리", "label": "예금 가입",        "component": "card"},
    "F02": {"category": "가입상품관리", "label": "적금 가입",        "component": "card"},
    "F03": {"category": "가입상품관리", "label": "펀드/해지",        "component": "card"},
    "F04": {"category": "가입상품관리", "label": "주식계좌 개설",     "component": "card"},
    # 자산관리
    "F10": {"category": "자산관리",   "label": "자산 리포트",      "component": "card"},
    # 공과금
    "F14": {"category": "공과금",     "label": "공과금 납부",      "component": "icon"},
    # 외환
    "F15": {"category": "외환",       "label": "해외 송금",        "component": "icon"},
    "F16": {"category": "외환",       "label": "환율 계산기",      "component": "icon"},
    # 금융편의
    "F11": {"category": "금융편의",   "label": "계좌 조회",        "component": "icon"},
    "F06": {"category": "금융편의",   "label": "송금/이체",        "component": "icon"},
    "F12": {"category": "금융편의",   "label": "자동이체 관리",     "component": "list_item"},
    "F13": {"category": "금융편의",   "label": "카드 내역",        "component": "icon"},
    # 대출/보험
    "F05": {"category": "대출",       "label": "대출 조회",        "component": "list_item"},
    "F07": {"category": "대출",       "label": "대출 상환",        "component": "list_item"},
    "F08": {"category": "대출",       "label": "대출 신청",        "component": "button"},
    "F09": {"category": "보험",       "label": "보험 조회",        "component": "list_item"},
}

GROUP_TEMPLATES: List[Dict[str, Any]] = [
    {"key": "quick",      "title": "빠른 실행",   "component": "icon_grid", "capacity": {"LOW": 4, "MID": 6, "HIGH": 8}},
    {"key": "transfer",   "title": "이체/계좌",   "component": "icon_grid", "capacity": {"LOW": 4, "MID": 6, "HIGH": 8}},
    {"key": "card",       "title": "카드/결제",   "component": "icon_grid", "capacity": {"LOW": 4, "MID": 6, "HIGH": 8}},
    {"key": "bill",       "title": "공과금/정부", "component": "icon_grid", "capacity": {"LOW": 4, "MID": 6, "HIGH": 8}},
    {"key": "fx",         "title": "외환",       "component": "icon_grid", "capacity": {"LOW": 4, "MID": 6, "HIGH": 8}},
    {"key": "wealth",     "title": "자산관리",   "component": "cards",     "capacity": {"LOW": 4, "MID": 6, "HIGH": 8}},
    {"key": "onboarding", "title": "가입상품",   "component": "cards",     "capacity": {"LOW": 3, "MID": 4, "HIGH": 6}},
    {"key": "loan",       "title": "대출",       "component": "list",      "capacity": {"LOW": 3, "MID": 5, "HIGH": 7}},
    {"key": "ins",        "title": "보험",       "component": "list",      "capacity": {"LOW": 3, "MID": 5, "HIGH": 7}},
    {"key": "others",     "title": "더보기",     "component": "chips",     "capacity": {"LOW": 6, "MID": 8, "HIGH": 12}},
]

CATEGORY_TO_GROUP = {
    "금융편의": "transfer",
    "카드":     "card",
    "공과금":   "bill",
    "외환":     "fx",
    "자산관리": "wealth",
    "가입상품관리": "onboarding",
    "대출":     "loan",
    "보험":     "ins",
}

def density_to_style(density: str) -> Dict[str, Any]:
    # 전역 스타일(폰트/버튼/대비/최대타일)
    if density == "LOW":
        base = {"fontScale": 1.25, "buttonSize": "xl", "maxTiles": 4, "contrast": "high"}
    elif density == "MID":
        base = {"fontScale": 1.1,  "buttonSize": "lg", "maxTiles": 6, "contrast": "normal"}
    else:
        base = {"fontScale": 1.0,  "buttonSize": "md", "maxTiles": 8, "contrast": "normal"}
    return base

def group_layout_props(density: str, component: str) -> Dict[str, Any]:
    # 그룹별 레이아웃 속성(밀도 기준)
    if component == "icon_grid":
        cols = {"LOW": 2, "MID": 3, "HIGH": 4}[density]
        return {
            "cols": cols,
            "gutter": {"LOW": 16, "MID": 12, "HIGH": 8}[density],
            "rowHeight": {"LOW": 84, "MID": 72, "HIGH": 64}[density],
            "titleStyle": {"size": {"LOW": 20, "MID": 18, "HIGH": 16}[density], "weight": 700},
            "itemSize": {"LOW": "xl", "MID": "lg", "HIGH": "md"}[density],
        }
    if component == "cards":
        return {
            "cardAspect": {"LOW": 1.6, "MID": 1.4, "HIGH": 1.2}[density],
            "gutter": {"LOW": 16, "MID": 12, "HIGH": 8}[density],
            "titleStyle": {"size": {"LOW": 20, "MID": 18, "HIGH": 16}[density], "weight": 700},
        }
    if component == "list":
        return {
            "dense": {"LOW": False, "MID": True, "HIGH": True}[density],
            "divider": True,
            "titleStyle": {"size": {"LOW": 20, "MID": 18, "HIGH": 16}[density], "weight": 700},
        }
    if component == "chips":
        return {
            "chipSize": {"LOW": "lg", "MID": "md", "HIGH": "sm"}[density],
            "wrap": True,
            "gutter": {"LOW": 12, "MID": 10, "HIGH": 8}[density],
            "titleStyle": {"size": {"LOW": 20, "MID": 18, "HIGH": 16}[density], "weight": 700},
        }
    return {}

# ---------------- 점수식(강화) + 사용자별 시드 기반 타이브레이커 ----------------

def _seed_from_user(row: pd.Series) -> int:
    uid = str(row.get("user_id", "unknown"))
    h = hashlib.sha256(uid.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit

def _safe(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default

def _raw(fid: str, row: pd.Series) -> Dict[str, float]:
    return {
        "click": _safe(row, f"click_rate_{fid}", 0.0),       # 0~0.5
        "entry": _safe(row, f"entry_count_{fid}", 0.0),      # 0~20
        "ret":   _safe(row, f"return_count_{fid}", 0.0),     # 0~10
        "dur":   _safe(row, f"visit_duration_{fid}", 0.0),   # 1~300
        "days":  _safe(row, f"last_access_days_{fid}", 30.0) # 0~30 (작을수록 최근)
    }

def _score_for(fid: str, row: pd.Series, rng: np.random.RandomState) -> float:
    c = _raw(fid, row)
    # 정규화 + 비선형
    click = c["click"] / 0.5                         # 0~1
    entry = min(c["entry"] / 20.0, 1.0)              # 0~1
    ret   = min(c["ret"]   / 10.0, 1.0)              # 0~1
    dur   = 1.0 - min(c["dur"] / 300.0, 1.0)         # 짧을수록 가산 (페이지 체류가 길면 탐색 헤매는 것으로 간주)
    rec   = 1.0 / (1.0 + c["days"])                  # 최근성 (0~1, 최근일수 작을수록 큼)
    # 가중합 + 상호작용
    base = 0.40*click + 0.35*entry + 0.15*ret + 0.10*rec + 0.10*dur
    # 최근성 보정(지수 감쇠): 오래 안 쓴 건 깎기
    decay = math.exp(-c["days"]/15.0)                # 0~1
    score = base * decay
    # 사용자별 deterministic jitter(동점 방지)
    score += rng.normal(0, 0.01)
    return float(max(score, 0.0))

def _rank_functions(row: pd.Series, candidates: List[str]) -> List[Tuple[str, float]]:
    rng = np.random.RandomState(_seed_from_user(row))
    scored = [(fid, _score_for(fid, row, rng)) for fid in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# --------------- 그룹 빌드 & 개인화 제안 로직 ----------------

def _make_groups_shell(density: str) -> List[Dict[str, Any]]:
    groups = []
    for tpl in GROUP_TEMPLATES:
        g = {
            "key": tpl["key"],
            "title": tpl["title"],
            "component": tpl["component"],
            "capacity": tpl["capacity"][density],
            "layout": group_layout_props(density, tpl["component"]),
            "items": []
        }
        groups.append(g)
    return groups

def _target_group_for(fid: str) -> str:
    meta = FUNCTION_META.get(fid)
    if not meta:
        return "others"
    return CATEGORY_TO_GROUP.get(meta["category"], "others")

def _insert_item(groups: List[Dict[str, Any]], gkey: str, fid: str):
    meta = FUNCTION_META.get(fid, {"label": fid, "component": "icon"})
    g = next(gr for gr in groups if gr["key"] == gkey)
    g["items"].append({"fid": fid, "label": meta["label"], "component": meta["component"]})

def _cap(groups: List[Dict[str, Any]], gkey: str) -> int:
    return next(gr for gr in groups if gr["key"] == gkey)["capacity"]

def _fill_pinned(groups: List[Dict[str, Any]], pinned_fids: List[str]):
    quick = next(gr for gr in groups if gr["key"] == "quick")
    for fid in pinned_fids:
        if len(quick["items"]) < quick["capacity"]:
            _insert_item(groups, "quick", fid)

def _personalize_group_order(groups: List[Dict[str, Any]], scored: List[Tuple[str,float]]):
    # quick은 맨 위 고정, 나머지는 카테고리 총점 기준으로 재정렬
    totals: Dict[str, float] = {g["key"]: 0.0 for g in groups}
    for fid, s in scored:
        gkey = _target_group_for(fid)
        for g in groups:
            if g["key"] == gkey:
                totals[gkey] += s
                break
    fixed = [g for g in groups if g["key"] == "quick"]
    rest  = [g for g in groups if g["key"] != "quick"]
    rest.sort(key=lambda g: totals.get(g["key"], 0.0), reverse=True)
    return fixed + rest

def _place_ranked(groups: List[Dict[str, Any]], ranked: List[Tuple[str,float]], exclude: set):
    # exclude: 이미 홈에 있는 기능, 핀 고정 등
    for fid, _ in ranked:
        if fid in exclude:
            continue
        gkey = _target_group_for(fid)
        g = next(gr for gr in groups if gr["key"] == gkey)
        if len(g["items"]) < g["capacity"]:
            _insert_item(groups, gkey, fid)

def _suggest_additions(groups: List[Dict[str, Any]], ranked: List[Tuple[str,float]], exclude: set, row: pd.Series, topk_global=6):
    """현재 홈에 없는 상위 기능들을 그룹별/전역 제안으로 제공"""
    # 아직 어떤 그룹에도 들어가지 않은 후보
    already = {it["fid"] for g in groups for it in g["items"]}
    cand = [(fid, sc) for fid, sc in ranked if fid not in exclude and fid not in already]
    # 그룹별 제안
    suggestions_by_group: Dict[str, List[Dict[str, Any]]] = {g["key"]: [] for g in groups}
    for fid, sc in cand:
        gkey = _target_group_for(fid)
        days = float(row.get(f"last_access_days_{fid}", 30.0))
        suggestions_by_group[gkey].append({
            "fid": fid,
            "label": FUNCTION_META.get(fid, {"label": fid})["label"],
            "reason": {"score": round(sc, 3), "recency_days": days, "expected_slot": gkey}
        })
    # 전역 TOP-N
    global_top = sorted(cand, key=lambda x: x[1], reverse=True)[:topk_global]
    global_suggestions = []
    for fid, sc in global_top:
        gkey = _target_group_for(fid)
        days = float(row.get(f"last_access_days_{fid}", 30.0))
        global_suggestions.append({
            "fid": fid,
            "label": FUNCTION_META.get(fid, {"label": fid})["label"],
            "reason": {"score": round(sc, 3), "recency_days": days, "expected_slot": gkey}
        })
    return suggestions_by_group, global_suggestions

def build_uiconfig(
        row: pd.Series,
        predicted_density: str,
        pinned_fids: List[str] | None = None,
        is_senior: bool = False,
        current_home_fids: List[str] | None = None
) -> Dict[str, Any]:
    pinned_fids = pinned_fids or []
    current_home_fids = set(current_home_fids or [])

    style = density_to_style(predicted_density)

    # 후보 기능(데이터가 실제로 있는 것만)
    candidates = [fid for fid in FUNCTION_IDS if f"entry_count_{fid}" in row.index]

    # 점수 & 개인화 정렬
    scored = _rank_functions(row, candidates)

    # 그룹 골격 + 개인화된 그룹 순서
    groups = _make_groups_shell(predicted_density)
    groups = _personalize_group_order(groups, scored)

    # 핀 고정
    _fill_pinned(groups, pinned_fids)

    # 현재 홈에 이미 있는 기능은 제외(exclude)하고 배치
    exclude = set(pinned_fids) | set(current_home_fids)
    _place_ranked(groups, scored, exclude)

    # 추천(현재 홈에 없는 것들 중 상위)
    suggest_by_group, global_suggest = _suggest_additions(groups, scored, exclude, row)

    # 플랫 순서
    layout_order_flat: List[str] = []
    for g in groups:
        layout_order_flat.extend([it["fid"] for it in g["items"]])

    # 각 그룹에 추천도 함께 담아주기(상위 3개만 표시)
    for g in groups:
        s = suggest_by_group.get(g["key"], [])
        g["suggest_additions"] = s[:3]

    # 점수 표기
    scores = {fid: round(sc, 3) for fid, sc in scored}

    highlight_for_senior = bool(is_senior and predicted_density == "LOW")

    return {
        "density": predicted_density,
        "style": style,
        "groups": groups,
        "layout_order_flat": layout_order_flat,
        "scores": scores,
        "pinned": list(pinned_fids),
        "highlight_for_senior": highlight_for_senior,
        "global_suggestions": global_suggest
    }
