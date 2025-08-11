# ai/rank_and_uiconfig.py
# --------------------------------------------------------------------------------------
# 목적
# - 사용자 행동 로그(row: pandas.Series)를 입력으로 받아 홈 화면 UI 구성을 생성한다.
# - 핵심 단계:
#     1) 전역 스타일(density 기반) 계산
#     2) 기능 후보(F01~F16) 개인화 점수 산출 및 정렬
#     3) 그룹 골격 생성 + 그룹 순서 개인화(카테고리 합산 점수 기반)
#     4) 핀 고정 아이템 채우기
#     5) 나머지 상위 기능을 그룹 용량(capacity)에 맞춰 배치
#     6) 홈에 없는 상위 기능은 그룹별/전역 추천(suggest_additions / global_suggestions)
#     7) 납품용 JSON 스키마로 반환
# --------------------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import hashlib
import math

# --------------------------------------------------------------------------------------
# 기능 ID 레인지 정의
# - F01 ~ F16 까지의 문자열 ID를 생성한다.
# - 실제 후보는 build 시점에 "해당 기능의 entry_count_* 컬럼이 존재"해야만 포함된다.
# --------------------------------------------------------------------------------------
FUNCTION_IDS = [f"F{str(i).zfill(2)}" for i in range(1, 17)]

# --------------------------------------------------------------------------------------
# 기능 메타데이터
# - 각 기능의 카테고리, 노출 라벨, 기본 컴포넌트 타입을 정의.
# - 그룹 배치는 CATEGORY_TO_GROUP 매핑을 통해 카테고리→그룹으로 연결된다.
# - component는 그룹 내 표현(아이콘/카드/리스트아이템/버튼 등) 힌트로 사용.
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# 그룹 템플릿
# - 그룹별 기본 설정(키, 타이틀, 컴포넌트 타입, 밀도별 수용량)을 정의.
# - build 시점에 density에 맞춰 capacity가 확정된다.
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# 카테고리 → 그룹 매핑
# - 기능 카테고리가 어느 그룹에 배치될지를 결정.
# - 매핑되지 않은 카테고리는 "others"로 간다(방어 로직 _target_group_for 참고).
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# density_to_style
# - 전역 스타일(폰트 배율, 버튼 크기, 최대 타일 수, 대비)을 밀도에 따라 결정.
# - LOW 밀도: 시니어 친화(큰 폰트/큰 버튼/적은 타일/고대비)
# - HIGH 밀도: 정보 밀집(작은 폰트/작은 버튼/많은 타일/보통 대비)
# --------------------------------------------------------------------------------------
def density_to_style(density: str) -> Dict[str, Any]:
    # 전역 스타일(폰트/버튼/대비/최대타일)
    if density == "LOW":
        base = {"fontScale": 1.25, "buttonSize": "xl", "maxTiles": 4, "contrast": "high"}
    elif density == "MID":
        base = {"fontScale": 1.1,  "buttonSize": "lg", "maxTiles": 6, "contrast": "normal"}
    else:
        base = {"fontScale": 1.0,  "buttonSize": "md", "maxTiles": 8, "contrast": "normal"}
    return base

# --------------------------------------------------------------------------------------
# group_layout_props
# - 그룹 컴포넌트 타입(icon_grid/cards/list/chips)에 따라
#   density별 레이아웃 세부 속성(열 수, 간격, 행 높이, 타이틀 크기 등)을 리턴.
# --------------------------------------------------------------------------------------
def group_layout_props(density: str, component: str) -> Dict[str, Any]:
    # 그룹별 레이아웃 속성(밀도 기준)
    if component == "icon_grid":
        # icon grid는 열 수(cols)와 간격/행높이/아이템 크기가 density에 따라 달라진다.
        cols = {"LOW": 2, "MID": 3, "HIGH": 4}[density]
        return {
            "cols": cols,
            "gutter": {"LOW": 16, "MID": 12, "HIGH": 8}[density],
            "rowHeight": {"LOW": 84, "MID": 72, "HIGH": 64}[density],
            "titleStyle": {"size": {"LOW": 20, "MID": 18, "HIGH": 16}[density], "weight": 700},
            "itemSize": {"LOW": "xl", "MID": "lg", "HIGH": "md"}[density],
        }
    if component == "cards":
        # cards는 카드 종횡비(cardAspect)와 간격, 타이틀 스타일을 density에 따라 조정
        return {
            "cardAspect": {"LOW": 1.6, "MID": 1.4, "HIGH": 1.2}[density],
            "gutter": {"LOW": 16, "MID": 12, "HIGH": 8}[density],
            "titleStyle": {"size": {"LOW": 20, "MID": 18, "HIGH": 16}[density], "weight": 700},
        }
    if component == "list":
        # list는 dense 여부가 MID/HIGH에서 True, LOW에서 False (여유 공간 확보)
        return {
            "dense": {"LOW": False, "MID": True, "HIGH": True}[density],
            "divider": True,
            "titleStyle": {"size": {"LOW": 20, "MID": 18, "HIGH": 16}[density], "weight": 700},
        }
    if component == "chips":
        # chips는 칩 크기, 줄바꿈, 간격 등을 density에 따라 조정
        return {
            "chipSize": {"LOW": "lg", "MID": "md", "HIGH": "sm"}[density],
            "wrap": True,
            "gutter": {"LOW": 12, "MID": 10, "HIGH": 8}[density],
            "titleStyle": {"size": {"LOW": 20, "MID": 18, "HIGH": 16}[density], "weight": 700},
        }
    # 정의되지 않은 컴포넌트 타입의 경우 빈 dict 반환(방어)
    return {}

# ======================================================================================
# 점수 산출 관련 유틸
# - _seed_from_user: 사용자 고유 seed 생성(동점 시 안정적 난수 jiter)
# - _safe: row 컬럼 조회 시 안전 파싱
# - _raw: 기능별 원시 피처 추출(클릭률/진입수/재방문/체류/최근일수)
# - _score_for: 원시 피처 → 정규화/가중합/감쇠/지터 → 최종 점수
# - _rank_functions: 모든 후보에 대해 점수 계산 후 내림차순 정렬
# ======================================================================================

def _seed_from_user(row: pd.Series) -> int:
    """
    사용자 고유 seed 생성:
    - user_id가 있으면 해당 값 기준, 없으면 "unknown".
    - SHA-256 해시 → 상위 8 hex → 32-bit int 변환.
    - 목적: 사용자별로 작은 난수 지터를 '일관성 있게' 주어 동점 방지.
    """
    uid = str(row.get("user_id", "unknown"))
    h = hashlib.sha256(uid.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit

def _safe(row: pd.Series, key: str, default: float = 0.0) -> float:
    """
    안전한 컬럼 접근:
    - 숫자형 변환 실패/결측 시 default 리턴
    - 모든 점수식 입력은 float로 처리
    """
    try:
        return float(row.get(key, default))
    except Exception:
        return default

def _raw(fid: str, row: pd.Series) -> Dict[str, float]:
    """
    기능별 원시 피처 묶음:
    - click_rate_Fxx: 0~0.5 범위를 가정(내부 규칙)
    - entry_count_Fxx: 0~20
    - return_count_Fxx: 0~10
    - visit_duration_Fxx: 1~300 (초) — 길수록 '헤맸다'로 간주되어 감점
    - last_access_days_Fxx: 0~30 (작을수록 최근)
    """
    return {
        "click": _safe(row, f"click_rate_{fid}", 0.0),       # 0~0.5
        "entry": _safe(row, f"entry_count_{fid}", 0.0),      # 0~20
        "ret":   _safe(row, f"return_count_{fid}", 0.0),     # 0~10
        "dur":   _safe(row, f"visit_duration_{fid}", 0.0),   # 1~300
        "days":  _safe(row, f"last_access_days_{fid}", 30.0) # 0~30 (작을수록 최근)
    }

def _score_for(fid: str, row: pd.Series, rng: np.random.RandomState) -> float:
    """
    기능 점수 산출식:
    1) 원시값 정규화
       - click: /0.5 → 0~1
       - entry: /20, ret: /10 → 0~1에서 clip
       - dur: 1 - min(dur/300, 1) → 짧을수록 가산(헤맴 방지)
       - recency: 1/(1+days) → 최근일수 작을수록 큼(0~1)
    2) 가중합:
       base = 0.40*click + 0.35*entry + 0.15*ret + 0.10*rec + 0.10*dur
    3) 최근성 감쇠:
       decay = exp(-days/15) → 오래 안 쓴 기능 페널티
    4) 사용자 고유 지터:
       rng.normal(0, 0.01) 더해 동점 방지(일관된 시드 기반)
    """
    c = _raw(fid, row)

    # 1) 정규화
    click = c["click"] / 0.5                         # 0~1
    entry = min(c["entry"] / 20.0, 1.0)              # 0~1
    ret   = min(c["ret"]   / 10.0, 1.0)              # 0~1
    dur   = 1.0 - min(c["dur"] / 300.0, 1.0)         # 짧을수록 가산
    rec   = 1.0 / (1.0 + c["days"])                  # 최근성

    # 2) 가중합
    base = 0.40*click + 0.35*entry + 0.15*ret + 0.10*rec + 0.10*dur

    # 3) 최근성 보정(지수 감쇠)
    decay = math.exp(-c["days"]/15.0)                # 0~1
    score = base * decay

    # 4) 사용자별 deterministic jitter(동점 방지)
    score += rng.normal(0, 0.01)

    return float(max(score, 0.0))

def _rank_functions(row: pd.Series, candidates: List[str]) -> List[Tuple[str, float]]:
    """
    후보 기능 리스트에 대해 개인화 점수 계산 후 내림차순으로 정렬해 반환.
    - rng 시드는 사용자 고유(seed_from_user)로 동일 사용자에 대해 일관성 유지.
    """
    rng = np.random.RandomState(_seed_from_user(row))
    scored = [(fid, _score_for(fid, row, rng)) for fid in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ======================================================================================
# 그룹 빌드 & 개인화 배치/추천
# - _make_groups_shell: density에 맞춘 빈 그룹 골격 생성
# - _target_group_for: fid → 그룹 키로 변환(메타 카테고리 기반)
# - _insert_item: 특정 그룹에 기능 아이템 추가
# - _cap: 그룹 용량 조회
# - _fill_pinned: quick 그룹에 핀 고정 기능 먼저 채우기
# - _personalize_group_order: 그룹 순서 개인화(카테고리별 점수 합계 내림차순)
# - _place_ranked: exclude 제외하고 상위 기능을 그룹 용량 내에서 채우기
# - _suggest_additions: 홈에 없는 상위 기능을 그룹별/전역 추천으로 산출
# ======================================================================================

def _make_groups_shell(density: str) -> List[Dict[str, Any]]:
    """
    빈 그룹 골격 생성:
    - GROUP_TEMPLATES를 순회하며 해당 density의 capacity/레이아웃을 채워 넣는다.
    - items는 빈 리스트로 시작.
    """
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
    """
    fid의 카테고리를 조회해 해당 그룹 키를 반환.
    - 미정의/매핑 불가 시 "others"로 폴백.
    """
    meta = FUNCTION_META.get(fid)
    if not meta:
        return "others"
    return CATEGORY_TO_GROUP.get(meta["category"], "others")

def _insert_item(groups: List[Dict[str, Any]], gkey: str, fid: str):
    """
    지정 그룹(gkey)에 기능(fid)을 아이템으로 추가.
    - label/component는 FUNCTION_META를 사용(미정의 시 fid/기본 icon으로 폴백).
    """
    meta = FUNCTION_META.get(fid, {"label": fid, "component": "icon"})
    g = next(gr for gr in groups if gr["key"] == gkey)
    g["items"].append({"fid": fid, "label": meta["label"], "component": meta["component"]})

def _cap(groups: List[Dict[str, Any]], gkey: str) -> int:
    """해당 그룹의 capacity(최대 아이템 수)를 반환."""
    return next(gr for gr in groups if gr["key"] == gkey)["capacity"]

def _fill_pinned(groups: List[Dict[str, Any]], pinned_fids: List[str]):
    """
    quick 그룹에 핀 고정 기능부터 채운다.
    - quick 그룹은 최상단 고정이며, capacity를 초과하지 않는 선에서 순차 배치.
    """
    quick = next(gr for gr in groups if gr["key"] == "quick")
    for fid in pinned_fids:
        if len(quick["items"]) < quick["capacity"]:
            _insert_item(groups, "quick", fid)

def _personalize_group_order(groups: List[Dict[str, Any]], scored: List[Tuple[str,float]]):
    """
    그룹 순서 개인화:
    - quick은 항상 맨 위 고정.
    - 나머지 그룹들은 카테고리별 점수 합계(totals)에 따라 내림차순 정렬.
    """
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
    """
    상위 점수 기능을 그룹에 실제 배치:
    - exclude(이미 홈에 있는 기능, 핀 고정 등)는 건너뜀.
    - 각 기능의 target group에 여유 capacity가 있으면 배치.
    - 그룹별 capacity 초과 시 해당 그룹에는 더 이상 배치하지 않음.
    """
    for fid, _ in ranked:
        if fid in exclude:
            continue
        gkey = _target_group_for(fid)
        g = next(gr for gr in groups if gr["key"] == gkey)
        if len(g["items"]) < g["capacity"]:
            _insert_item(groups, gkey, fid)

def _suggest_additions(groups: List[Dict[str, Any]], ranked: List[Tuple[str,float]], exclude: set, row: pd.Series, topk_global=6):
    """
    추천 기능 산출:
    - 현재 홈(groups)에 없는 상위 기능들 중에서 그룹별 추천/전역 추천을 만든다.
    - reason에는 (score, recency_days, expected_slot)을 포함해 추천 근거를 표현.
    - 전역 추천(global_top)은 상위 topk_global개.
    """
    # 현재 이미 배치된 fid 집합
    already = {it["fid"] for g in groups for it in g["items"]}

    # 추천 후보: exclude(핀/이미 홈) + already 제외
    cand = [(fid, sc) for fid, sc in ranked if fid not in exclude and fid not in already]

    # 그룹별 추천 리스트 초기화
    suggestions_by_group: Dict[str, List[Dict[str, Any]]] = {g["key"]: [] for g in groups}
    for fid, sc in cand:
        gkey = _target_group_for(fid)
        days = float(row.get(f"last_access_days_{fid}", 30.0))  # recency 표시용
        suggestions_by_group[gkey].append({
            "fid": fid,
            "label": FUNCTION_META.get(fid, {"label": fid})["label"],
            "reason": {"score": round(sc, 3), "recency_days": days, "expected_slot": gkey}
        })

    # 전역 TOP-N 추천
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

# --------------------------------------------------------------------------------------
# build_uiconfig: 최종 엔트리 함수
# - 입력:
#     row: 사용자별 행동 피처가 담긴 pandas.Series
#     predicted_density: "LOW"/"MID"/"HIGH" (난이도 역매핑 결과 등으로 외부에서 결정)
#     pinned_fids: quick 그룹에 고정할 fid 리스트(옵션)
#     is_senior: 시니어 여부 (LOW 밀도일 때 강조 테마 표시 여부 결정)
#     current_home_fids: 이미 홈에 있는 기능(fid) 목록(옵션) — 재배치/중복 방지
# - 출력:
#     JSON(dict) 스키마:
#       density/style/groups/layout_order_flat/scores/pinned/highlight_for_senior/global_suggestions
# --------------------------------------------------------------------------------------
def build_uiconfig(
        row: pd.Series,
        predicted_density: str,
        pinned_fids: List[str] | None = None,
        is_senior: bool = False,
        current_home_fids: List[str] | None = None
) -> Dict[str, Any]:
    pinned_fids = pinned_fids or []
    current_home_fids = set(current_home_fids or [])

    # 1) 전역 스타일 결정 (density 기반)
    style = density_to_style(predicted_density)

    # 2) 후보 기능 결정:
    #    - row에 entry_count_Fxx 컬럼이 존재하는 기능만 후보에 포함
    #    - (데이터 없는 기능은 점수 계산 불가 → 제외)
    candidates = [fid for fid in FUNCTION_IDS if f"entry_count_{fid}" in row.index]

    # 3) 후보 기능 점수 계산 & 정렬(개인화)
    scored = _rank_functions(row, candidates)

    # 4) 그룹 골격 생성 + 그룹 순서 개인화(카테고리 점수 합산 기반)
    groups = _make_groups_shell(predicted_density)
    groups = _personalize_group_order(groups, scored)

    # 5) 핀 고정 기능을 quick 그룹에 우선 채움
    _fill_pinned(groups, pinned_fids)

    # 6) 이미 홈에 있는 기능(exclude) 제외하고 상위 기능 배치
    exclude = set(pinned_fids) | set(current_home_fids)
    _place_ranked(groups, scored, exclude)

    # 7) 추천(홈에 없는 상위 기능) 계산: 그룹별 + 전역
    suggest_by_group, global_suggest = _suggest_additions(groups, scored, exclude, row)

    # 8) 플랫 순서(위→아래, 좌→우): 프론트가 단순 렌더 순서 검증/디버깅에 활용 가능
    layout_order_flat: List[str] = []
    for g in groups:
        layout_order_flat.extend([it["fid"] for it in g["items"]])

    # 9) 각 그룹에 상위 3개 추천만 담아 글머리표처럼 가볍게 노출
    for g in groups:
        s = suggest_by_group.get(g["key"], [])
        g["suggest_additions"] = s[:3]

    # 10) 기능별 점수(dict) 표기 (디버깅/분석/툴팁에 활용)
    scores = {fid: round(sc, 3) for fid, sc in scored}

    # 11) 시니어 강조 여부: 시니어이고 LOW 밀도일 때만 True
    highlight_for_senior = bool(is_senior and predicted_density == "LOW")

    # 12) 최종 JSON 반환
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