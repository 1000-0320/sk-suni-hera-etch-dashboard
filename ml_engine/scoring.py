"""공통 종합 품질 점수 공식 (Pass Rate 60% + Uniformity 25% + 목표 근접도 15%).

model.py와 ml_engine/{isolation,trench,gate,metal}_core.py 양쪽에서 같은 공식을 써야
"목표 품질"이 예측(Output B)뿐 아니라 파라미터 추천(Output E)에도 일관되게 반영된다.
model.py -> ml_engine.{process}_core 방향의 기존 import와 순환되지 않도록 별도 모듈로 분리했다.
"""


def _clip(value, lo=0.0, hi=100.0):
    if value != value:  # NaN (결측 실측치) -> 만점으로 둔갑하는 대신 최하점 처리
        return lo
    return max(lo, min(hi, value))


def _uniformity_score(value: float, max_allowed: float) -> float:
    """0이면 100점, max_allowed 이상이면 0점 (선형)"""
    if max_allowed <= 0:
        return 0.0
    return _clip(100 * (1 - value / max_allowed))


def _proximity_score(predicted: float, target: float, tolerance_pct: float = 0.05) -> float:
    """target과 오차 0이면 100점, 오차가 target의 tolerance_pct 이상이면 0점 (선형)"""
    if target == 0:
        return 100.0 if predicted == 0 else 0.0
    tolerance = abs(target) * tolerance_pct
    error = abs(predicted - target)
    return _clip(100 * (1 - error / tolerance))


def compute_composite_score(result: dict, targets: dict) -> dict:
    """종합 품질 점수 = Pass Rate 60% + Uniformity 25% + 목표 근접도 15%"""
    pass_rate_score = _clip(result["Overall Spec Pass Rate"])

    cd_u_score = _uniformity_score(result["CD Uniformity"], targets["max_cd_uniformity"])
    depth_u_score = _uniformity_score(result["Depth Uniformity"], targets["max_depth_uniformity"])
    uniformity_score = (cd_u_score + depth_u_score) / 2

    proximity_scores = [
        _proximity_score(result["Top CD"], targets["target_top_cd"]),
        _proximity_score(result["Mid CD"], targets["target_mid_cd"]),
        _proximity_score(result["Bottom CD"], targets["target_bottom_cd"]),
        _proximity_score(result["Depth"], targets["target_depth"]),
    ]
    target_proximity_score = sum(proximity_scores) / len(proximity_scores)

    total = pass_rate_score * 0.60 + uniformity_score * 0.25 + target_proximity_score * 0.15

    return {
        "total": round(total, 1),
        "pass_rate_score": round(pass_rate_score, 1),
        "uniformity_score": round(uniformity_score, 1),
        "target_proximity_score": round(target_proximity_score, 1),
    }
