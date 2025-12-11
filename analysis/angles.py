import math
from typing import Optional, Tuple

Point = Tuple[float, float]


def angle_between_3points(a: Point, b: Point, c: Point) -> Optional[float]:
    """
    a, b, c 는 (x, y) 좌표.
    b를 꼭짓점으로 하는 ∠ABC 각도를 degree로 반환.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    # 벡터 BA, BC
    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])

    if mag_ba == 0 or mag_bc == 0:
        return None

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    cos_theta = dot / (mag_ba * mag_bc)

    # 수치 오차 보정
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def angle_from_vertical(p1: Point, p2: Point) -> Optional[float]:
    """
    p1 -> p2 벡터가 '수직선'과 이루는 각도.
    - 0~90도 범위에서 기울기 크기를 표현
    - 오른쪽으로 기울면 +, 왼쪽으로 기울면 - 부호
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    mag = math.hypot(dx, dy)
    if mag == 0:
        return None

    # 아래 방향 (0,1) 기준 각도 계산
    cos_theta = dy / mag
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    # 수직선과의 "차이"를 0~90도로 보정
    if angle_deg > 90:
        angle_deg = 180 - angle_deg

    # 방향성: 오른쪽으로 기울면 +, 왼쪽은 -
    sign = 1.0
    if dx < 0:
        sign = -1.0

    return angle_deg * sign
