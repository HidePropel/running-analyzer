import json
from typing import Dict, List, Optional

from .angles import angle_between_3points, angle_from_vertical

# MediaPipe Pose Landmarks 인덱스 (오른쪽만 사용)
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28
RIGHT_HEEL = 30
RIGHT_FOOT_INDEX = 32
RIGHT_EAR = 8
RIGHT_EYE = 5


def load_pose_frames(path: str) -> List[Dict]:
    """pose_data.json 파일을 읽어 frames 리스트 반환."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("frames", [])


def _get_point(frame: Dict, landmark_id: int):
    """
    한 프레임에서 특정 landmark_id의 (x, y) 좌표 가져오기.
    해당 포인트가 없으면 None 반환.
    """
    for kp in frame.get("keypoints", []):
        if kp["id"] == landmark_id:
            return (kp["x"], kp["y"])
    return None


def _summary_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """리스트 값에 대한 min/max/mean 계산 (값 없으면 None)."""
    if not values:
        return {"min": None, "max": None, "mean": None}
    v_min = min(values)
    v_max = max(values)
    v_mean = sum(values) / len(values)
    return {"min": v_min, "max": v_max, "mean": v_mean}

def _smooth(values: List[float], window: int = 5) -> List[float]:
    """
    이동 평균 기반 간단 smoothing.
    최근 window개 값의 평균을 사용.
    """
    if not values:
        return []

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i+1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed

def _filter_spikes(values: List[float], max_step: float = 60.0) -> List[float]:
    """
    프레임 간 변화가 max_step(도)보다 큰 값은 '튀는 값'으로 보고 제거.
    (예: 90도였다가 바로 170도로 점프하는 경우)
    """
    if not values:
        return []

    filtered = [values[0]]
    last = values[0]

    for v in values[1:]:
        if abs(v - last) <= max_step:
            filtered.append(v)
            last = v
        # 너무 튀면 그냥 버린다

    return filtered

def compute_knee_angles(frames: List[Dict]) -> Dict:
    """오른쪽 무릎 각도(hip-knee-ankle)를 프레임별로 계산하고 요약."""
    raw_angles = []
    frame_indices = []

    for idx, frame in enumerate(frames):
        hip = _get_point(frame, RIGHT_HIP)
        knee = _get_point(frame, RIGHT_KNEE)
        ankle = _get_point(frame, RIGHT_ANKLE)

        if hip and knee and ankle:
            angle = angle_between_3points(hip, knee, ankle)
            if angle is not None:
                raw_angles.append(angle)
                frame_indices.append(idx)

    clipped = [a for a in raw_angles if 40 <= a <= 180]
    no_spikes = _filter_spikes(clipped, max_step=50.0)
    smoothed = _smooth(no_spikes, window=5)

    return {
        "name": "right_knee_angle",
        "unit": "deg",
        "per_frame": smoothed,
        "frame_indices": frame_indices[:len(smoothed)],
        "summary": _summary_stats(smoothed),
    }


def compute_elbow_angles(frames: List[Dict]) -> Dict:
    raw_angles = []
    frame_indices = []

    for idx, frame in enumerate(frames):
        shoulder = _get_point(frame, RIGHT_SHOULDER)
        elbow = _get_point(frame, RIGHT_ELBOW)
        wrist = _get_point(frame, RIGHT_WRIST)

        if shoulder and elbow and wrist:
            angle = angle_between_3points(shoulder, elbow, wrist)
            if angle is not None:
                raw_angles.append(angle)
                frame_indices.append(idx)

    clipped = [a for a in raw_angles if 50 <= a <= 160]
    no_spikes = _filter_spikes(clipped, max_step=50.0)
    smoothed = _smooth(no_spikes, window=5)

    return {
        "name": "right_elbow_angle",
        "unit": "deg",
        "per_frame": smoothed,
        "frame_indices": frame_indices[:len(smoothed)],
        "summary": _summary_stats(smoothed),
    }


def compute_torso_lean(frames: List[Dict]) -> Dict:
    raw_angles = []
    frame_indices = []

    for idx, frame in enumerate(frames):
        hip = _get_point(frame, RIGHT_HIP)
        shoulder = _get_point(frame, RIGHT_SHOULDER)

        if hip and shoulder:
            angle = angle_from_vertical(hip, shoulder)
            if angle is not None:
                raw_angles.append(angle)
                frame_indices.append(idx)

    clipped = [a for a in raw_angles if -30 <= a <= 30]
    smoothed = _smooth(clipped, window=5)

    return {
        "name": "torso_lean",
        "unit": "deg",
        "per_frame": smoothed,
        "frame_indices": frame_indices[:len(smoothed)],
        "summary": _summary_stats(smoothed),
    }

def compute_head_tilt(frames: List[Dict]) -> Dict:
    """
    머리 기울기 (어깨 기준 귀/눈의 수직 대비 각도).
    - 귀가 있으면 귀 기준, 없으면 눈 기준 사용.
    - 프레임 간 갑작스러운 큰 변화가 있으면 보조 포인트(눈)를 사용해 완화.
    """
    raw_angles = []
    frame_indices = []
    last_angle = None

    for idx, frame in enumerate(frames):
        shoulder = _get_point(frame, RIGHT_SHOULDER)
        ear = _get_point(frame, RIGHT_EAR)
        eye = _get_point(frame, RIGHT_EYE)

        if shoulder is None:
            continue

        ear_angle = angle_from_vertical(shoulder, ear) if ear else None
        eye_angle = angle_from_vertical(shoulder, eye) if eye else None

        angle = ear_angle
        if angle is None:
            angle = eye_angle
        else:
            # 직전 프레임과 20도 이상 튀면 눈 기준 각도도 참고
            if last_angle is not None and abs(angle - last_angle) > 20 and eye_angle is not None:
                angle = eye_angle

        if angle is not None:
            raw_angles.append(angle)
            frame_indices.append(idx)
            last_angle = angle

    smoothed = _smooth(raw_angles, window=5)

    return {
        "name": "head_tilt",
        "unit": "deg",
        "per_frame": smoothed,
        "frame_indices": frame_indices[:len(smoothed)],
        "summary": _summary_stats(smoothed),
    }


def compute_foot_strike_distance(frames: List[Dict]) -> Dict:
    """
    발 착지 위치 (엉덩이 기준 x 거리, 다리 길이로 정규화된 %).
    - ankle_y 분포 상위 15% 부근(0.85 분위수) 근처를 '지면 근처'로 보고,
      그보다 위에 떠 있는 프레임(중간 스윙 등)은 제외.
    - foot_strike_px = ankle.x - hip.x
      > 0 : 몸보다 앞에서 착지 (overstride 경향)
      < 0 : 몸보다 뒤/아래에서 착지
    - leg_length(pixels) 로 나누고 *100 → %로 표현.
    """
    # 1) 전체 프레임에서 발목 y값 모아 지면 근사값 계산
    ankle_ys = []
    for frame in frames:
        ankle = _get_point(frame, RIGHT_ANKLE)
        if ankle:
            ankle_ys.append(ankle[1])

    if not ankle_ys:
        return {
            "name": "foot_strike_distance",
            "unit": "%",
            "per_frame": [],
            "frame_indices": [],
            "summary": _summary_stats([]),
        }

    sorted_y = sorted(ankle_ys)
    idx_thr = int(len(sorted_y) * 0.85)
    idx_thr = max(0, min(idx_thr, len(sorted_y) - 1))
    threshold_y = sorted_y[idx_thr]

    values = []
    frame_indices = []

    # 2) 지면 근처(ankle_y >= threshold_y)에서만 착지 위치 계산
    for idx, frame in enumerate(frames):
        hip = _get_point(frame, RIGHT_HIP)
        ankle = _get_point(frame, RIGHT_ANKLE)
        if not hip or not ankle:
            continue

        ankle_y = ankle[1]
        # 지면 근처가 아닌 프레임(발이 떠 있는 중간 스윙)은 스킵
        if ankle_y < threshold_y:
            continue

        foot_strike_px = ankle[0] - hip[0]
        leg_length = abs(hip[1] - ankle[1])
        if leg_length <= 0:
            continue

        normalized = (foot_strike_px / leg_length) * 100.0
        values.append(normalized)
        frame_indices.append(idx)

    if not values:
        return {
            "name": "foot_strike_distance",
            "unit": "%",
            "per_frame": [],
            "frame_indices": [],
            "summary": _summary_stats([]),
        }

    smoothed = _smooth(values, window=3)

    return {
        "name": "foot_strike_distance",
        "unit": "%",
        "per_frame": smoothed,
        "frame_indices": frame_indices[:len(smoothed)],
        "summary": _summary_stats(smoothed),
    }



def compute_ankle_angle(frames: List[Dict]) -> Dict:
    """
    발목 각도 (무릎-발목-발끝 3점 각도).
    - 50~160도 범위만 유효값으로 사용 (너무 접히거나 펴진 값은 인식 오류로 처리).
    """
    raw_angles = []
    frame_indices = []

    for idx, frame in enumerate(frames):
        knee = _get_point(frame, RIGHT_KNEE)
        ankle = _get_point(frame, RIGHT_ANKLE)
        foot = _get_point(frame, RIGHT_FOOT_INDEX)

        if knee and ankle and foot:
            angle = angle_between_3points(knee, ankle, foot)
            if angle is not None:
                raw_angles.append(angle)
                frame_indices.append(idx)

    clipped = [a for a in raw_angles if 50 <= a <= 160]
    no_spikes = _filter_spikes(clipped, max_step=50.0)
    smoothed = _smooth(no_spikes, window=5)

    return {
        "name": "right_ankle_angle",
        "unit": "deg",
        "per_frame": smoothed,
        "frame_indices": frame_indices[:len(smoothed)],
        "summary": _summary_stats(smoothed),
    }


def compute_knee_lift(frames: List[Dict]) -> Dict:
    """
    무릎 높이 (각도 기반):
    - 상체(어깨→엉덩이) 벡터와 허벅지(엉덩이→무릎) 벡터 사이의 각도를 사용.
    - 오른쪽 다리가 화면 오른쪽으로 나가는 상황을 가정하고,
      knee.x > hip.x 인 프레임(앞다리 스윙)에 대해서만 계산한다.
    - angle_hip = angle_between_3points(shoulder, hip, knee)  [deg]
      * 서있는 다리: ~180도 (무릎 거의 안 든 상태)
      * 허벅지 수평에 가까울수록: 90도 근처
    - 정규화 knee_lift = (180 - angle_hip) / 90
      * 0.0  ≈ 무릎 거의 안 든 상태
      * 1.0  ≈ 허벅지가 수평 이상으로 올라간 상태
      (0~1 사이로 클램프)
    """
    values = []
    frame_indices = []

    for idx, frame in enumerate(frames):
        shoulder = _get_point(frame, RIGHT_SHOULDER)
        hip = _get_point(frame, RIGHT_HIP)
        knee = _get_point(frame, RIGHT_KNEE)

        if not (shoulder and hip and knee):
            continue

        # 앞다리(카메라 기준 오른쪽 방향)만 사용
        if knee[0] <= hip[0]:
            continue

        angle_hip = angle_between_3points(shoulder, hip, knee)  # deg
        if angle_hip is None:
            continue

        # 180도(다리 수직 아래) → 0.0, 90도(허벅지 수평) → 1.0
        raw_lift = (180.0 - angle_hip) / 90.0
        lift = max(0.0, min(1.0, raw_lift))  # 0~1로 클램프

        values.append(lift)
        frame_indices.append(idx)

    if not values:
        return {
            "name": "right_knee_lift",
            "unit": "norm",
            "per_frame": [],
            "frame_indices": [],
            "summary": _summary_stats([]),
        }

    smoothed = _smooth(values, window=5)

    return {
        "name": "right_knee_lift",
        "unit": "norm",
        "per_frame": smoothed,
        "frame_indices": frame_indices[:len(smoothed)],
        "summary": _summary_stats(smoothed),
    }


def compute_arm_swing(frames: List[Dict]) -> Dict:
    """
    팔 스윙 폭 (어깨 대비 손목 x 위치를 상체 높이로 정규화한 %).
    - per_frame: 각 프레임별 현재 팔의 상대 위치 (%)
    - summary: min/max/mean, swing_range(앞/뒤로 흔든 범위)
    """
    values = []
    frame_indices = []

    for idx, frame in enumerate(frames):
        shoulder = _get_point(frame, RIGHT_SHOULDER)
        wrist = _get_point(frame, RIGHT_WRIST)
        hip = _get_point(frame, RIGHT_HIP)

        if not shoulder or not wrist or not hip:
            continue

        dx = wrist[0] - shoulder[0]
        body_height = abs(shoulder[1] - hip[1])
        if body_height <= 0:
            continue

        normalized = (abs(dx) / body_height) * 100.0
        values.append(normalized)
        frame_indices.append(idx)

    if not values:
        return {
            "name": "right_arm_swing",
            "unit": "%",
            "per_frame": [],
            "frame_indices": [],
            "summary": _summary_stats([]),
        }

    smoothed = _smooth(values, window=5)
    summary = _summary_stats(smoothed)
    swing_range = None
    if summary["max"] is not None and summary["min"] is not None:
        swing_range = summary["max"] - summary["min"]

    return {
        "name": "right_arm_swing",
        "unit": "%",
        "per_frame": smoothed,
        "frame_indices": frame_indices[:len(smoothed)],
        "summary": summary,
        "swing_range": swing_range,
    }



def analyze_pose_file(path: str) -> Dict:
    """
    pose_data.json 경로를 받아서,
    여러 지표를 한 번에 계산해 종합 결과 반환.
    """
    frames = load_pose_frames(path)

    knee = compute_knee_angles(frames)
    elbow = compute_elbow_angles(frames)
    torso = compute_torso_lean(frames)
    head = compute_head_tilt(frames)
    foot_strike = compute_foot_strike_distance(frames)
    ankle = compute_ankle_angle(frames)
    knee_lift = compute_knee_lift(frames)
    arm_swing = compute_arm_swing(frames)

    return {
        "source": path,
        "num_frames": len(frames),
        "metrics": [
            knee,
            elbow,
            torso,
            head,
            foot_strike,
            ankle,
            knee_lift,
            arm_swing,
        ],
    }

def _compute_body_width(frame):
    """어깨~엉덩이 픽셀 거리(가로 폭)를 이용해 정규화 기준을 만든다."""
    shoulder = _get_point(frame, RIGHT_SHOULDER)
    hip = _get_point(frame, RIGHT_HIP)

    if shoulder and hip:
        return abs(shoulder[0] - hip[0])
    return None