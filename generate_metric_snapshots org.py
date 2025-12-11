import json
import cv2
import os
import statistics
from typing import Dict, Tuple, Any, List
from datetime import datetime
from analysis.angles import angle_between_3points, angle_from_vertical
import os
print("[DEBUG] CWD:", os.getcwd())
from PIL import Image, ImageDraw, ImageFont
import numpy as np

POSE_JSON_PATH = "pose_data.json"
ANALYSIS_JSON_PATH = "analysis_result.json"
VIDEO_PATH = "run_side.mp4"
OUTPUT_DIR = "snapshots"

# Landmarks
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28
RIGHT_FOOT_INDEX = 32
RIGHT_EAR = 8
RIGHT_EYE = 5


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(analysis: Dict[str, Any], name: str) -> Dict[str, Any]:
    for m in analysis.get("metrics", []):
        if m.get("name") == name:
            return m
    return {}


# ---------- 1. 기준 길이 계산 (레그/암/토르소) ----------

def compute_reference_lengths(pose_data: Dict[str, Any]) -> Dict[str, float]:
    leg_lengths = []
    arm_lengths = []
    torso_heights = []
    ankle_ys = []

    for frame in pose_data["frames"]:
        pts = {}
        for kp in frame["keypoints"]:
            pts[kp["id"]] = (kp["x"], kp["y"])

        hip = pts.get(RIGHT_HIP)
        knee = pts.get(RIGHT_KNEE)
        ankle = pts.get(RIGHT_ANKLE)
        shoulder = pts.get(RIGHT_SHOULDER)
        wrist = pts.get(RIGHT_WRIST)

        if hip and ankle:
            lh = ((hip[0] - ankle[0]) ** 2 + (hip[1] - ankle[1]) ** 2) ** 0.5
            leg_lengths.append(lh)
            ankle_ys.append(ankle[1])

        if shoulder and wrist:
            ah = ((shoulder[0] - wrist[0]) ** 2 + (shoulder[1] - wrist[1]) ** 2) ** 0.5
            arm_lengths.append(ah)

        if shoulder and hip:
            th = abs(shoulder[1] - hip[1])
            torso_heights.append(th)

    def med(lst):
        return statistics.median(lst) if lst else 0.0

    # “지면”을 발목 y의 상위 90% 근처로 추정
    ground_y = 0.0
    if ankle_ys:
        sorted_y = sorted(ankle_ys)
        idx = int(len(sorted_y) * 0.9)
        idx = max(0, min(idx, len(sorted_y) - 1))
        ground_y = sorted_y[idx]

    ref = {
        "leg": med(leg_lengths),
        "arm": med(arm_lengths),
        "torso": med(torso_heights),
        "ground_y": ground_y,
    }
    print("[INFO] reference lengths:", ref)
    return ref



# ---------- 2. 프레임 품질 체크 ----------

def is_good_frame_for_metric(metric_name: str,
                             pts: Dict[int, Tuple[int, int]],
                             ref_len: Dict[str, float]) -> bool:
    hip = pts.get(RIGHT_HIP)
    knee = pts.get(RIGHT_KNEE)
    ankle = pts.get(RIGHT_ANKLE)
    shoulder = pts.get(RIGHT_SHOULDER)
    wrist = pts.get(RIGHT_WRIST)

    ref_leg = ref_len.get("leg", 0) or 1.0
    ref_arm = ref_len.get("arm", 0) or 1.0
    ref_torso = ref_len.get("torso", 0) or 1.0
    ground_y = ref_len.get("ground_y", None)

    # ---------------- 다리/발 계열 metric ----------------
    if metric_name in ["right_knee_angle", "right_knee_lift", "right_ankle_angle", "foot_strike_distance"]:
        if not (hip and knee and ankle):
            return False

        # 다리 길이: 평소 leg length의 0.7~1.3배 정도만 허용 (이전보다 빡셈)
        leg_len = ((hip[0] - ankle[0]) ** 2 + (hip[1] - ankle[1]) ** 2) ** 0.5
        if not (0.7 * ref_leg <= leg_len <= 1.3 * ref_leg):
            return False

        # y 순서: hip < knee < ankle (조금의 오차 허용)
        if not (hip[1] + 5 < knee[1] < ankle[1] + 5):
            return False

        # 발목이 “지면 아래로 너무 내려간” 프레임 배제
        if ground_y is not None and ankle[1] > ground_y + 20:
            # UI 쪽까지 쓸려 내려간 프레임 방지
            return False

    # ---------------- 팔/팔 스윙 계열 ----------------
    if metric_name in ["right_elbow_angle", "right_arm_swing"]:
        if not (shoulder and wrist):
            return False

        arm_len = ((shoulder[0] - wrist[0]) ** 2 + (shoulder[1] - wrist[1]) ** 2) ** 0.5
        if not (0.5 * ref_arm <= arm_len <= 1.5 * ref_arm):
            return False

    # ---------------- 상체/머리 계열 ----------------
    if metric_name in ["torso_lean", "head_tilt"]:
        if not (shoulder and hip):
            return False

        torso_h = abs(shoulder[1] - hip[1])
        if not (0.6 * ref_torso <= torso_h <= 1.4 * ref_torso):
            return False

    if metric_name == "head_tilt":
        ear_or_eye = pts.get(RIGHT_EAR) or pts.get(RIGHT_EYE)
        if not ear_or_eye:
            return False

    return True



# ---------- 3. 대표 프레임 선택 ----------

""" def choose_frame_index(metric: Dict[str, Any],
                       metric_name: str,
                       pose_data: Dict[str, Any],
                       ref_len: Dict[str, float]) -> int:
    values: List[float] = metric.get("per_frame", [])
    frame_indices: List[int] = metric.get("frame_indices", [])
    summary = metric.get("summary", {})
    mean = summary.get("mean")

    if not values or not frame_indices or mean is None:
        return 0

    # 1) 값 기준으로 "후보 index" 선택 (이전 로직)
    if metric_name == "right_knee_angle":
        if 115 <= mean <= 140:
            idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
        elif mean < 115:
            idx = values.index(min(values))
        else:
            idx = values.index(max(values))
    elif metric_name == "right_elbow_angle":
        if 70 <= mean <= 110:
            idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
        elif mean < 70:
            idx = values.index(min(values))
        else:
            idx = values.index(max(values))
    elif metric_name == "torso_lean":
        if 3 <= mean <= 10:
            idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
        elif mean < 3:
            idx = values.index(min(values))
        else:
            idx = values.index(max(values))
    elif metric_name == "head_tilt":
        if 5 <= mean <= 15:
            idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
        elif mean < 5:
            idx = values.index(min(values))
        else:
            idx = values.index(max(values))
    elif metric_name == "foot_strike_distance":
        if -5 <= mean <= 5:
            idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
        elif mean > 5:
            idx = values.index(max(values))
        else:
            idx = values.index(min(values))
    elif metric_name == "right_ankle_angle":
        if 90 <= mean <= 120:
            idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
        elif mean < 90:
            idx = values.index(min(values))
        else:
            idx = values.index(max(values))
    elif metric_name == "right_knee_lift":
        idx = values.index(max(values))  # 무릎이 가장 많이 올라간 순간
    elif metric_name == "right_arm_swing":
        idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
    else:
        idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))

    base_frame = frame_indices[idx]

    # 2) 품질 체크: 해당 프레임이 나쁜 프레임이면 주변에서 좋은 프레임을 찾음
    def pts_for_frame(fi: int) -> Dict[int, Tuple[int, int]]:
        frames = pose_data["frames"]
        fi = max(0, min(fi, len(frames) - 1))
        kps = frames[fi]["keypoints"]
        d = {}
        for kp in kps:
            d[kp["id"]] = (int(kp["x"]), int(kp["y"]))
        return d

    base_pts = pts_for_frame(base_frame)
    if is_good_frame_for_metric(metric_name, base_pts, ref_len):
        return base_frame

    # 주변 ±10 프레임 안에서 좋은 프레임 찾기
    for offset in range(1, 11):
        for cand in [base_frame - offset, base_frame + offset]:
            if cand < 0 or cand >= len(pose_data["frames"]):
                continue
            cand_pts = pts_for_frame(cand)
            if is_good_frame_for_metric(metric_name, cand_pts, ref_len):
                return cand

    # 그래도 없으면 그냥 기본 프레임 사용
    return base_frame """
def choose_frame_index(metric: Dict[str, Any],
                       metric_name: str,
                       pose_data: Dict[str, Any],
                       ref_len: Dict[str, float]) -> int:
    values: List[float] = metric.get("per_frame", [])
    frame_indices: List[int] = metric.get("frame_indices", [])
    summary = metric.get("summary", {})
    mean = summary.get("mean")

    if not values or not frame_indices:
        return 0

    # 1) 값 기준으로 "후보 index" 선택
    #    (여기 부분만 이전 것에서 조금 손 본 것임)

    if mean is None:
        idx = 0

    # ── (1) 각도 계열: 기존 로직 유지 ─────────────────
    elif metric_name == "right_knee_angle":
        # 115~140도: 양호 → 평균에 가까운 프레임
        if 115 <= mean <= 140:
            idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
        # 과도하게 많이 굽힌 쪽(작은 각도) 강조
        elif mean < 115:
            idx = values.index(min(values))
        # 과도하게 펴진 쪽(큰 각도) 강조
        else:
            idx = values.index(max(values))

    elif metric_name == "right_elbow_angle":
        # 70~110도: 양호
        if 70 <= mean <= 110:
            idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
        elif mean < 70:
            idx = values.index(min(values))
        else:
            idx = values.index(max(values))

    elif metric_name == "torso_lean":
        # 3~10도: 살짝 전경사 → 양호
        if 3 <= mean <= 10:
            idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
        # 너무 뒤로 젖힌 프레임(작은 값) 또는
        # 너무 많이 숙인 프레임(큰 값)을 대표로
        elif mean < 3:
            idx = values.index(min(values))
        else:
            idx = values.index(max(values))

    elif metric_name == "head_tilt":
        # 5~15도: 살짝 앞을 보는 자연스러운 각도
        if 5 <= mean <= 15:
            idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
        elif mean < 5:
            idx = values.index(min(values))
        else:
            idx = values.index(max(values))

    elif metric_name == "right_ankle_angle":
        # 90~120도: 보통 범위
        if 90 <= mean <= 120:
            idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))
        elif mean < 90:
            idx = values.index(min(values))
        else:
            idx = values.index(max(values))

    # ── (2) 거리/폭 계열: 개선된 부분 ─────────────────

    elif metric_name == "foot_strike_distance":
        # 착지 위치는 "가장 많이 앞/뒤로 나간 순간"이 설명하기 좋음
        # → 절대값이 가장 큰 프레임 선택
        idx = max(range(len(values)), key=lambda i: abs(values[i]))

    elif metric_name == "right_arm_swing":
        # 팔 스윙 폭도 "앞/뒤로 가장 멀리 간 순간"이 대표 프레임
        idx = max(range(len(values)), key=lambda i: abs(values[i]))

    # ── (3) 무릎 높이: 항상 최대값 프레임 ─────────────

    elif metric_name == "right_knee_lift":
        # 무릎을 가장 많이 든 순간
        idx = values.index(max(values))

    # ── (4) 기타 기본값: 평균에 가장 가까운 프레임 ────

    else:
        idx = min(range(len(values)), key=lambda i: abs(values[i] - mean))

    # values 리스트 안에서 골라진 index → 실제 frame index로 매핑
    base_frame = frame_indices[idx]

    # 2) 품질 체크: 해당 프레임이 나쁜 프레임이면 주변에서 좋은 프레임 찾음
    def pts_for_frame(fi: int) -> Dict[int, Tuple[int, int]]:
        frames = pose_data["frames"]
        fi = max(0, min(fi, len(frames) - 1))
        kps = frames[fi]["keypoints"]
        d = {}
        for kp in kps:
            d[kp["id"]] = (int(kp["x"]), int(kp["y"]))
        return d

    base_pts = pts_for_frame(base_frame)
    if is_good_frame_for_metric(metric_name, base_pts, ref_len):
        return base_frame

    # 주변 ±10 프레임 안에서 좋은 프레임 찾기
    for offset in range(1, 11):
        for cand in [base_frame - offset, base_frame + offset]:
            if cand < 0 or cand >= len(pose_data["frames"]):
                continue
            cand_pts = pts_for_frame(cand)
            if is_good_frame_for_metric(metric_name, cand_pts, ref_len):
                return cand

    # 그래도 없으면 그냥 기본 프레임 사용
    return base_frame


# ---------- 4. 비디오/포즈 읽기 & 그림 그리기 유틸 ----------

def get_frame_from_video(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("비디오 파일을 열 수 없습니다.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = max(0, min(frame_index, frame_count - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()

    back = 1
    while not ret and frame_index - back >= 0 and back < 10:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - back)
        ret, frame = cap.read()
        back += 1

    cap.release()

    if not ret:
        raise RuntimeError(f"{frame_index}번째 근처 프레임을 읽을 수 없습니다.")
    return frame


def get_points_for_frame(pose_data: Dict[str, Any], frame_index: int) -> Dict[int, Tuple[int, int]]:
    frames = pose_data["frames"]
    frame_index = max(0, min(frame_index, len(frames) - 1))
    kps = frames[frame_index]["keypoints"]
    pts = {}
    for kp in kps:
        pts[kp["id"]] = (int(kp["x"]), int(kp["y"]))
    return pts


def draw_line(img, a, b, color, thickness=2):
    cv2.line(img, a, b, color, thickness)
    cv2.circle(img, a, 4, color, -1)
    cv2.circle(img, b, 4, color, -1)


def put_title(img, title: str):
    # OpenCV → PIL 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size
    bar_height = 60
    draw.rectangle([0, 0, w, bar_height], fill=(0, 0, 0))

    # 한글 폰트 로드 (윈도우 기본: Malgun Gothic)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 28)

    draw.text((20, 15), title, font=font, fill=(255, 255, 255))

    # PIL → OpenCV 변환
    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)



# ---- metric별 드로잉 ----

def draw_knee(img, pts, value: float):
    hip = pts.get(RIGHT_HIP)
    knee = pts.get(RIGHT_KNEE)
    ankle = pts.get(RIGHT_ANKLE)
    if hip and knee and ankle:
        draw_line(img, hip, knee, (0, 255, 0))
        draw_line(img, knee, ankle, (0, 255, 0))
        cv2.putText(img, f"knee angle: {value:.1f} deg", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_elbow(img, pts, value: float):
    shoulder = pts.get(RIGHT_SHOULDER)
    elbow = pts.get(RIGHT_ELBOW)
    wrist = pts.get(RIGHT_WRIST)
    if shoulder and elbow and wrist:
        draw_line(img, shoulder, elbow, (255, 0, 0))
        draw_line(img, elbow, wrist, (255, 0, 0))
        cv2.putText(img, f"elbow angle: {value:.1f} deg", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_torso(img, pts, value: float):
    hip = pts.get(RIGHT_HIP)
    shoulder = pts.get(RIGHT_SHOULDER)
    if hip and shoulder:
        draw_line(img, hip, shoulder, (0, 0, 255))
        cv2.putText(img, f"torso lean: {value:.1f} deg", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_head(img, pts, value: float):
    shoulder = pts.get(RIGHT_SHOULDER)
    ear = pts.get(RIGHT_EAR) or pts.get(RIGHT_EYE)
    if shoulder and ear:
        draw_line(img, shoulder, ear, (0, 0, 255))
        cv2.putText(img, f"head tilt: {value:.1f} deg", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_foot_strike(img, pts, value: float):
    hip = pts.get(RIGHT_HIP)
    ankle = pts.get(RIGHT_ANKLE)
    if hip and ankle:
        draw_line(img, hip, ankle, (0, 255, 255))
        cv2.putText(img, f"foot strike: {value:.1f} %", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_ankle(img, pts, value: float):
    knee = pts.get(RIGHT_KNEE)
    ankle = pts.get(RIGHT_ANKLE)
    foot = pts.get(RIGHT_FOOT_INDEX)
    if knee and ankle and foot:
        draw_line(img, knee, ankle, (255, 255, 0))
        draw_line(img, ankle, foot, (255, 255, 0))
        cv2.putText(img, f"ankle angle: {value:.1f} deg", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_knee_lift(img, pts, value: float):
    """
    무릎 높이(각도 기반)를 시각화:
    - 어깨→엉덩이 (상체)
    - 엉덩이→무릎   (허벅지)
    두 선을 그리고, 정규화된 knee lift 값을 표시한다.
    """
    shoulder = pts.get(RIGHT_SHOULDER)
    hip = pts.get(RIGHT_HIP)
    knee = pts.get(RIGHT_KNEE)
    if not (shoulder and hip and knee):
        return

    # 상체 (초록)
    draw_line(img, hip, shoulder, (0, 255, 0))
    # 허벅지 (초록)
    draw_line(img, hip, knee, (0, 255, 0))

    text = f"knee lift: {value:.2f} (0~1)"
    cv2.putText(img, text, (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)



def draw_arm_swing(img, pts, value: float):
    shoulder = pts.get(RIGHT_SHOULDER)
    wrist = pts.get(RIGHT_WRIST)
    if shoulder and wrist:
        draw_line(img, shoulder, wrist, (255, 0, 255))
        cv2.putText(img, f"arm swing: {value:.1f} %", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# 판정 타이틀
def get_judgement_title(metric_name: str, mean: float, extra: Dict[str, Any]) -> str:
    if metric_name == "right_knee_angle":
        if 115 <= mean <= 140:
            return "무릎 각도 - 양호 판정"
        else:
            return "무릎 각도 - 개선 권장"
    if metric_name == "right_elbow_angle":
        if 70 <= mean <= 110:
            return "팔꿈치 각도 - 양호 판정"
        else:
            return "팔꿈치 각도 - 개선 권장"
    if metric_name == "torso_lean":
        if 3 <= mean <= 10:
            return "상체 기울기 - 양호 판정"
        else:
            return "상체 기울기 - 개선 여지"
    if metric_name == "head_tilt":
        if 5 <= mean <= 15:
            return "머리 기울기 - 양호 판정"
        else:
            return "머리 기울기 - 개선 권장"
    if metric_name == "foot_strike_distance":
        if -5 <= mean <= 5:
            return "발 착지 위치 - 양호 판정"
        elif mean > 15:
            return "발 착지 위치 - 개선 필요"
        else:
            return "발 착지 위치 - 개선 권장"
    if metric_name == "right_ankle_angle":
        if 90 <= mean <= 120:
            return "발목 각도 - 양호 판정"
        else:
            return "발목 각도 - 개선 권장"
    if metric_name == "right_knee_lift":
        if mean >= 0.15:
            return "무릎 높이 - 양호 판정"
        else:
            return "무릎 높이 - 개선 권장"
    if metric_name == "right_arm_swing":
        swing_range = extra.get("swing_range", 0)
        if 10 <= swing_range <= 25:
            return "팔 스윙 폭 - 양호 판정"
        elif swing_range > 40:
            return "팔 스윙 폭 - 개선 필요"
        else:
            return "팔 스윙 폭 - 개선 여지"
    return metric_name


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    analysis = load_json(ANALYSIS_JSON_PATH)
    pose_data = load_json(POSE_JSON_PATH)

    # 기준 길이 계산
    ref_len = compute_reference_lengths(pose_data)

    metric_names = [
        "right_knee_angle",
        "right_elbow_angle",
        "torso_lean",
        "head_tilt",
        "foot_strike_distance",
        "right_ankle_angle",
        "right_knee_lift",
        "right_arm_swing",
    ]

    for name in metric_names:
        metric = get_metric(analysis, name)
        if not metric:
            print(f"[WARN] metric {name} not found, skip")
            continue

        summary = metric.get("summary", {})
        mean = summary.get("mean")
        if mean is None:
            print(f"[WARN] metric {name} has no mean, skip")
            continue

        frame_index = choose_frame_index(metric, name, pose_data, ref_len)
        values = metric.get("per_frame", [])
        frame_indices = metric.get("frame_indices", [])

        if frame_index in frame_indices:
            idx_local = frame_indices.index(frame_index)
            value = values[idx_local]
        else:
            value = mean

        print(f"[INFO] {name}: frame {frame_index}, value={value:.3f}")

        frame = get_frame_from_video(VIDEO_PATH, frame_index)
        pts = get_points_for_frame(pose_data, frame_index)

        title = get_judgement_title(name, mean, metric)
        put_title(frame, title)

        if name == "right_knee_angle":
            draw_knee(frame, pts, value)
            out_name = "knee_angle.png"
        elif name == "right_elbow_angle":
            draw_elbow(frame, pts, value)
            out_name = "elbow_angle.png"
        elif name == "torso_lean":
            draw_torso(frame, pts, value)
            out_name = "torso_lean.png"
        elif name == "head_tilt":
            draw_head(frame, pts, value)
            out_name = "head_tilt.png"
        elif name == "foot_strike_distance":
            draw_foot_strike(frame, pts, value)
            out_name = "foot_strike.png"
        elif name == "right_ankle_angle":
            draw_ankle(frame, pts, value)
            out_name = "ankle_angle.png"
        elif name == "right_knee_lift":
            draw_knee_lift(frame, pts, value)
            out_name = "knee_lift.png"
        elif name == "right_arm_swing":
            draw_arm_swing(frame, pts, value)
            out_name = "arm_swing.png"
        else:
            out_name = f"{name}.png"

        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, frame)
        print(f"[INFO] saved {out_path}")


if __name__ == "__main__":
    main()
