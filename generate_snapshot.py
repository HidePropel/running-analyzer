import json
import cv2
from typing import Dict, Tuple

from analysis.angles import angle_between_3points, angle_from_vertical

# 파일 경로 설정
POSE_JSON_PATH = "pose_data.json"
ANALYSIS_JSON_PATH = "analysis_result.json"
VIDEO_PATH = "run_side_2.mp4"
OUTPUT_IMAGE_PATH = "snapshot_head_mean.png"

# MediaPipe 인덱스 (오른쪽)
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28
RIGHT_FOOT_INDEX = 32
RIGHT_EYE = 5


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_head_mean_frame(analysis: Dict) -> int:
    """analysis_result.json에서 head_tilt의 mean에 가장 가까운 프레임 인덱스를 찾아 반환."""
    head_metric = None
    for m in analysis["metrics"]:
        if m["name"] == "head_tilt":
            head_metric = m
            break

    if head_metric is None:
        raise ValueError("head_tilt metric not found in analysis_result.json")

    values = head_metric["per_frame"]
    frame_indices = head_metric.get("frame_indices")
    mean_val = head_metric["summary"]["mean"]

    if not values or not frame_indices:
        raise ValueError("head_tilt metric has no values or frame_indices")

    # mean과 가장 가까운 값 찾기
    best_idx = min(range(len(values)), key=lambda i: abs(values[i] - mean_val))
    best_frame_index = frame_indices[best_idx]
    print(f"[INFO] head_tilt mean={mean_val:.2f}, 대표 프레임 index={best_frame_index}, 값={values[best_idx]:.2f}")
    return best_frame_index


def get_frame_from_video(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("비디오 파일을 열 수 없습니다.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"{frame_index} 번째 프레임을 읽을 수 없습니다.")

    return frame


def get_points_for_frame(pose_data: Dict, frame_index: int) -> Dict[int, Tuple[int, int]]:
    """pose_data.json에서 해당 frame_index의 keypoints를 딕셔너리로 변환."""
    frames = pose_data["frames"]
    if frame_index >= len(frames):
        raise IndexError("frame_index가 pose_data 프레임 수를 벗어났습니다.")

    kps = frames[frame_index]["keypoints"]
    points = {}
    for kp in kps:
        points[kp["id"]] = (int(kp["x"]), int(kp["y"]))
    return points


def draw_segment(img, a, b, color, thickness=2):
    cv2.line(img, a, b, color, thickness)
    cv2.circle(img, a, 4, color, -1)
    cv2.circle(img, b, 4, color, -1)


def main():
    # 1) 분석 결과와 pose 데이터 로드
    analysis = load_json(ANALYSIS_JSON_PATH)
    pose_data = load_json(POSE_JSON_PATH)

    # 2) head_tilt mean에 가장 가까운 프레임 찾기
    frame_index = find_head_mean_frame(analysis)

    # 3) 비디오에서 해당 프레임 이미지 가져오기
    frame = get_frame_from_video(VIDEO_PATH, frame_index)

    # 4) 해당 프레임의 keypoint 좌표 가져오기
    points = get_points_for_frame(pose_data, frame_index)

    # 필요한 포인트들
    hip = points.get(RIGHT_HIP)
    knee = points.get(RIGHT_KNEE)
    ankle = points.get(RIGHT_ANKLE)
    shoulder = points.get(RIGHT_SHOULDER)
    elbow = points.get(RIGHT_ELBOW)
    wrist = points.get(RIGHT_WRIST)
    eye = points.get(RIGHT_EYE)
    foot = points.get(RIGHT_FOOT_INDEX)

    # 5) 각도/높이 계산
    knee_angle = elbow_angle = torso_angle = head_angle = None
    knee_lift_norm = None

    if hip and knee and ankle:
        knee_angle = angle_between_3points(hip, knee, ankle)

    if shoulder and elbow and wrist:
        elbow_angle = angle_between_3points(shoulder, elbow, wrist)

    if hip and shoulder:
        torso_angle = angle_from_vertical(hip, shoulder)

    if shoulder and eye:
        head_angle = angle_from_vertical(shoulder, eye)

    if hip and knee and shoulder and ankle:
        raw = hip[1] - knee[1]
        body_len = abs(shoulder[1] - ankle[1])
        if body_len > 0:
            knee_lift_norm = raw / body_len

    # 6) 선·라인 그리기
    green = (0, 255, 0)
    blue = (255, 0, 0)
    red = (0, 0, 255)

    if hip and knee and ankle:
        draw_segment(frame, hip, knee, green)
        draw_segment(frame, knee, ankle, green)

    if shoulder and elbow and wrist:
        draw_segment(frame, shoulder, elbow, blue)
        draw_segment(frame, elbow, wrist, blue)

    if hip and shoulder:
        draw_segment(frame, hip, shoulder, red)

    if shoulder and eye:
        draw_segment(frame, shoulder, eye, red)

    # 7) 텍스트로 값 표시
    y0 = 40
    dy = 30
    x0 = 30

    def put(label, value, y):
        if value is None:
            text = f"{label}: N/A"
        else:
            text = f"{label}: {value:.1f}"
        cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    put("Knee angle (deg)", knee_angle, y0)
    put("Elbow angle (deg)", elbow_angle, y0 + dy)
    put("Torso lean (deg)", torso_angle, y0 + dy * 2)
    put("Head tilt (deg)", head_angle, y0 + dy * 3)
    if knee_lift_norm is not None:
        put("Knee lift (norm)", knee_lift_norm, y0 + dy * 4)

    # 8) 이미지 저장
    cv2.imwrite(OUTPUT_IMAGE_PATH, frame)
    print(f"[INFO] 스냅샷 저장 완료: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
