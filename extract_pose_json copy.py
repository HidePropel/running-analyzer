import cv2
import mediapipe as mp
import json
import numpy as np

mp_pose = mp.solutions.pose

VIDEO_PATH = "run_side_2.mp4"
OUTPUT_JSON = "pose_data.json"

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)  # 초당 프레임 수

if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

frames_data = []  # 여기에 프레임별 포즈 정보 저장

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 시간(초) 계산
        t = frame_idx / fps

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        keypoints = []

        if results.pose_landmarks:
            h, w, _ = frame.shape
            for i, lm in enumerate(results.pose_landmarks.landmark):
                # 실제 픽셀 좌표로 변환
                x = lm.x * w
                y = lm.y * h
                z = lm.z     # z는 상대값
                visibility = lm.visibility

                keypoints.append({
                    "id": i,         # 점 번호 (0~32)
                    "x": x,
                    "y": y,
                    "z": float(z),
                    "visibility": float(visibility)
                })

        frames_data.append({
            "t": t,
            "keypoints": keypoints
        })

        frame_idx += 1

cap.release()

# JSON 파일로 저장
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"frames": frames_data}, f, ensure_ascii=False, indent=2)

print(f"총 {len(frames_data)} 프레임 저장 완료: {OUTPUT_JSON}")
