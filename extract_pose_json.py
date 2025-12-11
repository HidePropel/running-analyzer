import cv2
import json
import mediapipe as mp

VIDEO_PATH = "run_side.mp4"
OUTPUT_JSON = "pose_data.json"

mp_pose = mp.solutions.pose

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("비디오를 열 수 없습니다.")

    frames = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape  # ★ 원본 해상도

            # Mediapipe는 0~1 좌표로 줌 → 이걸 다시 픽셀로 되돌리기
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            keypoints = []
            if result.pose_landmarks:
                for i, lm in enumerate(result.pose_landmarks.landmark):
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)
                    keypoints.append({
                        "id": i,
                        "x": x_px,
                        "y": y_px,
                    })

            frames.append({
                "frame_index": frame_idx,
                "keypoints": keypoints,
            })
            frame_idx += 1

    cap.release()

    data = {
        "video_path": VIDEO_PATH,
        "frame_width": int(w),
        "frame_height": int(h),
        "frames": frames,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 저장 완료: {OUTPUT_JSON}, 총 프레임 수: {len(frames)}")

if __name__ == "__main__":
    main()
