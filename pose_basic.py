import cv2
import mediapipe as mp

# MediaPipe Pose 모듈 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 비디오 파일 경로
VIDEO_PATH = "run_side_2.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

# Pose 모델 사용
with mp_pose.Pose(
    static_image_mode=False,       # 동영상이므로 False
    model_complexity=1,            # 0~2 (복잡도, 처음엔 1)
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝

        # BGR → RGB 변환 (MediaPipe는 RGB 기준)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 성능 향상을 위해 writeable=False 설정
        image.flags.writeable = False
        results = pose.process(image)

        # 다시 writeable=True로 바꾸고 BGR로 되돌리기
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 포즈가 인식되면 스켈레톤 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("Pose Test", image)

        # q를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
