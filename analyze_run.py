# analyze_run.py
import json
from typing import Dict, Any, List
from analysis.metrics import (
    compute_knee_angles,
    compute_elbow_angles,
    compute_torso_lean,
    compute_head_tilt,
    compute_foot_strike_distance,
    compute_ankle_angle,
    compute_knee_lift,
    compute_arm_swing,
)

POSE_JSON_PATH = "pose_data.json"
METRICS_JSON_PATH = "metrics_result.json"


def load_pose_data(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_metric(name: str, metric: Dict[str, Any]) -> None:
    s = metric.get("summary", {})
    print(f"\n=== {name} ===")
    print(f"  min : {s.get('min'):.2f}" if s.get("min") is not None else "  min : -")
    print(f"  max : {s.get('max'):.2f}" if s.get("max") is not None else "  max : -")
    print(f"  mean: {s.get('mean'):.2f}" if s.get("mean") is not None else "  mean: -")


def main():
    pose_data = load_pose_data(POSE_JSON_PATH)
    frames: List[Dict[str, Any]] = pose_data.get("frames", [])

    print(f"총 프레임 수: {len(frames)}")

    metrics_list: List[Dict[str, Any]] = []

    m_knee   = compute_knee_angles(frames)
    m_elbow  = compute_elbow_angles(frames)
    m_torso  = compute_torso_lean(frames)
    m_head   = compute_head_tilt(frames)
    m_fs     = compute_foot_strike_distance(frames)
    m_ankle  = compute_ankle_angle(frames)
    m_knee_l = compute_knee_lift(frames)
    m_arm    = compute_arm_swing(frames)

    metrics_list.extend([
        m_knee, m_elbow, m_torso, m_head,
        m_fs, m_ankle, m_knee_l, m_arm
    ])

    # 콘솔 출력
    for m in metrics_list:
        print_metric(m["name"], m)

    # JSON 저장 (리포트용)
    out = {
        "source": POSE_JSON_PATH,
        "num_frames": len(frames),
        "metrics": metrics_list,
    }
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] 메트릭 요약을 '{METRICS_JSON_PATH}' 파일로 저장했습니다.")


if __name__ == "__main__":
    main()
