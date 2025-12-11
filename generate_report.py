import json
from typing import Dict, Any, Optional, List

ANALYSIS_JSON_PATH = "analysis_result.json"
OUTPUT_REPORT_PATH = "running_report.txt"


def load_analysis(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(analysis: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for m in analysis.get("metrics", []):
        if m.get("name") == name:
            return m
    return None


def fmt(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def eval_knee(metric: Dict[str, Any]) -> Dict[str, str]:
    mean = metric["summary"]["mean"]
    if mean is None:
        return {"status": "⚠ 데이터 부족", "message": "무릎 각도를 계산할 수 있는 프레임이 충분하지 않습니다."}

    msg = f"평균 무릎 각도는 {mean:.1f}° 입니다. "
    if 115 <= mean <= 140:
        msg += "일반적인 러닝에서 이상적인 범위에 해당하며, 추진력과 충격 흡수의 균형이 잘 잡혀 있습니다."
        status = "✅ 양호"
    elif mean < 115:
        msg += "무릎을 비교적 많이 굽히는 편으로, 스피드 러닝이나 스프린트 성향에 가깝습니다."
        status = "⚠ 참고"
    else:
        msg += "무릎 굽힘이 다소 적은 편으로, 착지 시 충격이 다리 쪽으로 더 전달될 수 있습니다. 약간 더 무릎을 자연스럽게 굽히는 느낌을 가져가면 좋습니다."
        status = "⚠ 개선 권장"

    return {"status": status, "message": msg}


def eval_elbow(metric: Dict[str, Any]) -> Dict[str, str]:
    mean = metric["summary"]["mean"]
    if mean is None:
        return {"status": "⚠ 데이터 부족", "message": "팔꿈치 각도를 계산할 수 있는 프레임이 충분하지 않습니다."}

    msg = f"평균 팔꿈치 각도는 {mean:.1f}° 입니다. "
    if 70 <= mean <= 110:
        msg += "팔 각도가 매우 이상적인 범위로, 상체 회전이 과하지 않으면서도 효율적인 팔 스윙을 하고 있습니다."
        status = "✅ 양호"
    elif mean < 70:
        msg += "팔꿈치를 너무 많이 접는 경향이 있어, 어깨와 목 주변에 긴장을 줄 수 있습니다."
        status = "⚠ 개선 권장"
    else:
        msg += "팔꿈치를 다소 펴는 경향이 있어, 상체가 좌우로 흔들리기 쉬운 패턴입니다. 팔꿈치를 90° 안팎으로 유지하는 느낌을 가져가면 좋습니다."
        status = "⚠ 개선 권장"

    return {"status": status, "message": msg}


def eval_torso(metric: Dict[str, Any]) -> Dict[str, str]:
    mean = metric["summary"]["mean"]
    if mean is None:
        return {"status": "⚠ 데이터 부족", "message": "상체 기울기를 계산할 수 있는 프레임이 충분하지 않습니다."}

    msg = f"평균 상체 기울기는 {mean:.1f}° 입니다. "
    if 3 <= mean <= 10:
        msg += "가볍게 앞으로 기울어진 이상적인 자세로, 자연스러운 추진력을 잘 활용하고 있습니다."
        status = "✅ 양호"
    elif -3 <= mean < 3:
        msg += "상체가 거의 수직에 가깝습니다. 약간(3~7°)만 더 앞으로 기울어지면 지면 반발력을 활용하는 데 도움이 됩니다."
        status = "⚠ 개선 여지"
    elif mean > 10:
        msg += "상체가 다소 많이 앞으로 숙여져 있어, 하체에 부담이 갈 수 있습니다. 가슴을 조금만 더 세우는 느낌을 가져가면 좋습니다."
        status = "⚠ 개선 권장"
    else:
        msg += "상체가 뒤로 젖혀지는 구간이 많아, 러닝 효율이 떨어질 수 있습니다."
        status = "⚠ 개선 권장"

    return {"status": status, "message": msg}


def eval_head(metric: Dict[str, Any]) -> Dict[str, str]:
    mean = metric["summary"]["mean"]
    if mean is None:
        return {"status": "⚠ 데이터 부족", "message": "머리 기울기를 계산할 수 있는 프레임이 충분하지 않습니다."}

    msg = f"평균 머리 기울기는 {mean:.1f}° 입니다. "
    if 5 <= mean <= 15:
        msg += "머리가 살짝 앞으로 기울어진 정상적인 범위로, 목과 어깨에 과도한 부담을 주지 않는 자세입니다."
        status = "✅ 양호"
    elif mean < 5:
        msg += "머리가 거의 수직에 가깝거나 약간 뒤로 젖혀지는 경향이 있습니다. 시선은 수평, 턱은 살짝 안으로 당기는 느낌이 좋습니다."
        status = "⚠ 참고"
    else:
        msg += "머리가 다소 앞으로 빠지는 경향이 있어, 장거리 러닝 시 목과 승모근이 쉽게 피로해질 수 있습니다. 귀가 어깨 위에 위치하도록 가볍게 턱을 당기는 느낌을 가져가 보세요."
        status = "⚠ 개선 권장"

    return {"status": status, "message": msg}


def eval_foot_strike(metric: Dict[str, Any]) -> Dict[str, str]:
    mean = metric["summary"]["mean"]
    if mean is None:
        return {"status": "⚠ 데이터 부족", "message": "발 착지를 계산할 수 없습니다."}

    msg = f"발 착지 위치 평균은 {mean:.1f}% 입니다. "

    # 기준:
    # -5% ~ +5%  → 몸 바로 아래
    # +5% ~ +15% → 약간 앞 (경미한 오버스트라이딩)
    # +15% 이상 → 명확한 오버스트라이딩

    if -5 <= mean <= 5:
        msg += "발이 몸의 중심 바로 아래에 착지하고 있습니다. 매우 이상적입니다."
        status = "✅ 양호"
    elif 5 < mean <= 15:
        msg += "발이 몸보다 약간 앞에서 착지합니다. 케이던스를 +5 정도만 높여도 개선될 수 있습니다."
        status = "⚠ 개선 권장"
    elif mean > 15:
        msg += "명확한 오버스트라이딩 패턴입니다. 보폭을 줄이거나 케이던스를 증가시키는 것이 필요합니다."
        status = "⚠ 개선 필요"
    else:  # mean < -5
        msg += "발이 몸보다 뒤쪽에서 착지하는 경향이 있습니다. 비정상적이므로 영상 구도를 함께 점검해 보세요."
        status = "⚠ 참고"

    return {"status": status, "message": msg}



def eval_ankle(metric: Dict[str, Any]) -> Dict[str, str]:
    mean = metric["summary"]["mean"]
    if mean is None:
        return {"status": "⚠ 데이터 부족", "message": "발목 각도를 계산할 수 있는 프레임이 충분하지 않습니다."}

    msg = f"평균 발목 각도는 {mean:.1f}° 입니다. "
    if 90 <= mean <= 120:
        msg += "일반적인 러닝에서 이상적인 발목 각도로, 과도한 충격이나 불필요한 힘 손실이 적은 편입니다."
        status = "✅ 양호"
    elif mean < 90:
        msg += "발목이 많이 젖혀지는(dorsiflexion) 경향이 있어, 착지 충격이 커질 수 있습니다."
        status = "⚠ 개선 권장"
    else:
        msg += "발목이 다소 많이 내려가는(plantarflexion) 경향이 있어, 과도한 포어풋/미드풋 착지일 수 있습니다."
        status = "⚠ 참고"

    return {"status": status, "message": msg}


def eval_knee_lift(metric):
    max_v = metric["summary"]["max"]
    if max_v is None:
        return {"status": "⚠ 데이터 부족", "message": "무릎 높이를 평가할 수 없습니다."}

    if max_v >= 0.20:
        status = "✅ 양호"
        msg = f"최대 무릎 높이 지표는 {max_v:.2f}로, 허벅지를 충분히 들어주는 좋은 런닝 폼입니다."
    elif max_v >= 0.10:
        status = "⚠ 참고"
        msg = f"최대 무릎 높이 지표는 {max_v:.2f}로, 보통 수준입니다. 템포런이나 스킵 드릴을 통해 약간 더 knee drive를 키우면 좋습니다."
    else:
        status = "⚠ 개선 권장"
        msg = f"최대 무릎 높이 지표는 {max_v:.2f}로, 무릎을 거의 들지 않는 패턴입니다. A-스킵, 니 하이 드릴로 무릎 들기 감각을 연습해보세요."

    return {"status": status, "message": msg}



def eval_arm_swing(metric: Dict[str, Any]) -> Dict[str, str]:
    summary = metric["summary"]
    mean = summary["mean"]
    swing_range = metric.get("swing_range")

    if mean is None or swing_range is None:
        return {"status": "⚠ 데이터 부족", "message": "팔 스윙을 평가할 수 있는 데이터가 부족합니다."}

    msg = f"정규화된 팔 스윙 폭은 약 {swing_range:.1f}% 입니다. "

    if swing_range < 10:
        msg += "팔 스윙이 매우 작은 편입니다. 상체가 경직되어 있을 수 있으며, 러닝 리듬을 만들기 어려울 수 있습니다."
        status = "⚠ 개선 권장"
    elif 10 <= swing_range <= 25:
        msg += "이상적인 팔 스윙 폭으로, 상체 리듬과 추진력이 잘 만들어지고 있습니다."
        status = "✅ 양호"
    elif 25 < swing_range <= 40:
        msg += "팔 스윙이 다소 큰 편입니다. 상체가 약간 더 열리고 있을 수 있습니다. 팔꿈치를 몸 가까이 두면 안정적인 리듬을 만들 수 있습니다."
        status = "⚠ 참고"
    else:
        msg += "팔 스윙 폭이 매우 큰 편입니다. 이는 상체의 좌우 흔들림을 유발할 수 있으며, 에너지 손실이 발생할 수 있습니다. 팔은 위아래 방향으로 흔드는 느낌을 가져가면 개선됩니다."
        status = "⚠ 개선 필요"

    return {"status": status, "message": msg}



def build_report(analysis: Dict[str, Any]) -> str:
    num_frames = analysis.get("num_frames", 0)

    knee = get_metric(analysis, "right_knee_angle")
    elbow = get_metric(analysis, "right_elbow_angle")
    torso = get_metric(analysis, "torso_lean")
    head = get_metric(analysis, "head_tilt")
    foot = get_metric(analysis, "foot_strike_distance")
    ankle = get_metric(analysis, "right_ankle_angle")
    knee_lift = get_metric(analysis, "right_knee_lift")
    arm = get_metric(analysis, "right_arm_swing")

    sections: List[str] = []
    summary_points_good = []
    summary_points_improve = []

    sections.append(f"=== 러닝 자세 분석 리포트 ===\n총 분석 프레임 수: {num_frames}\n")

    # 1) 무릎
    if knee:
        r = eval_knee(knee)
        sections.append("[무릎 각도]\n" + r["status"] + " - " + r["message"] + "\n")
        if r["status"].startswith("✅"):
            summary_points_good.append("무릎 각도가 이상적인 범위에 있어, 추진력과 충격 흡수의 균형이 좋습니다.")
        else:
            summary_points_improve.append("무릎 굽힘 패턴을 점검해 보시면 러닝 효율 개선에 도움이 될 수 있습니다.")

    # 2) 팔꿈치
    if elbow:
        r = eval_elbow(elbow)
        sections.append("[팔꿈치 각도]\n" + r["status"] + " - " + r["message"] + "\n")
        if r["status"].startswith("✅"):
            summary_points_good.append("팔꿈치 각도가 안정적이라 상체 흔들림이 크지 않습니다.")
        else:
            summary_points_improve.append("팔꿈치 각도를 90° 안팎으로 유지하는 연습이 필요합니다.")

    # 3) 상체
    if torso:
        r = eval_torso(torso)
        sections.append("[상체 기울기]\n" + r["status"] + " - " + r["message"] + "\n")
        if r["status"].startswith("✅"):
            summary_points_good.append("상체 기울기가 자연스러워 지면 반발력을 잘 활용하고 있습니다.")
        else:
            summary_points_improve.append("상체를 약간 더 앞으로/뒤로 조정하면 효율이 좋아질 수 있습니다.")

    # 4) 머리
    if head:
        r = eval_head(head)
        sections.append("[머리 기울기]\n" + r["status"] + " - " + r["message"] + "\n")
        if r["status"].startswith("✅"):
            summary_points_good.append("머리 위치가 안정적이라 목과 어깨에 과부하가 적은 자세입니다.")
        else:
            summary_points_improve.append("머리가 너무 앞으로/뒤로 가지 않도록 귀와 어깨가 일직선에 오도록 신경 써 보세요.")

    # 5) 발 착지
    if foot:
        r = eval_foot_strike(foot)
        sections.append("[발 착지 위치]\n" + r["status"] + " - " + r["message"] + "\n")
        if r["status"].startswith("✅"):
            summary_points_good.append("발이 몸 아래에 잘 떨어져 overstriding 위험이 낮습니다.")
        else:
            summary_points_improve.append("발이 몸 앞에서 떨어지는 경향이 있어, 케이던스와 보폭 조절이 필요합니다.")

    # 6) 발목
    if ankle:
        r = eval_ankle(ankle)
        sections.append("[발목 각도]\n" + r["status"] + " - " + r["message"] + "\n")
        if r["status"].startswith("✅"):
            summary_points_good.append("발목 각도가 이상적이라 착지 충격이 과도하지 않은 편입니다.")
        else:
            summary_points_improve.append("발목 각도를 조금 조정하면 착지 충격을 더 줄일 수 있습니다.")

    # 7) 무릎 높이
    if knee_lift:
        r = eval_knee_lift(knee_lift)
        sections.append("[무릎 높이]\n" + r["status"] + " - " + r["message"] + "\n")
        if r["status"].startswith("✅"):
            summary_points_good.append("무릎 드라이브가 좋아 추진력을 잘 만들고 있습니다.")
        else:
            summary_points_improve.append("무릎 드라이브 향상을 위한 드릴(A-스킵, 니 하이 등)을 시도해 보시면 좋습니다.")

    # 8) 팔 스윙
    if arm:
        r = eval_arm_swing(arm)
        sections.append("[팔 스윙 폭]\n" + r["status"] + " - " + r["message"] + "\n")
        if r["status"].startswith("✅"):
            summary_points_good.append("팔 스윙 폭이 적절해 러닝 리듬을 잘 유지하고 있습니다.")
        else:
            summary_points_improve.append("팔 스윙이 너무 작거나 크지 않도록, 몸통 가까이에서 앞뒤로 흔드는 느낌을 가져가 보세요.")

    # 요약 섹션
    summary_lines = ["=== 요약 ==="]

    if summary_points_good:
        summary_lines.append("\n[강점]")
        for s in summary_points_good[:3]:
            summary_lines.append(f"- {s}")

    if summary_points_improve:
        summary_lines.append("\n[개선 포인트 (우선 순위)]")
        for s in summary_points_improve[:3]:
            summary_lines.append(f"- {s}")

    summary_text = "\n".join(summary_lines)
    full_report = summary_text + "\n\n" + "\n".join(sections)

    return full_report


def main():
    analysis = load_analysis(ANALYSIS_JSON_PATH)
    report = build_report(analysis)

    print(report)

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[INFO] 러닝 리포트가 {OUTPUT_REPORT_PATH} 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()
