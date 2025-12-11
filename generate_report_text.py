# generate_report_text.py
import json
from typing import Dict, Any, List

METRICS_JSON_PATH = "metrics_result.json"
REPORT_TEXT_PATH = "running_report.txt"


def load_metrics(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(metrics: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for m in metrics:
        if m.get("name") == name:
            return m
    return {}


def s(v: float, unit: str = "") -> str:
    """숫자를 보기 좋게 포맷"""
    if v is None:
        return "-"
    if unit:
        return f"{v:.1f}{unit}"
    return f"{v:.1f}"


# --------------------------
#  각 항목별 평가 로직
# --------------------------

def eval_knee_angle(m: Dict[str, Any]) -> str:
    s_ = m.get("summary", {})
    mean = s_.get("mean")
    min_ = s_.get("min")
    max_ = s_.get("max")

    if mean is None:
        return "[무릎 각도]\n데이터가 충분하지 않아 분석할 수 없습니다.\n"

    # 기준: 115~140도 사이를 이상적 범위로 가정
    if 115 <= mean <= 140:
        grade = "양호 판정"
        msg = (
            "평균 무릎 각도가 115~140도 범위에 있어 추진력을 만들기에 좋은 각도입니다. "
            "과도한 overstride 없이 엉덩이 아래에서 밀어내는 동작에 도움이 됩니다."
        )
    elif mean < 115:
        grade = "개선 권장"
        msg = (
            "평균 무릎 각도가 다소 작은 편입니다. 무릎 접힘이 많으면 보폭이 짧아지고 "
            "지면 접촉 시간이 길어질 수 있습니다. 뒷다리 밀어내기를 조금 더 의식해보세요."
        )
    else:
        grade = "개선 권장"
        msg = (
            "평균 무릎 각도가 큰 편입니다. 무릎을 과도하게 접었다가 펴면 에너지 소모가 증가할 수 있습니다. "
            "앞으로 끌어올리기보다 뒤로 밀어낸다는 느낌을 가져보세요."
        )

    text = (
        "[무릎 각도]\n"
        f"- 판정: {grade}\n"
        f"- 수치: 평균 {s(mean, 'deg')}, 최소 {s(min_, 'deg')} ~ 최대 {s(max_, 'deg')}\n"
        f"- 코칭: {msg}\n"
    )
    return text


def eval_elbow_angle(m: Dict[str, Any]) -> str:
    s_ = m.get("summary", {})
    mean = s_.get("mean")
    min_ = s_.get("min")
    max_ = s_.get("max")

    if mean is None:
        return "[팔꿈치 각도]\n데이터가 부족하여 분석할 수 없습니다.\n"

    # 70~110도: 이상적인 런닝 팔꿈치 각도
    if 70 <= mean <= 110:
        grade = "양호 판정"
        msg = (
            "팔꿈치를 70~110도 사이로 잘 유지하고 있어 상체 흔들림을 줄이는 데 도움이 됩니다. "
            "지금처럼 몸 옆에 가깝게 두고 앞뒤로만 흔드는 느낌을 유지하면 좋습니다."
        )
    elif mean < 70:
        grade = "개선 권장"
        msg = (
            "팔꿈치 각도가 70도 이하로 많이 접히는 편입니다. 지나치게 접힌 팔은 "
            "어깨에 힘이 들어가고 상체 긴장을 유발할 수 있습니다. 손을 몸통에서 조금 더 멀리 두고 "
            "팔 길이를 10~20cm 정도 늘린다는 느낌으로 스윙해보세요."
        )
    else:
        grade = "개선 권장"
        msg = (
            "팔꿈치가 110도 이상으로 많이 펴지는 경향이 있습니다. 팔이 너무 펴지면 "
            "상체가 좌우로 흔들리기 쉬워집니다. 팔꿈치를 90도 근처에서 접은 상태로 "
            "리듬을 맞춰주는 걸 목표로 해보세요."
        )

    text = (
        "[팔꿈치 각도]\n"
        f"- 판정: {grade}\n"
        f"- 수치: 평균 {s(mean, 'deg')}, 최소 {s(min_, 'deg')} ~ 최대 {s(max_, 'deg')}\n"
        f"- 코칭: {msg}\n"
    )
    return text


def eval_torso_lean(m: Dict[str, Any]) -> str:
    s_ = m.get("summary", {})
    mean = s_.get("mean")
    min_ = s_.get("min")
    max_ = s_.get("max")

    if mean is None:
        return "[상체 기울기]\n데이터가 부족하여 분석할 수 없습니다.\n"

    # 기준: +방향 = 앞으로 숙임, 3~10도 정도를 이상 범위로 설정
    if 3 <= mean <= 10:
        grade = "양호 판정"
        msg = (
            "상체가 3~10도 정도로 자연스럽게 앞으로 기울어져 있어, "
            "중심이 발 앞쪽으로 잘 이동하는 자세입니다."
        )
    elif mean < 3 and mean > -3:
        grade = "개선 여지"
        msg = (
            "상체 기울기가 거의 0도에 가까워 약간은 '세워서' 뛰는 형태입니다. "
            "발목에서 살짝 앞으로 쓰러지는 느낌(3~5도)을 만들어주면 추진력을 더 얻을 수 있습니다."
        )
    elif mean <= -3:
        grade = "개선 필요"
        msg = (
            "상체가 뒤로 젖혀진 구간이 많습니다. 상체가 뒤로 가면 보폭은 커져도 "
            "브레이크가 걸리는 착지가 되기 쉽습니다. 배꼽을 살짝 앞으로 내민다는 느낌으로 "
            "중심을 앞으로 가져와 보세요."
        )
    else:
        grade = "개선 권장"
        msg = (
            "상체 기울기가 10도 이상으로 많이 숙여진 편입니다. 과한 전경사는 허리와 햄스트링 부담을 늘립니다. "
            "머리-어깨-엉덩이가 일직선이 되도록 '발목에서 쓰러진다'는 이미지를 가져보세요."
        )

    text = (
        "[상체 기울기]\n"
        f"- 판정: {grade}\n"
        f"- 수치: 평균 {s(mean, 'deg')}, 최소 {s(min_, 'deg')} ~ 최대 {s(max_, 'deg')}\n"
        f"- 코칭: {msg}\n"
    )
    return text


def eval_head_tilt(m: Dict[str, Any]) -> str:
    s_ = m.get("summary", {})
    mean = s_.get("mean")
    min_ = s_.get("min")
    max_ = s_.get("max")

    if mean is None:
        return "[머리 기울기]\n데이터가 부족하여 분석할 수 없습니다.\n"

    # 기준: 5~15도 정도를 '자연스러운 약간의 숙임'으로 설정
    if 5 <= mean <= 15:
        grade = "양호 판정"
        msg = (
            "머리가 5~15도 정도로 자연스럽게 앞으로 기울어져 있어, "
            "목과 승모근에 과도한 긴장 없이 시야를 확보하는 자세입니다."
        )
    elif mean < 5:
        grade = "개선 권장"
        msg = (
            "머리 기울기가 거의 0도에 가깝습니다. 너무 정면(또는 위쪽)을 바라보면 목 뒤가 뻐근해질 수 있습니다. "
            "10~15m 앞 지면을 바라본다는 느낌으로 시선을 살짝 낮춰보세요."
        )
    else:
        grade = "개선 필요"
        msg = (
            "머리 기울기가 15도 이상으로 많이 숙여진 편입니다. 시선이 너무 아래로 떨어지면 "
            "호흡이 답답해지고 상체가 과하게 말릴 수 있습니다. "
            "가슴을 열고, 시선을 조금 앞쪽으로 들어 올려보세요."
        )

    text = (
        "[머리 기울기]\n"
        f"- 판정: {grade}\n"
        f"- 수치: 평균 {s(mean, 'deg')}, 최소 {s(min_, 'deg')} ~ 최대 {s(max_, 'deg')}\n"
        f"- 코칭: {msg}\n"
    )
    return text


def eval_foot_strike(m: Dict[str, Any]) -> str:
    s_ = m.get("summary", {})
    mean = s_.get("mean")
    min_ = s_.get("min")
    max_ = s_.get("max")

    if mean is None:
        return "[발 착지 위치]\n데이터가 부족하여 분석할 수 없습니다.\n"

    # 값: 엉덩이-발 사이 x거리 / 다리길이 * 100 (%)
    # -5% ~ +5% : 거의 몸 아래
    if -5 <= mean <= 5:
        grade = "양호 판정"
        msg = (
            "발이 몸의 중심 바로 아래쪽에 가깝게 착지하고 있습니다. "
            "브레이크를 최소화하면서 추진력을 얻는 효율적인 착지입니다."
        )
    elif mean > 5:
        grade = "개선 권장"
        msg = (
            f"평균 발 착지가 엉덩이보다 약 {mean:.1f}% 앞에서 일어납니다. "
            "약간의 overstride 경향이 있어, 착지 순간 발이 몸 아래로 더 들어오도록 "
            "케이던스를 5~10rpm 정도 올리는 연습을 추천합니다."
        )
    else:  # mean < -5
        grade = "개선 여지"
        msg = (
            "발이 몸 중심보다 약간 뒤쪽에서 닿는 구간이 있습니다. "
            "단거리 스프린트 상황이 아니라면, 너무 뒤에서 차는 동작은 "
            "과한 뒤꿈치 킥이 될 수 있으니 중립적인 착지를 유지하는지 확인해보세요."
        )

    text = (
        "[발 착지 위치]\n"
        f"- 판정: {grade}\n"
        f"- 수치: 평균 {s(mean, '%')}, 최소 {s(min_, '%')} ~ 최대 {s(max_, '%')}\n"
        f"- 코칭: {msg}\n"
    )
    return text


def eval_ankle_angle(m: Dict[str, Any]) -> str:
    s_ = m.get("summary", {})
    mean = s_.get("mean")
    min_ = s_.get("min")
    max_ = s_.get("max")

    if mean is None:
        return "[발목 각도]\n데이터가 부족하여 분석할 수 없습니다.\n"

    # 대략 90~120도 정도를 자연스러운 범위로 가정
    if 90 <= mean <= 120:
        grade = "양호 판정"
        msg = (
            "발목 각도가 90~120도 사이에 있어 무릎과 함께 자연스럽게 충격을 흡수하고 있습니다. "
            "지금처럼 발목을 적당히 고정하되, 힘을 빼고 리듬을 유지하면 좋습니다."
        )
    elif mean < 90:
        grade = "개선 권장"
        msg = (
            "발목 각도가 90도 이하로 많이 접히는 구간이 있습니다. "
            "과도한 dorsiflexion(발등을 당기는 동작)은 종아리와 앞정강이 긴장을 유발할 수 있으니, "
            "발목 힘을 조금 빼고 자연스럽게 떨어뜨리는 느낌을 가져보세요."
        )
    else:
        grade = "개선 권장"
        msg = (
            "발목이 120도 이상으로 많이 펴지는 경향이 있습니다. "
            "착지 순간 발목이 너무 뻣뻣하면 충격이 그대로 무릎과 허리로 올라갈 수 있습니다. "
            "발목을 살짝 탄성 있게 써 준다는 감각을 가져보세요."
        )

    text = (
        "[발목 각도]\n"
        f"- 판정: {grade}\n"
        f"- 수치: 평균 {s(mean, 'deg')}, 최소 {s(min_, 'deg')} ~ 최대 {s(max_, 'deg')}\n"
        f"- 코칭: {msg}\n"
    )
    return text


def eval_knee_lift(m: Dict[str, Any]) -> str:
    s_ = m.get("summary", {})
    mean = s_.get("mean")

    if mean is None:
        return "[무릎 높이]\n데이터가 부족하여 분석할 수 없습니다.\n"

    # 0.0~1.0 정규화 값 (0: 거의 안 듦, 1: 허벅지 수평 이상)
    if 0.3 <= mean <= 0.6:
        grade = "양호 판정"
        msg = (
            "무릎을 과하지도, 부족하지도 않게 적당한 높이까지 끌어올리고 있습니다. "
            "현재 케이던스와 페이스에서는 효율적인 무릎 높이입니다."
        )
    elif mean < 0.3:
        grade = "개선 권장"
        msg = (
            "무릎 높이가 다소 낮은 편입니다. 보폭이 짧아지거나 지면에 끌리는 느낌이 날 수 있습니다. "
            "가벼운 스킵 드릴 또는 짧은 스트라이드(가속주)를 통해 허벅지 들어올리는 감각을 연습해보세요."
        )
    else:
        grade = "개선 여지"
        msg = (
            "무릎을 꽤 높게 끌어올리는 편입니다. 단거리 스피드 훈련에는 좋지만, "
            "롱런/마라톤 페이스에서는 에너지 소모가 커질 수 있습니다. "
            "평상시 러닝에서는 무릎을 60~70% 정도만 든다는 느낌으로 조절해도 좋습니다."
        )

    text = (
        "[무릎 높이]\n"
        f"- 판정: {grade}\n"
        f"- 수치: 평균 {s(mean, '')} (0.0~1.0 정규화 값)\n"
        f"- 코칭: {msg}\n"
    )
    return text


def eval_arm_swing(m: Dict[str, Any]) -> str:
    s_ = m.get("summary", {})
    mean = s_.get("mean")
    min_ = s_.get("min")
    max_ = s_.get("max")
    swing_range = m.get("swing_range")  # % 단위

    if mean is None:
        return "[팔 스윙 폭]\n데이터가 부족하여 분석할 수 없습니다.\n"

    # swing_range: 어깨 기준 손목 x 위치의 앞뒤 총 범위 (%)
    if swing_range is None:
        swing_range = 0.0

    # 대략 20~40% 사이를 적당한 팔 스윙 폭으로 가정
    if 20 <= swing_range <= 40:
        grade = "양호 판정"
        msg = (
            "팔 스윙 폭이 20~40% 범위로, 상체 균형을 잡기에 좋은 수준입니다. "
            "몸통 회전 없이 앞뒤로만 리듬을 유지하면 좋습니다."
        )
    elif swing_range < 20:
        grade = "개선 권장"
        msg = (
            "팔 스윙 폭이 다소 작은 편입니다. 팔을 거의 움직이지 않으면 보폭과 리듬을 만들기 어려울 수 있습니다. "
            "팔꿈치는 그대로 두고, 손이 가슴선 앞까지는 살짝 올라오도록 스윙 범위를 늘려보세요."
        )
    else:
        grade = "개선 필요"
        msg = (
            f"팔 스윙 폭이 약 {s(swing_range, '%')}로 큰 편입니다. "
            "팔이 크게 흔들리면 상체가 좌우로 함께 흔들리면서 에너지가 분산됩니다. "
            "손이 몸 중앙선을 지나치지 않도록, 몸 가까이에 두고 위아래 방향 스윙을 의식해보세요."
        )

    text = (
        "[팔 스윙 폭]\n"
        f"- 판정: {grade}\n"
        f"- 수치: 평균 {s(mean, '%')}, 범위 {s(min_, '%')} ~ {s(max_, '%')}, "
        f"앞뒤 총 스윙 폭 약 {s(swing_range, '%')}\n"
        f"- 코칭: {msg}\n"
    )
    return text


# --------------------------
#  전체 리포트 생성
# --------------------------

def generate_report() -> str:
    data = load_metrics(METRICS_JSON_PATH)
    metrics = data.get("metrics", [])

    knee   = get_metric(metrics, "right_knee_angle")
    elbow  = get_metric(metrics, "right_elbow_angle")
    torso  = get_metric(metrics, "torso_lean")
    head   = get_metric(metrics, "head_tilt")
    fs     = get_metric(metrics, "foot_strike_distance")
    ankle  = get_metric(metrics, "right_ankle_angle")
    knee_l = get_metric(metrics, "right_knee_lift")
    arm    = get_metric(metrics, "right_arm_swing")

    sections = [
        eval_knee_angle(knee),
        eval_knee_lift(knee_l),
        eval_foot_strike(fs),
        eval_ankle_angle(ankle),
        eval_elbow_angle(elbow),
        eval_arm_swing(arm),
        eval_torso_lean(torso),
        eval_head_tilt(head),
    ]

    header = (
        "============================\n"
        " 러닝 자세 분석 리포트 (Beta)\n"
        "============================\n\n"
        f"- 분석 소스: {data.get('source', 'N/A')}\n"
        f"- 분석 프레임 수: {data.get('num_frames', 0)}\n\n"
        "[전체 요약]\n"
        "각 항목은 현재 러닝 자세를 기준으로 강점/개선 포인트를 제안한 것입니다. "
        "훈련 수준, 페이스, 거리(마라톤/조깅 등)에 따라 최적값은 달라질 수 있으니, "
        "참고 기준으로 활용해주세요.\n\n"
        "----------------------------------------\n\n"
    )

    report_text = header + "\n\n----------------------------------------\n\n".join(sections)
    return report_text


def main():
    text = generate_report()
    with open(REPORT_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] 러닝 자세 리포트를 '{REPORT_TEXT_PATH}' 파일로 저장했습니다.")


if __name__ == "__main__":
    main()
