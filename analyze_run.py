from analysis.metrics import analyze_pose_file
import json

POSE_JSON_PATH = "pose_data.json"
OUTPUT_PATH = "analysis_result.json"


def main():
    result = analyze_pose_file(POSE_JSON_PATH)

    # 터미널에 요약 출력
    print(f"총 프레임 수: {result['num_frames']}")
    for metric in result["metrics"]:
        name = metric["name"]
        summary = metric["summary"]
        print(f"\n=== {name} ===")
        print(f"  min : {summary['min']:.2f}" if summary["min"] is not None else "  min : None")
        print(f"  max : {summary['max']:.2f}" if summary["max"] is not None else "  max : None")
        print(f"  mean: {summary['mean']:.2f}" if summary["mean"] is not None else "  mean: None")

    # 전체 결과를 JSON 파일로 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n분석 결과가 {OUTPUT_PATH} 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()
