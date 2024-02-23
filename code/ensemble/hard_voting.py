import json
from glob import glob

prediction_files = glob("./results/*.json")
# 각 모델의 예측 결과를 담을 딕셔너리를 초기화합니다.
ensemble_predictions = {}

# 각 prediction 파일을 읽어와서 ensemble_predictions에 합산합니다.
for file_name in prediction_files:
    with open(file_name, "r") as file:
        predictions = json.load(file)
        # 파일 이름에 "first"가 포함되는 경우 가중치를 +1 곱합니다.
        weight = 2.718 if "third" in file_name else 1
        for key, value in predictions.items():
            if key not in ensemble_predictions:
                ensemble_predictions[key] = []
            # 가중치를 적용하여 예측 결과를 추가합니다.
            ensemble_predictions[key].extend([(prediction, weight) for prediction in value])

# Hard voting을 사용하여 각 예측 결과에 대한 투표를 진행합니다.
# 각 예측 결과는 "text" 키를 기준으로 투표를 합니다.
final_predictions = {}
for key, value in ensemble_predictions.items():
    text_counts = {}
    for prediction, weight in value:
        text = prediction["text"]
        if text not in text_counts:
            text_counts[text] = 0
        # 가중치를 적용하여 투표를 합니다.
        text_counts[text] += (1 * weight)
    # 가장 많은 투표를 받은 text를 최종 예측 결과로 선택합니다.
    final_text = max(text_counts, key=text_counts.get)
    final_predictions[key] = {"text": final_text}

converted_data = {}

# 주어진 JSON 데이터를 요구하는 형식으로 변환
for key, value in final_predictions.items():
    converted_data[key] = value["text"]

# 결과를 출력
print(json.dumps(converted_data, indent=4, ensure_ascii=False))
json.dump(converted_data, open("final_predictions_hard_weighted.json", "w"), indent=4, ensure_ascii=False)