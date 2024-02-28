import json
from collections import defaultdict

import json
from glob import glob


prediction_files = glob("./results/*.json")

# 각 모델의 예측 결과를 담을 딕셔너리를 초기화합니다.
ensemble_predictions = defaultdict(list)

# 각 prediction 파일을 읽어와서 ensemble_predictions에 추가합니다.
for file_name in prediction_files:
    with open(file_name, "r") as file:
        predictions = json.load(file)
        for key, value in predictions.items():
            ensemble_predictions[key].extend(value)

# Soft voting을 사용하여 각 예측 결과에 대한 가중 평균을 계산합니다.
# 가중치는 probability 값으로 설정합니다.
final_predictions = {}
for key, predictions in ensemble_predictions.items():
    # 각 예측 결과의 probability를 합산합니다.
    total_probability = sum(prediction["probability"] for prediction in predictions)
    
    # 가중 평균을 계산합니다.
    weighted_sum = defaultdict(float)
    for prediction in predictions:
        text = prediction["text"]
        probability = prediction["probability"]
        weighted_sum[text] += probability / total_probability  # 가중 평균을 계산하여 더해줍니다.
    
    # 가장 높은 가중 평균을 가진 text를 최종 예측 결과로 선택합니다.
    final_text = max(weighted_sum, key=weighted_sum.get)
    final_predictions[key] = {"text": final_text, "probability": weighted_sum[final_text]}

# 결과를 출력합니다.
# print(json.dumps(final_predictions, indent=4, ensure_ascii=False))
# 변환된 결과를 담을 딕셔너리
converted_data = {}

# 주어진 JSON 데이터를 요구하는 형식으로 변환
for key, value in final_predictions.items():
    converted_data[key] = value["text"]

# 결과를 출력
print(json.dumps(converted_data, indent=4, ensure_ascii=False))
json.dump(converted_data, open("final_predictions.json", "w"), indent=4, ensure_ascii=False)