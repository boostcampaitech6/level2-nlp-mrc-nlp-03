import json
from collections import defaultdict
import json
from glob import glob
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
import evaluate
from itertools import combinations
from tqdm.auto import tqdm

metric = evaluate.load('squad')
prediction_files = glob("./results/*.json")

best_EM = 0

for i in range(6, len(prediction_files)):
    # 각 모델의 예측 결과를 담을 딕셔너리를 초기화합니다.
    combinations_list = list(combinations(prediction_files, i))
    for combi in tqdm(combinations_list):
        combi = list(combi)
        ensemble_predictions = defaultdict(list)
        # 각 prediction 파일을 읽어와서 ensemble_predictions에 추가합니다.

        for file_name in combi:
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

        # 변환된 결과를 담을 딕셔너리
        converted_data = {}

        # 주어진 JSON 데이터를 요구하는 형식으로 변환
        for key, value in final_predictions.items():
            converted_data[key] = value["text"]

        # 결과를 출력
        # print(json.dumps(converted_data, indent=4, ensure_ascii=False))
        json.dump(converted_data, open("final_predictions_soft.json", "w"), indent=4, ensure_ascii=False)
        predictions = [{"id": id, "prediction_text": pred} for id, pred in converted_data.items()]

        # 정답 데이터 읽기
        with open('./references.json', encoding='utf-8') as reader:
            references = json.load(reader)

        # 정답 데이터와 예측 데이터를 비교하여 정확도 계산
        metrics = metric.compute(predictions=predictions, references=references)
        
        if best_EM < metrics["exact_match"]:
            best_EM = metrics["exact_match"]
            best_combi = combi
            best_metrics = metrics

print(best_combi)