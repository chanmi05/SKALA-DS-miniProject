# data

원본 데이터는 용량이 커서(약 8GB) 저장소에 포함하지 않습니다.

- 출처 : MIT-Stanford Battery Dataset (Severson et al., *Nature Energy* 2019), Kaggle `itshpark/data-driven-prediction-of-battery-cycle`
- 다운로드 : `notebooks/01_EDA.ipynb` 실행 시 `src/preprocess.py`의 `download()`가 `kagglehub`로 자동 다운로드
- 사용 파일

| 파일 | 배치 | 용도 |
|---|---|---|
| 2017-05-12_batchdata_updated_struct_errorcorrect.mat | Batch 1 | 학습 |
| 2018-02-20_batchdata_updated_struct_errorcorrect.mat | Batch 2 | 테스트 |
| 2018-04-12_batchdata_updated_struct_errorcorrect.mat | Batch 3 | 추가 검증 |

## processed/ (실행 시 생성)

| 파일 | 내용 |
|---|---|
| `dataset.pkl` | 정제된 셀 메타·사이클 요약·일부 사이클의 Qdlin (git 제외) |
| `features.csv` | 셀 × 피처 테이블 (02_feature_engineering 출력) |
| `feature_sets.json` | 모델링에 쓸 피처 세트와 ΔQ 사이클 쌍 |
