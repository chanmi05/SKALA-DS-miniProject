# ESS 배터리 수명 예측

ESS 셀 교체 비용은 설비 CAPEX의 30~40%를 차지한다. 이 프로젝트는 **초기 100 사이클 데이터만으로 셀의 수명(cycle life)을 예측**해, 열화가 눈에 보이기 전에 교체 시점을 계획할 수 있는지 검증한다.

## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019) — LFP/graphite 셀 A123 APR18650M1A (1.1 Ah)
- 학습 데이터 : Batch 1 (2017-05-12) · 41셀
- 평가 데이터 : Batch 2 (2018-02-20) · 39셀
- 비교 데이터 : Batch 3 (2018-04-12) · 40셀 — EDA의 배치 간 비교에만 사용 (모델 테스트에는 사용하지 않음)
- 태스크 : **Regression** — 타깃 `log10(cycle_life)`, 평가 지표 MAPE
- 원논문 목표 : MAPE 9.1%

> **데이터 주의** : 원논문의 Batch 2는 `2017-06-30` 파일이며, 이 과제의 `2018-02-20` 파일은 논문에 없는 별도 실험이다 (논문의 B1–B2 병합 쌍이 정책·용량 모두 이어지지 않음, `01_EDA` 0-2). 따라서 원논문 성능과의 차이(Gap)는 이 조건 차이를 전제로 해석한다.

## 파일 구조
~~~
├── data/
│   └── README.md                    # 데이터 출처 · 다운로드 · 생성 파일 설명
├── notebooks/
│   ├── 01_EDA.ipynb                 # 품질 진단 · 정제 근거 · Q1~Q5 (배치 비교)
│   ├── 02_feature_engineering.ipynb # 타깃 변환 · ΔQ 사이클 쌍 · 배치 안정성 · VIF · 피처 세트
│   └── 03_modeling.ipynb            # 후보 모델 비교 · 선택 · 성능 리포트 · 오류 분석
├── src/
│   ├── preprocess.py                # kagglehub 다운로드, h5py 선택 로딩, 진단, 정제
│   ├── features.py                  # ΔQ(V) · 용량 곡선 · 온도 · 저항 · 정책 피처, VIF
│   ├── train.py                     # 분할, GroupKFold 탐색, 모델 비교·선택, 평가
│   └── utils.py                     # 결과 로그 · 그림 저장
├── results/
│   ├── model_performance.csv        # 성능 리포트 (포맷)
│   ├── model_comparison.csv         # 전체 후보 모델 × 피처 세트 결과
│   └── 0X_*/                        # 노트북별 그림 · 로그 · 예측값
├── requirements.txt
└── README.md
~~~

## 환경 설정
~~~bash
git clone https://github.com/<github-id>/ess-battery-project
cd ess-battery-project
pip install -r requirements.txt
~~~
노트북을 `01 → 02 → 03` 순서로 실행한다. 01이 데이터(약 8GB)를 내려받아 `data/processed/dataset.pkl`로 저장하고, 02가 `features.csv`·`feature_sets.json`을, 03이 `results/`를 만든다. 학습만 다시 돌릴 때는 `python -m src.train`.
Google Colab에서는 각 노트북 첫 셀의 `WORK_DIR`을 Drive의 프로젝트 위치로 바꾸면 된다.

## EDA

데이터 정제 : 139셀 → **120셀** (B1 41 · B2 39 · B3 40). EOL 전에 측정이 끝난 셀, 고정 정책이 아닌 실험(VarCharge·SLOWCYCLE), 논문 지정 노이즈 채널을 제외했다. B1 장수명 5셀(b1c0~4)은 논문이 보고한 최종 수명을 라벨로 썼다.

- **Cycle Life 분포**
  - 로그 변환으로 왜도 1.46 → 0.27. 세 배치의 분포는 서로 다르다 (크루스칼-월리스 p = 2.5×10⁻¹¹)
  - B2는 단수명(<500) 72%, B1·B3는 0%
  - 핵심 발견 : **B2 셀의 77%가 학습 데이터(B1)의 최솟값 534보다 짧다** → 학습 범위 밖을 예측해야 하는 문제

- **열화 곡선 분석**
  - 열화 속도는 수명 후반에 14~36배 빨라진다 (20~40% 구간 대비 80~100% 구간)
  - Knee point는 배치와 무관하게 수명의 약 75~80% 지점 (knee–수명 r = 0.99)
  - 핵심 발견 : 초기 100 사이클은 knee보다 훨씬 앞이라 **용량이 거의 줄지 않는다**(80% 셀은 오히려 증가) → 용량 값만으로는 예측 불가

- **ΔQ(V) 곡선 분석**
  - ΔQ₁₀₀₋₁₀(V) = 사이클 100과 10의 방전 전압 곡선 차이. 단수명 셀일수록 3.0~3.2 V 부근의 골이 깊다
  - log Var(ΔQ)와 log 수명 : r = −0.945 (B1) · −0.918 (B2) · −0.805 (B3), 논문 −0.93 재현
  - 핵심 발견 : **세 배치에서 모두 강하고 방향이 같은 유일한 신호**. 단, 같은 ΔQ에서 B2 수명이 약 20% 짧은 평행 이동이 있다

- **충전 속도(C-rate)와 수명의 관계**
  - B1에서는 평균 C-rate가 높을수록 수명이 짧다 (Spearman ρ = −0.62)
  - B2·B3는 모든 셀이 약 10분 충전으로 설계되어 평균 C-rate가 사실상 상수 (표준편차 0.004~0.005)
  - 핵심 발견 : 같은 정책이라도 배치·`newstructure` 표기에 따라 수명이 최대 3배 다르다. B2 안에서 old 451 vs new 904 (맨-휘트니 p = 7×10⁻⁶), B1에는 new 셀이 없다

- **상관관계 · 다중공선성**
  - 후보 피처 24개 중 14개는 배치 간 상관 부호가 바뀐다 (충전 시간, 평균 C-rate, 초기 용량, IR 등)
  - B1 기준 24개 중 23개가 VIF ≥ 10

## Modeling

### 피처 엔지니어링 전략
원본 신호를 수명과의 관계가 드러나는 **파생변수**로 바꿨다 : 타깃·ΔQ 분산의 로그 변환, ΔQ 곡선 통계량(분산·왜도·첨도 등), 정책 문자열의 수치 분해(C1·Q1·C2·평균 C-rate), 사이클 구간별 용량 기울기·온도·저항 요약값.

선택은 "B1에서 강한가"가 아니라 **"세 배치에서 같은 방향인가"**를 기준으로 했다.

후보 24개 → ① 중복 제거(VIF < 10) 13개 → ② 세 배치 상관 부호 일치 **7개 (Selected)**

| 세트 | 구성 | 용도 |
|---|---|---|
| Selected (7) | dq_log_var, dq_skew, dq_kurt, qd_slope_91_100, tmax_2_100, Q1, C2 | 주 피처 |
| Variance (1) | dq_log_var | 기준선 (논문 Variance model) |
| Discharge (13) | ΔQ 통계 6 + 용량 곡선 7 | 논문 Discharge model 구성 |
| Full (20) | Discharge + 충전 시간 · 온도 · 저항 | 논문 Full model 구성 |

### 모델 선택 및 근거
- 선정 기준 (EDA 근거) : A. 학습 범위 밖 예측 (Q1) · B. 공선성에 강함 (Q5) · C. 41셀 소표본 과적합 억제 · D. 로그-로그 선형 구조 (Q3) · E. 해석 가능성
- 후보 모델 : Linear(1 피처), Ridge, Lasso, **Elastic Net**, PLS, SVR, GPR, Random Forest, XGBoost
  - 트리 모델은 예측값이 학습 수명 범위(534~2237)를 벗어나지 못해 기준 A를 충족하지 못한다 → 비교군
- 검증 : B1을 정책 그룹 단위로 Train / Valid(≈25%) 분할, Train 안에서 정책 단위 5-fold GroupKFold로 하이퍼파라미터 탐색
- 선택 규칙 : ① CV MAPE 최소 ② Valid − CV > 5%p면 과적합으로 제외 ③ 1%p 이내면 단순한 모델 ④ **B2 성능은 선택에 사용하지 않음**
- 최종 모델 : `(03 실행 결과로 채움)`
- 선택 이유 : `(03 실행 결과로 채움)`

## 성능 결과

| 구분 | MAPE (%) | 비고 |
|---|---|---|
| Train (Batch 1 CV) | | |
| Valid (Batch 1 Hold-out) | | |
| Test (Batch 2) | | |
| Gap (Train-Valid) | | (+) : 과적합 의심 |
| Gap (Valid-Test) | | (+) : 배치간 일반화 저하 의심 |
| Gap (Target-Test) | | Target : 원논문 9.1% |

## 오류 분석
- 모델이 가장 크게 틀린 셀의 공통점 : `(03 실행 결과로 채움)`
- 원인 가설 및 개선 방향 : `(03 실행 결과로 채움)`

## ESS 도메인 해석
- 활용 : `(03 실행 결과로 채움)`
- 한계 및 실 배포 시 필요한 것 : `(03 실행 결과로 채움)`

## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.
- 원논문 공개 코드 : https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation

## 팀 구성
- 박찬미 (울산 3반) : EDA, 피처 엔지니어링, 모델 개발, 성능 평가 (Batch 2)
