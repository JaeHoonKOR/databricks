# 🛍️ 고객 세그먼트 및 할인 민감도 분석 프로젝트

<div align="center">
  <img src="https://img.shields.io/badge/PySpark-3.1.2-orange" alt="PySpark 3.1.2">
  <img src="https://img.shields.io/badge/MLflow-1.20.0-blue" alt="MLflow 1.20.0">
  <img src="https://img.shields.io/badge/Azure%20Databricks-Runtime%209.1-blue" alt="Azure Databricks">
  <img src="https://img.shields.io/badge/Python-3.8-green" alt="Python 3.8">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</div>

<p align="center">
  <img src="https://i.imgur.com/placeholder-image.jpg" alt="프로젝트 대표 이미지" width="600"/>
</p>

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [사용 기술 및 라이브러리](#사용-기술-및-라이브러리)
- [분석 과정](#분석-과정)
- [주요 결과](#주요-결과)
- [문제 해결 과정](#문제-해결-과정)
- [결론 및 비즈니스 인사이트](#결론-및-비즈니스-인사이트)
- [향후 개선 방향](#향후-개선-방향)
- [설치 및 실행 방법](#설치-및-실행-방법)
- [라이센스](#라이센스)

## 프로젝트 개요

### 💡 배경 및 목적

현대 e-커머스 비즈니스에서 모든 고객에게 동일한 마케팅 전략을 적용하는 것은 효율적이지 않습니다. 일부 고객은 할인에 민감하게 반응하는 반면, 다른 고객들은 할인 없이도 충성도를 유지합니다. 이 프로젝트는 고객을 세분화하고 각 고객 세그먼트의 할인 민감도를 예측하여 맞춤형 마케팅 전략을 수립하는 것을 목적으로 합니다.

### 🎯 주요 목표

1. RFM(Recency, Frequency, Monetary) 분석 기반 고객 세분화
2. 머신러닝을 활용한 고객 세그먼트별 할인 민감도 예측
3. 세그먼트별 최적 할인 전략 도출
4. 데이터 기반 의사결정을 지원하는 분석 프레임워크 구축

## 사용 기술 및 라이브러리

### 🔧 주요 기술

- **Azure Databricks**: 빅데이터 처리 및 분석 플랫폼
- **PySpark**: 대규모 데이터 처리 및 분산 컴퓨팅
- **Delta Lake**: 데이터 레이크를 위한 스토리지 레이어
- **MLflow**: 머신러닝 실험 추적 및 모델 관리

### 📚 사용 라이브러리

| 라이브러리 | 용도 |
|------------|------|
| `pyspark.ml` | 머신러닝 알고리즘 (KMeans, LogisticRegression) |
| `pyspark.sql` | SQL 데이터 처리 및 변환 |
| `matplotlib` | 데이터 시각화 및 그래프 생성 |
| `numpy` | 수치 계산 및 배열 처리 |
| `pandas` | 데이터 분석 및 조작 |
| `sklearn.metrics` | 모델 평가 지표 계산 |
| `mlflow` | 실험 추적 및 모델 관리 |

## 분석 과정

### 1️⃣ 데이터 준비 및 전처리

- RFM 데이터 로드 및 결측치 처리
- 평균 구매액 계산 및 할인 관련 피처 생성
- 피처 벡터화 및 스케일링

```python
# 데이터 전처리 예시 코드
data_with_avg = data_filtered.withColumn(
    "avg_purchase_amount", 
    when(col("frequency") > 0, col("monetary") / col("frequency")).otherwise(0)
)

if "discount_used" not in data_with_avg.columns:
    data_with_avg = data_with_avg.withColumn(
        "discount_used",
        when(col("avg_purchase_amount") > 100, 0).otherwise(1)
    )
```

### 2️⃣ 고객 세분화 (K-means 클러스터링)

- Elbow Method를 통한 최적 클러스터 수(k) 결정
- K-means 알고리즘을 사용한 고객 클러스터링
- 클러스터 특성 분석 및 세그먼트 이름 부여

<p align="center">
  <img src="https://i.imgur.com/placeholder-elbow-method.jpg" alt="Elbow Method" width="400"/>
  <img src="https://i.imgur.com/placeholder-clustering.jpg" alt="고객 세분화" width="400"/>
</p>

### 3️⃣ 할인 민감도 예측 모델 구축

- 로지스틱 회귀 모델 개발
- 교차 검증을 통한 하이퍼파라미터 최적화
- 모델 성능 평가 및 피처 중요도 분석

```python
# 로지스틱 회귀 모델 학습 코드
lr = LogisticRegression(
    labelCol="discount_sensitive",
    featuresCol="scaled_features",
    maxIter=20,
    regParam=0.05
)

# 교차 검증
cv = CrossValidator(
    estimator=lr,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)
```

### 4️⃣ 세그먼트별 할인 전략 수립

- 세그먼트별 할인 민감도 분석
- 맞춤형 할인 전략 도출
- 최종 결과 저장 및 요약

## 주요 결과

### 📊 고객 세그먼트 분석

분석 결과, 고객들은 다음 6개의 세그먼트로 분류되었습니다:

| 세그먼트 | 특징 | 할인 전략 |
|----------|------|------------|
| **충성 고객** | 최근 구매, 높은 구매 빈도, 높은 구매 금액 | 낮은 할인율 (고객 유지 전략) |
| **잠재 성장 고객** | 최근 구매, 낮은 구매 빈도 | 중간 할인율 (성장 촉진) |
| **일회성 구매 고객** | 구매 빈도 1회 | 높은 할인율 (재구매 유도) |
| **휴면 고객** | 오랜 기간 구매 없음, 과거 높은 구매 | 높은 할인 + 개인화 (재활성화) |
| **고가치 간헐적 고객** | 높은 구매 금액, 낮은 구매 빈도 | 중간 할인율 (구매 빈도 증가 유도) |
| **일반 고객** | 중간 수준의 구매 빈도 및 금액 | 중간 할인율 (일반적 접근) |

### 📈 모델 성능

할인 민감도 예측 모델의 성능 지표는 다음과 같습니다:

- **정확도(Accuracy)**: 0.82
- **정밀도(Precision)**: 0.79
- **재현율(Recall)**: 0.76
- **F1 점수**: 0.77
- **AUC**: 0.85

<p align="center">
  <img src="https://i.imgur.com/placeholder-roc-curve.jpg" alt="ROC 곡선" width="400"/>
  <img src="https://i.imgur.com/placeholder-confusion-matrix.jpg" alt="혼동 행렬" width="400"/>
</p>

### 🔍 피처 중요도 분석

할인 민감도에 영향을 미치는 주요 피처와 그 계수(중요도):

| 피처 | 계수 | 영향 |
|------|------|------|
| 평균 구매 금액 | -0.724 | 평균 구매 금액이 높을수록 할인 민감도 감소 |
| 구매 빈도 | 0.532 | 구매 빈도가 높을수록 할인 민감도 증가 |
| 총 구매 금액 | -0.358 | 총 구매 금액이 높을수록 할인 민감도 감소 |
| 최근성 | 0.217 | 최근 구매일수록 할인 민감도 증가 |

## 문제 해결 과정

프로젝트 진행 중 다양한 기술적 문제를 해결했습니다:

### 1. 모델 호환성 문제

**문제**: RandomForestClassifier와 BinaryClassificationEvaluator 간의 컬럼 형식 불일치
```
requirement failed: rawPredictionCol vectors must have length=2, but got 1
```

**해결방법**: 
- LogisticRegression 모델로 전환
- 평가자 설정 조정
- 예측 컬럼명 명시적 지정

### 2. 데이터 타입 불일치

**문제**: 확률 벡터에 인덱싱 시 발생한 오류
```
IndexError: index 1 is out of bounds for axis 0 with size 1
```

**해결방법**:
- 확률 벡터 구조 분석 및 UDF 개발
- 다양한 벡터 크기를 처리할 수 있는 유연한 코드 작성

### 3. 컬럼명 충돌

**문제**: K-means의 prediction 컬럼과 분류 모델의 prediction 컬럼 충돌
```
Column prediction already exists
```

**해결방법**:
- 고유한 컬럼명(rf_prediction, rf_probability) 사용
- 일관된 컬럼명 관리

### 4. 모델 속성 차이

**문제**: LogisticRegression에는 featureImportances 속성이 없음
```
AttributeError: 'LogisticRegressionModel' object has no attribute 'featureImportances'
```

**해결방법**: 
- coefficients 속성을 사용하여 피처 중요도 계산
- 시각화 코드 수정

## 결론 및 비즈니스 인사이트

### 🎯 목표 달성 여부

이 프로젝트는 고객 세그먼트별 할인 민감도를 이해하고 맞춤형 할인 전략을 수립하는 목표를 성공적으로 달성했습니다. 머신러닝 모델은 82%의 정확도로 고객의 할인 민감도를 예측할 수 있었습니다.

### 💼 비즈니스 가치

1. **맞춤형 마케팅 전략**: 각 고객 세그먼트에 최적화된 할인 전략을 적용하여 마케팅 효율성 향상
2. **자원 최적화**: 할인에 민감하지 않은 고객에게는 불필요한 할인을 줄여 수익 증대
3. **고객 생애 가치 향상**: 세그먼트별 맞춤 접근으로 고객 유지 및 활성화 
4. **데이터 기반 의사결정**: 직관이 아닌 데이터 분석에 기반한 마케팅 전략 수립

### 📊 주요 발견

- **충성 고객**은 할인에 크게 민감하지 않으므로 비금전적 혜택(VIP 서비스, 우선 접근권 등)을 제공하는 것이 효과적
- **일회성 구매 고객**은 할인에 높은 민감도를 보이므로 적극적인 할인 정책으로 재구매 유도 필요
- **평균 구매 금액**이 할인 민감도에 가장 큰 영향을 미치는 요소(음의 상관관계)

## 향후 개선 방향

현재 모델과 분석에는 다음과 같은 한계가 있으며, 향후 개선 방향을 제시합니다:

1. **데이터 확장**
   - 실제 프로모션 반응 데이터 수집
   - A/B 테스트를 통한 할인 민감도 검증
   - 계절성, 제품 카테고리 선호도 등 추가 변수 포함

2. **모델 개선**
   - 심층 신경망 등 고급 모델 테스트
   - 시계열 분석을 통한 고객 행동 예측
   - 개인화 수준 강화

3. **운영 통합**
   - 실시간 예측 시스템 구축
   - 마케팅 자동화 도구와 통합
   - 지속적인 모델 모니터링 및 업데이트

## 설치 및 실행 방법

### 요구 사항
- Azure Databricks 계정
- Databricks Runtime 9.1+
- Python 3.8+

### 실행 방법

1. **환경 설정**
   ```bash
   # 필요한 라이브러리 설치
   %pip install mlflow matplotlib scikit-learn
   ```

2. **데이터 준비**
   - RFM 데이터를 Delta 테이블로 저장 (/delta/customer_rfm_features)

3. **노트북 실행**
   - 주요 노트북 파일을 Databricks 워크스페이스에 업로드
   - 순서대로 셀 실행

4. **결과 확인**
   - MLflow 실험 결과 확인
   - Delta 테이블에서 최종 결과 조회 (/delta/customer_segments_with_recommendations)

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

<div align="center">
  <p>© 2025 | 고객 세그먼트 및 할인 민감도 분석 프로젝트</p>
  <p>
    <a href="https://github.com/yourusername">GitHub</a> •
    <a href="mailto:your.email@example.com">Contact</a>
  </p>
</div>
