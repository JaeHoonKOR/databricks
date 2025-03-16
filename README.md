# 🛍️ 고객 세그먼트 및 할인 민감도 분석 프로젝트

<div align="center">
  <img src="https://img.shields.io/badge/PySpark-3.1.2-orange" alt="PySpark 3.1.2">
  <img src="https://img.shields.io/badge/MLflow-1.20.0-blue" alt="MLflow 1.20.0">
  <img src="https://img.shields.io/badge/Azure%20Databricks-Runtime%209.1-blue" alt="Azure Databricks">
  <img src="https://img.shields.io/badge/Python-3.8-green" alt="Python 3.8">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/09e2e7de-879e-4d5c-ae4d-4e41d719f022" alt="프로젝트 대표 이미지" width="600"/>
</p>

---

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [사용 기술 및 라이브러리](#사용-기술-및-라이브러리)
- [분석 과정](#분석-과정)
- [주요 결과](#주요-결과)
- [문제 해결 과정](#문제-해결-과정)
- [결론 및 비즈니스 인사이트](#결론-및-비즈니스-인사이트)
- [향후 개선 방향](#향후-개선-방향)
- [설치 및 실행 방법](#설치-및-실행-방법)

---

## 프로젝트 개요

### 💡 배경 및 목적

많은 e-커머스 기업들이 여전히 모든 고객에게 동일한 할인 전략을 적용하고 있지만, 이는 효율적이지 않습니다. 할인에 민감한 고객에게는 적극적인 프로모션이 필요하고, 충성도가 높은 고객에게는 보다 정교한 접근이 필요합니다.  
본 프로젝트는 고객의 특성을 분석하여, 각 세그먼트의 할인 민감도를 예측하고, 데이터 기반으로 효율적인 마케팅 전략을 수립하는 것이 목적입니다.

### 🎯 주요 목표

1. RFM(Recency, Frequency, Monetary) 분석으로 고객 세분화
2. 머신러닝을 통한 할인 민감도 예측
3. 세그먼트별 최적의 할인 전략 제안
4. 데이터 중심의 마케팅 의사결정 지원 프레임워크 개발

---
<a id="사용-기술-및-라이브러리">
  
## 🔧 사용 기술 및 라이브러리

- **Azure Databricks**: 빅데이터 분석 플랫폼
- **PySpark**: 대규모 데이터 분산 처리
- **Delta Lake**: 데이터 레이크 관리
- **MLflow**: 모델 추적 및 성능 관리

| 라이브러리 | 용도 |
|------------|------|
| `pyspark.ml` | KMeans, LogisticRegression 등 모델 구축 |
| `pyspark.sql` | 데이터 전처리 |
| `matplotlib` | 시각화 |
| `numpy`, `pandas` | 데이터 분석 |
| `sklearn.metrics` | 모델 평가 |
| `mlflow` | 실험 관리 |
</a>

---
<a id="분석-과정">
  
## 📊 분석 과정

### 1️⃣ 데이터 준비 및 전처리

- RFM 데이터를 정제하고 결측치를 처리했습니다.
- 평균 구매액과 할인 사용 여부 등 추가 변수를 계산했습니다.
- 분석을 위해 데이터를 정규화했습니다.

### 2️⃣ K-means 고객 세분화

- Elbow Method를 사용하여 최적의 고객 클러스터 개수를 결정하고, K-means 알고리즘을 통해 세분화했습니다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1fb8d382-9f76-4037-a395-38d013e4bc0f" alt="Elbow Method" width="400"/>

  <img src="https://github.com/user-attachments/assets/0451f791-81c2-4c26-8eaa-0728d1a01e7e" alt="고객 세분화" width="400"/>

</p>

### 2️⃣ 할인 민감도 예측 모델 구축

- 로지스틱 회귀 모델을 통해 각 세그먼트의 할인 민감도를 예측했습니다.
- 교차 검증으로 모델 정확도를 높이고 성능을 평가했습니다.

```python
# 모델 학습 예시 코드
cv = CrossValidator(
    estimator=LogisticRegression(featuresCol='features', labelCol='discount_sensitive'),
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)
```
</a>

---

<a id="주요-결과">
   
## 📌 주요 결과

| 고객 세그먼트 | 특징 | 제안된 할인 전략 |
|--------------|------|---------------|
| **충성 고객** | 빈도↑, 금액↑ | 낮은 할인, 비금전적 혜택 제공 |
| **잠재 성장 고객** | 최근 구매, 빈도 낮음 | 중간 할인 |
| **일회성 고객** | 할인 민감도 높음 | 적극적 할인 적용 |
| **휴면 고객** | 장기 미구매 고객 | 재구매 유도 프로모션 |
| **고액 간헐적 고객** | 구매액 높고 빈도 낮음 | 중간 수준 할인 |

### 모델 성능 지표

- **정확도**: 82%
- **F1 점수**: 0.77
- **AUC**: 0.85

---
<p align="center">

  <img src="https://github.com/user-attachments/assets/36916bec-2bc9-4ffe-95d8-3e9626e0f264" alt="혼동 행렬" width="400"/>

</p
</a>
<a id="문제-해결-과정">
  
## 🚧 문제 해결 과정

| 문제 | 해결 방안 |
|------|----------|
| 벡터 길이 불일치 | 모델 교체 (RandomForest → LogisticRegression) |
| 데이터 컬럼 충돌 | 명확한 컬럼명으로 수정 |
| 피처 중요도 확인 불가 | `coefficients` 활용하여 중요도 산출 |

---
</a>
<a id="결론-및-비즈니스-인사이트">
  
## 💼 결론 및 비즈니스 인사이트

고객 세그먼트별로 할인 민감도가 명확히 달랐습니다. 특히 **평균 구매 금액**이 할인 민감도에 가장 큰 영향을 주는 것으로 나타났습니다. 충성 고객에게는 특별한 VIP 서비스 같은 혜택을, 민감한 세그먼트에는 적극적인 할인 프로모션이 유효합니다.

이러한 맞춤형 접근을 통해 마케팅 효율성 개선과 장기적 고객 생애 가치(LTV) 증대가 가능할 것으로 기대합니다.

</a>
---
<a id="향후-개선-방향">
  
## 🚀 향후 개선 방향

- 실제 고객 프로모션 데이터를 통해 모델 정확성 지속적으로 검증 및 개선
- 심층 신경망이나 앙상블 모델 등을 도입하여 성능 개선
- 실시간 마케팅 자동화를 위한 예측 시스템 도입 검토

---

</a>

## ⚙️ 설치 및 실행 방법

### 요구 사항
- Azure Databricks Runtime 9.1 이상
- Python 3.8 이상

### 실행 순서
```bash
%pip install mlflow matplotlib scikit-learn
```

1. Databricks에서 노트북 업로드
2. 데이터를 Delta 테이블로 저장 후 셀 순서대로 실행
3. MLflow로 모델 결과 및 성능 확인

- 데이터 출처:
  [Walmart Customer Purchase Dataset](https://www.kaggle.com/datasets/logiccraftbyhimanshi/walmart-customer-purchase-behavior-dataset/data)

---

<div align="center">
  <p>© 2025 | 고객 세그먼트 및 할인 민감도 분석 프로젝트</p>
  <p>
    <a href="https://github.com/yourusername">GitHub</a> •
    <a href="mailto:your.email@example.com">Contact</a>
  </p>
</div>
