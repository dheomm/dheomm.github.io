---
title: XGBoost
date: 2026-02-24 
categories: [데이터분석, 머신러닝]
tags: [XGBoost, Python, 머신러닝, 앙상블]
---

## XGBoost란?

XGBoost(eXtreme Gradient Boosting)는 **그래디언트 부스팅** 알고리즘을 기반으로 한 머신러닝 라이브러리입니다. 빠른 속도와 높은 성능으로 Kaggle 대회에서 가장 많이 사용되는 알고리즘 중 하나입니다.

## 핵심 개념

### 1. 앙상블 학습
여러 개의 약한 모델(weak learner)을 결합해 강한 모델을 만드는 방식입니다.

### 2. 부스팅(Boosting)

이전 모델이 틀린 부분을 다음 모델이 집중적으로 학습합니다. 순차적으로 모델을 개선해 나가는 방식입니다.

### 3. 그래디언트 부스팅

손실 함수의 그래디언트(기울기)를 이용해 오차를 줄여나갑니다.

## XGBoost의 장점

- **빠른 속도**: 병렬 처리와 캐시 최적화
- **과적합 방지**: 정규화(regularization) 내장
- **결측치 처리**: 자동으로 결측값 처리
- **유연성**: 분류, 회귀, 랭킹 등 다양한 문제에 적용 가능

## 설치

```python
pip install xgboost
```

## 간단한 예제
붓꽃(Iris) 데이터로 분류 모델을 만들어봅니다.
```python
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size = 0.2, radom_state = 42
)

# 모델 생성 및 학습
model = xgb.XGBClassifier(
  n_estimators = 100,
  max_depth = 3,
  learning_rate = 0.1,
  random_state = 42
)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.4f}")

## 주요 하이퍼파라미터
| 파라미터 | 설명 | 기본값 |
| 'n_estimators' | 트리 개수 | 100 |
| 'max_depth' | 트리 최대 깊이 | 6 |
| 'learning_rate' | 학습률(낮을수록 천천히 학습) | 0.3 |
| 'subsample' | 각 트리에 사용할 샘플 비율 | 1 |
| 'colsample_bytree' | 각 트리에 사용할 피처 비율 | 1 |

## 마무리
XGBoost는 정형 데이터에서 뛰어난 성능을 보여주는 강력한 알고리즘입니다. 다음 글에서는 하이퍼파라미터 튜닝과 Feature Importance 해석 방법을 알아보겠습니다.

## 참고 자료
- [XGBoost 공식 문서](https://xgboost.readthedocs.io/)
- [XGBoost 논문](https://arxiv.org/abs/1603.02754)




























