import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

CAT_FEATURES = [
    "weather",
    "season",
    "windspeed",
    "humidity",
]

# TODO: 전처리 파이프라인 작성
# 1. 방의 크기는 제곱근을 적용함 (FunctionTransformer 사용)
# 2. 층수는 실제 층수를 추출하되 숫자가 아닌 Basement 등은 0층으로 표기함
# 3. 범주형 변수(CAT_FEATURES)는 타겟 인코딩 적용 (from category_encoders import TargetEncoder)
preprocess_pipeline = ColumnTransformer(
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocess_pipeline.set_output(transform="pandas")
