```python
# app.py
# 실행:
# streamlit run app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# -----------------------------
# 운영체제별 한글 폰트 설정
# -----------------------------
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")

elif platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")

else:
    # Streamlit Cloud (Linux)
    plt.rc("font", family="DejaVu Sans")

# 마이너스 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

sns.set()

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="폐암 예측 시스템",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 폐암 데이터 분석 및 군집 예측")

# -----------------------------
# 파일 경로
# -----------------------------
CSV_PATH = "dataset.csv"

MODEL_PATH = "model-longdisease.pkl"

SCALER_PATH = "longdisease-scaler.pkl"

# -----------------------------
# 데이터 불러오기
# -----------------------------
df = pd.read_csv(CSV_PATH)

# 컬럼 공백 제거
df.columns = df.columns.str.strip()

# -----------------------------
# 모델 / 스케일러 로드
# -----------------------------
model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

# -----------------------------
# 데이터 미리보기
# -----------------------------
st.subheader("📄 데이터 미리보기")

st.dataframe(df.head())

# -----------------------------
# 사이드바 입력
# -----------------------------
st.sidebar.header("🧪 사용자 입력")

age = st.sidebar.slider(
    "나이 (AGE)",
    min_value=int(df["AGE"].min()),
    max_value=int(df["AGE"].max()),
    value=30
)

smoking = st.sidebar.slider(
    "흡연 여부 (SMOKING)",
    min_value=0,
    max_value=2,
    value=1
)

alcohol = st.sidebar.slider(
    "음주 여부 (ALCOHOL_CONSUMING)",
    min_value=0,
    max_value=2,
    value=1
)

chest_pain = st.sidebar.slider(
    "가슴 통증 여부 (CHEST_PAIN)",
    min_value=0,
    max_value=2,
    value=1
)

# -----------------------------
# 사용자 입력 데이터 생성
# -----------------------------
input_data = pd.DataFrame(
    [[age, smoking, alcohol, chest_pain]],
    columns=[
        "AGE",
        "SMOKING",
        "ALCOHOL_CONSUMING",
        "CHEST_PAIN"
    ]
)

# -----------------------------
# 스케일링
# -----------------------------
scaled_data = scaler.transform(input_data)

# -----------------------------
# 예측
# -----------------------------
prediction = model.predict(scaled_data)

cluster = prediction[0]

# -----------------------------
# 결과 출력
# -----------------------------
st.subheader("🤖 예측 결과")

if cluster == 0:
    st.success(f"예측 군집: {cluster}")
    st.write("상대적으로 위험도가 낮은 그룹")

elif cluster == 1:
    st.warning(f"예측 군집: {cluster}")
    st.write("상대적으로 위험도가 높은 그룹")

else:
    st.info(f"예측 군집: {cluster}")

# -----------------------------
# 입력 데이터 출력
# -----------------------------
st.subheader("📌 사용자 입력 데이터")

st.dataframe(input_data)

# -----------------------------
# 시각화
# -----------------------------
st.subheader("📊 데이터 시각화")

tab1, tab2, tab3 = st.tabs([
    "흡연 vs 나이",
    "음주 vs 가슴통증",
    "상관관계 분석"
])

# -----------------------------
# 그래프 1
# -----------------------------
with tab1:

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="SMOKING",
        y="AGE",
        hue="LUNG_CANCER",
        ax=ax
    )

    ax.scatter(
        smoking,
        age,
        color="red",
        s=250,
        marker="X",
        label="사용자"
    )

    ax.set_title("흡연 여부 vs 나이", fontsize=18)

    ax.set_xlabel("흡연 여부")

    ax.set_ylabel("나이")

    st.pyplot(fig)

# -----------------------------
# 그래프 2
# -----------------------------
with tab2:

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="ALCOHOL_CONSUMING",
        y="CHEST_PAIN",
        hue="LUNG_CANCER",
        ax=ax
    )

    ax.scatter(
        alcohol,
        chest_pain,
        color="black",
        s=250,
        marker="X",
        label="사용자"
    )

    ax.set_title("음주 여부 vs 가슴 통증", fontsize=18)

    ax.set_xlabel("음주 여부")

    ax.set_ylabel("가슴 통증")

    st.pyplot(fig)

# -----------------------------
# 그래프 3
# -----------------------------
with tab3:

    numeric_df = df.select_dtypes(include="number")

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("상관관계 분석", fontsize=18)

    st.pyplot(fig)

# -----------------------------
# 통계 정보
# -----------------------------
st.subheader("📈 데이터 통계")

st.dataframe(df.describe())

# -----------------------------
# 하단
# -----------------------------
st.markdown("---")

st.caption("Streamlit 기반 폐암 군집 분석 시스템")
```
