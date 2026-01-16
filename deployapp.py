import streamlit as st
import numpy as np
import time
from PIL import Image
import tensorflow as tf


# ==================================================
# 기본 설정
# ==================================================
st.set_page_config(
    page_title="Waste Classification Demo",
    layout="centered"
)

CLASS_NAMES = [
    "cardboard", "glass", "metal", "paper", "plastic", "trash"
]

IMG_SIZE = (224, 224)


# ==================================================
# 모델 로딩 (Cloud 안전)
# ==================================================
@st.cache_resource
def load_tf_models():
    """
    TensorFlow SavedModel 로드
    (추론 전용, signatures 기반)
    """
    mn = tf.saved_model.load("mobilenetv2_infer_savedmodel")
    ef = tf.saved_model.load("efficientnetb0_infer_savedmodel")
    return mn, ef


# ==================================================
# 전처리
# ==================================================
def preprocess_image(image: Image.Image):
    img = image.resize(IMG_SIZE)
    x = np.array(img).astype(np.float32)
    return x


# ==================================================
# TensorFlow SavedModel 추론
# ==================================================
def predict_tf(model, image: Image.Image, model_name: str):
    x = preprocess_image(image)
    x = x[None, ...]

    if model_name == "MobileNetV2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    else:
        x = tf.keras.applications.efficientnet.preprocess_input(x)

    infer = model.signatures["serving_default"]

    start = time.time()
    outputs = infer(tf.constant(x))
    latency = (time.time() - start) * 1000

    preds = list(outputs.values())[0].numpy()[0]
    return preds, latency


# ==================================================
# UI
# ==================================================
st.title("♻️ 쓰레기 분류 데모")
st.caption("TensorFlow 추론 (Streamlit Cloud 전용)")

st.sidebar.header("모델 선택")

model_name = st.sidebar.selectbox(
    "모델",
    ["MobileNetV2", "EfficientNet-B0"]
)

uploaded_file = st.file_uploader(
    "이미지 업로드",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드 된 이미지", width=450)

    if st.button("추론 시작"):
        with st.spinner("모델 추론 중..."):
            mn, ef = load_tf_models()
            model = mn if model_name == "MobileNetV2" else ef
            preds, latency = predict_tf(model, image, model_name)

        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])

        st.success(f"추론 결과: **{CLASS_NAMES[pred_idx]}**")
        st.metric("신뢰도", f"{confidence:.2%}")
        st.metric("추론 시간", f"{latency:.2f} ms")

        st.subheader("클래스 확률")
        st.bar_chart(
            {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        )

else:
    st.info("추론을 위해 이미지를 업로드 해주세요.")
