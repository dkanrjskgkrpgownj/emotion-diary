import streamlit as st
from deep_translator import GoogleTranslator
from transformers import pipeline
from datetime import datetime

# 감정 분석 모델 준비
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=3
    )

emotion_classifier = load_model()
emotion_messages = {
    "joy": "😊 기쁜 하루였네요! 그 긍정 에너지로 내일도 힘내요!",
    "sadness": "💙 슬픔이 느껴지는 하루였어요. 감정도 충분히 쉬어야 해요.",
    "anger": "🧘 분노가 느껴졌다면 천천히 심호흡하며 자신을 다독여보세요.",
    "fear": "🌿 불안한 하루였다면 따뜻한 차 한잔으로 마음을 쉬게 해보세요.",
    "surprise": "✨ 놀라움이 가득한 하루였네요! 변화도 성장의 일부랍니다.",
    "disgust": "🛡️ 불쾌한 감정은 스스로를 지키는 방어일 수 있어요. 잘 버텼어요."
}


st.title("📝 감정 일기 분석기 (Korean Diary Emotion Analyzer)")
st.markdown("**오늘의 일기를 한국어로 입력하면, AI가 감정을 분석해줍니다.**")

diary = st.text_area("✏️ 오늘의 일기 입력", height=200)

if st.button("감정 분석하기"):
    if diary.strip() == "":
        st.warning("일기를 입력해 주세요.")
    else:
        
        with st.spinner("일기를 영어로 번역 중..."):
            translated = GoogleTranslator(source='ko', target='en').translate(diary)
        
        st.success("번역 완료!")
        st.markdown(f"**🔁 번역된 일기:** {translated}")

       
        with st.spinner("감정 분석 중..."):
            result = emotion_classifier(translated)
        
        main_emotion = result[0][0]['label']
        main_score = result[0][0]['score']
        percent = f"{main_score*100:.2f}%"

        st.markdown(f"## 📘 오늘의 감정: **{main_emotion.capitalize()}** ({percent})")
        st.info(emotion_messages.get(main_emotion.lower(), "당신의 하루도 소중해요."))

       
        st.markdown("### 📊 감정 점수")
        labels = [r['label'].capitalize() for r in result[0]]
        scores = [r['score'] for r in result[0]]
        st.bar_chart(data=scores, use_container_width=True)

        