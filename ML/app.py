# NỘI DUNG CỦA FILE app.py

import streamlit as st
import joblib
import numpy as np
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

# ==========================================================
# BƯỚC 1: SAO CHÉP CÁC HÀM TIỆN ÍCH TỪ NOTEBOOK VÀO ĐÂY
# Các hàm này phải giống hệt lúc bạn huấn luyện mô hình.
# ==========================================================

# Tải stopwords một lần duy nhất
try:
    stopwords_english = stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')
    stopwords_english = stopwords.words('english')

def process_tweet(tweet):
    stemmer = PorterStemmer()
    tweet = re.sub(r'^\$?\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    return tweets_clean

def extract_six_features(tweet, freqs):
    word_l = process_tweet(tweet)
    first_second_pronouns = {'i', 'me', 'myself', 'we', 'us', 'ourself', 'ourselves', 'you', 'yourself', 'yourselves'}
    x = np.zeros((1, 7))
    x[0,0] = 1 # Bias term
    for word in word_l:
        x[0,1] += freqs.get((word, 1.0), 0)
        x[0,2] += freqs.get((word, 0.0), 0)
        if word in first_second_pronouns:
            x[0,4] += 1
    x[0,3] = 1 if 'no' in word_l else 0
    x[0,5] = 1 if '!' in tweet else 0
    x[0,6] = np.log(len(word_l) + 1)
    return x

# ==========================================================
# BƯỚC 2: TẢI MODEL VÀ TỪ ĐIỂN ĐÃ LƯU
# ==========================================================

try:
    model_pipeline = joblib.load('knn_sentiment_pipeline.pkl')
    freqs = joblib.load('freqs.pkl')
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy file model hoặc freqs. Vui lòng chạy notebook để tạo các file .pkl trước.")
    st.stop()

# ==========================================================
# BƯỚC 3: XÂY DỰNG GIAO DIỆN STREAMLIT

# ==========================================================

st.set_page_config(page_title="Phân tích Cảm xúc", page_icon="💬", layout="centered")
st.title("💬 Bộ phân tích Cảm xúc Tweet")
st.write("Ứng dụng này sử dụng mô hình K-Nearest Neighbors (KNN) đã được huấn luyện để dự đoán cảm xúc của một câu tweet là Tích cực hay Tiêu cực.")

# Ô nhập liệu
user_input = st.text_area("Nhập câu tweet của bạn (bằng tiếng Anh):", "This movie was absolutely fantastic! I loved every minute of it.")

# Nút bấm dự đoán
if st.button("Phân tích"):
    if user_input:
        # 1. Trích xuất 7 features (bao gồm bias)
        features = extract_six_features(user_input, freqs)
        
        # 2. Dự đoán bằng pipeline
        # Pipeline sẽ tự động scale dữ liệu rồi mới đưa vào mô hình KNN
        prediction = model_pipeline.predict(features)
        prediction_proba = model_pipeline.predict_proba(features)
        
        # 3. Hiển thị kết quả
        st.subheader("Kết quả phân tích:")
        if prediction[0] == 1.0:
            st.success(f"Dự đoán: Tích cực (Positive) 😊")
            st.write(f"Độ tin cậy: {prediction_proba[0][1]*100:.2f}%")
        else:
            st.error(f"Dự đoán: Tiêu cực (Negative) 😠")
            st.write(f"Độ tin cậy: {prediction_proba[0][0]*100:.2f}%")
            
    else:
        st.warning("Vui lòng nhập một câu tweet để phân tích.")