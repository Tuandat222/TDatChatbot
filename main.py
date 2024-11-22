import pandas as pd
import string

from underthesea import word_tokenize  # Import tokenization từ underthesea
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
from flask_cors import CORS

# Lấy dữ liệu từ data
data = pd.read_csv('data.csv')

# Đọc danh sách stopwords tiếng Việt
with open('vietnamese_stopwords.txt', 'r', encoding='utf-8') as f:
    vietnamese_stopwords = set(f.read().splitlines())


# Tiền xử lý dữ liệu
def preprocess_text(text):
    text = text.lower()  # Chuyển sang chữ thường
    text = text.translate(str.maketrans('', '', string.punctuation))  # Loại bỏ dấu câu
    tokens = word_tokenize(text)  # Sử dụng underthesea để tách từ
    tokens = [word for word in tokens if word not in vietnamese_stopwords]  # Loại bỏ từ dừng
    return ' '.join(tokens)


# Tiền xử lý cột câu hỏi
data['cleaned_question'] = data['question'].apply(preprocess_text)

# Tạo vector cho câu hỏi
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_question'])
y = data['intent']

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LogisticRegression()
model.fit(X_train, y_train)

# Khởi tạo FLASK
app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return "Đã chạy Chatbot API."


@app.route('/ask', methods=['POST'])
def ask():
    if not request.is_json:
        return jsonify({"error": "Unsupported Media Type. Use application/json format"}), 415

    user_input = request.json.get('question')
    if not user_input:
        return jsonify({"error": "Question not provided in the request"}), 400

    cleaned_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    predicted_intent = model.predict(input_vector)[0]

    # Trả lời từ dữ liệu
    response = data[data['intent'] == predicted_intent]['response'].values
    if len(response) > 0:
        response = response[0]
    else:
        response = "Sorry, I couldn't understand your question. Can you rephrase it?"

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
