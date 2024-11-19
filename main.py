import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
from flask_cors import CORS
import string

# Tải dữ liệu từ tệp CSV
df = pd.read_csv('data.csv')

# Tiền xử lý dữ liệu
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['cleaned_question'] = df['question'].apply(preprocess_text)

# Tạo vector cho câu hỏi
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_question'])
y = df['intent']

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LogisticRegression()
model.fit(X_train, y_train)

# Đánh giá mô hình
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Khởi tạo Flask
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the Chatbot API! Use the '/ask' endpoint to ask questions."

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

    response = df[df['intent'] == predicted_intent]['response'].values
    if len(response) > 0:
        response = response[0]
    else:
        response = "Sorry, I couldn't understand your question. Can you rephrase it?"

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)