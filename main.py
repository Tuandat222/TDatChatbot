import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
import string

# Dữ liệu mẫu cho chatbot
data = {
    'question': [
        "What are your opening hours?",
        "How do I reset my password?",
        "Where is your office located?",
        "What is your contact number?",
        "Can I change my email address?"
    ],
    'intent': [
        "opening_hours",
        "reset_password",
        "office_location",
        "contact_number",
        "change_email"
    ],
    'response': [
        "Our opening hours are from 9 AM to 5 PM.",
        "To reset your password, click on 'Forgot Password' on the login page.",
        "Our office is located at 123 Main Street, City.",
        "You can contact us at 123-456-7890.",
        "You can change your email address in your account settings."
    ]
}

# Chuyển đổi thành DataFrame
df = pd.DataFrame(data)

# Tiền xử lý dữ liệu
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    # Chuyển sang chữ thường
    text = text.lower()
    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Loại bỏ từ dừng
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['cleaned_question'] = df['question'].apply(preprocess_text)

# Tạo vector cho câu hỏi
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_question'])
y = df['intent']

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sử dụng Logistic Regression làm mô hình phân loại
model = LogisticRegression()
model.fit(X_train, y_train)

# Đánh giá mô hình
#accuracy = model.score(X_test, y_test)
#print(f"Model accuracy: {accuracy * 100:.2f}%")

# Kiểm tra dữ liệu
#print("Training data (X_train):", X_train.toarray())
#print("Training labels (y_train):", y_train.tolist())
#print("Cleaned questions:", df['cleaned_question'].tolist())

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json['question']
    cleaned_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    predicted_intent = model.predict(input_vector)[0]

    # Lấy câu trả lời tương ứng từ DataFrame
    response = df[df['intent'] == predicted_intent]['response'].values[0]

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
