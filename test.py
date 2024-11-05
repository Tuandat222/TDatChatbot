import nltk

# Kiểm tra tải các gói cần thiết
nltk.download('punkt')
nltk.download('stopwords')

# Kiểm tra tokenization với một câu mẫu
from nltk.tokenize import word_tokenize
sample_text = "Hello! How can I help you today?"
tokens = word_tokenize(sample_text)
print("Tokenized words:", tokens)

# Kiểm tra từ dừng
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Filtered tokens (without stop words):", filtered_tokens)
