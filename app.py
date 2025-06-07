from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
CORS(app)

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only necessary columns and clean
data = data[['Category', 'Message']]
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Split data
X = data['Message']
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizer and model training
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # Vectorize input message using the same vectorizer
    message_vec = vectorizer.transform([message])

    # Predict spam or not spam
    prediction = model.predict(message_vec)[0]

    return jsonify({'result': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
