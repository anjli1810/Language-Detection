from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
x = data["Text"]
y = data["language"]

# Feature extraction
cv = CountVectorizer()
X = cv.fit_transform(x)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['text']
        data = cv.transform([user_input]).toarray()
        prediction = model.predict(data)
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
