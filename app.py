from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/detect", methods=["POST"])
def detect():
    review = request.json["review"]
    review_clean = re.sub(r'\W+', ' ', review.lower())
    review_vectorized = vectorizer.transform([review_clean])

    prediction = model.predict(review_vectorized)[0]
    result = "Fake Review" if prediction == 1 else "Genuine Review"
    
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
