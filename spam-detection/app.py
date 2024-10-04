from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/spam-detection')
def spam_detection():
    return render_template('spam_detection.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        data = request.get_json()
        message = data['message']
        
     
        message_features = vectorizer.transform([message])
  
        prediction = model.predict(message_features)
        
        
        result = 'Ham' if prediction[0] == 1 else 'Spam' 

        return jsonify(prediction=result)
    except Exception as e:
        return jsonify(prediction="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
