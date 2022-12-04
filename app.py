import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
	return render_template('form.html')

@app.route('/form',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    scaler = StandardScaler()
    scaler.fit(final_features)
    final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)

    output = prediction[0]

    def message(output):
        if output == 1:
            return 'Patient will most likely have a longer stay (more than 3 days)'
        else:
            return 'Patient will most likely have a shorter stay (less than 3 days)'

    return render_template('form.html', prediction_text=message(output))

if __name__ == "__main__":
    app.run(debug=True)
