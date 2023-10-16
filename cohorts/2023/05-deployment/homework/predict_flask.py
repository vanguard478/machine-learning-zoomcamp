import pickle
from flask import Flask
from flask import request
from flask import jsonify
'''The /app folder has the following structure:
/app# tree .
.
├── Pipfile
├── Pipfile.lock
├── dv.bin
├── model2.bin
└── predict.py
'''
model_file = 'model2.bin'
dv_file = 'dv.bin'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)
# model2.bin  predict.py
app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    # customer = {'job': 'retired', 'duration': 445, 'poutcome': 'success'}
    customer = request.get_json()
    feature_x = dv.transform([customer])
    y_pred = model.predict_proba(feature_x)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
