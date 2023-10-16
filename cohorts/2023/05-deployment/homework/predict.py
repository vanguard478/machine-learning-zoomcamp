import pickle
model_file = 'homework/model1.bin'
dv_file = 'homework/dv.bin'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)


def predict():
    customer = {'job': 'retired', 'duration': 445, 'poutcome': 'success'}
    feature_x = dv.transform([customer])
    y_pred = model.predict_proba(feature_x)[0, 1]
    return y_pred

print(f'Probability of churn: {round(predict(),3)}')
