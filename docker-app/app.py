from flask import Flask
from flask import request
import numpy as np
import pickle

app = Flask(__name__)
cnt_visit = 0


best_model_svm = 'saved_models/best_model_svm.pkl'
best_model_decision = 'saved_models/best_model_decision.pkl'

def load_model(path):
    print("\nloading the model...")
    load_file = open(path, "rb")
    loaded_model = pickle.load(load_file)
    return loaded_model


# load models
clf_svm = load_model(best_model_svm)
print("***svm model loaded***")

clf_decision = load_model(best_model_svm)
print("***decision tree model loaded***\n")


@app.route('/')
def hello_world():
    global cnt_visit
    cnt_visit += 1
    return f'Hello, Docker! [visit count: {cnt_visit}]\n'
    

# http://localhost:5000/svm_predict/?input=this is good
# curl -G "http://localhost:5000/svm_predict/" --data-urlencode "input=hello world"
@app.route('/svm_predict_get/')
def svm_predict_get():
    input_ = request.args.get('input')
    return f"called svm_predict with input: {input_}\n"


# call this using curl
@app.route("/svm_predict/", methods=['POST'])
def svm_predict():
    image = request.json["image"]

    image = np.array(image).reshape(1, -1)
    prediction = clf_svm.predict(image)
    print("input image:\n", image)
    return f"svm prediction: {str(prediction[0])}\n\n"


# call this using curl
@app.route("/decision_tree_predict/", methods=['POST'])
def decision_tree_predict():
    image = request.json["image"]

    image = np.array(image).reshape(1, -1)
    prediction = clf_decision.predict(image)
    print("input image:\n", image)
    return f"decision tree prediction: {str(prediction[0])}\n\n"
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')