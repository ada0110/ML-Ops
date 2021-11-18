from logging import debug
from flask import Flask, request
# from src.utils import load_model
import numpy as np
import pickle

app = Flask(__name__)

best_model_path = '/home/ada/codes/ML-Ops_Scikit/models/best_clf_0.6715.pkl'

def load_model(path):
    print("\nloading the model...")
    load_file = open(path, "rb")
    loaded_model = pickle.load(load_file)
    return loaded_model


@app.route("/")
def hello_world():
    print("Server started!!")
    return "<p>Hello, World!</p>"


# curl http://localhost:5000/predict -X POST  -H 'Content-Type: application/json' -d '{"image": ["1.0", "2.0", "3.0"]}'

@app.route("/predict", methods=['POST'])
def predict():
    clf = load_model(best_model_path)
    print("model loaded\n")

    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)

    prediction = clf.predict(image)
    print("image:", image)
    print("prediction:", prediction[0])

    return str(prediction[0])
    # return f'<p>image: {image}<p><br>'
    

if __name__ == '__main__':
    app.run(debug=True)

'''
curl http://localhost:5000/predict -X POST  -H 'Content-Type: application/json' -d '{"image":
["0.0","0.0","0.0","2.000000000000008","12.99999999999999","2.3092638912203262e-14","0.0","0.0","0.0","0.0","0.0","7.99999999999998","14.999999999999988","2.664535259100375e-14","0.0","0.0","0.0","0.0","4.9999999999999885","15.999999999999975","5.000000000000027","2.0000000000000027","3.552713678800496e-15","0.0","0.0","0.0","14.999999999999975","12.000000000000007","1.0000000000000182","15.999999999999961","4.000000000000018","7.1054273576009955e-15","3.5527136788004978e-15","3.9999999999999925","15.999999999999984","2.0000000000000275","8.999999999999984","15.999999999999988","8.00000000000001","1.4210854715201997e-14","3.1554436208840472e-30","3.5527136788004974e-15","9.999999999999995","13.999999999999986","15.99999999999999","16.0","4.000000000000025","7.105427357601008e-15","0.0","0.0","0.0","0.0","12.999999999999982","8.000000000000009","1.4210854715202004e-14","0.0","0.0","0.0","0.0","0.0","12.999999999999982","6.000000000000012","1.0658141036401503e-14","0.0"]}'
'''
