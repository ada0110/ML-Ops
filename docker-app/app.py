from flask import Flask
from flask import request
app = Flask(__name__)

cnt_visit = 0

@app.route('/')
def hello_world():
    global cnt_visit
    cnt_visit += 1
    return f'Hello, Docker! [visit count: {cnt_visit}]\n'
    

# http://localhost:5000/svm_predict/?input=this is good
# curl -G "http://localhost:5000/svm_predict/" --data-urlencode "input=hello world"
@app.route('/svm_predict/')
def svm_predict():
    input_ = request.args.get('input')
    return f"called svm_predict with input: {input_}\n"


# call this using curl
@app.route("/svm_predict_post", methods=['POST'])
def svm_predict_post():
    image = request.json["image"]

    # image = np.array(image).reshape(1, -1)
    # prediction = clf.predict(image)
    # print("image:\n", image)
    # print("prediction:", prediction[0], end="\n\n")
    # return str(prediction[0])

    return f"called svm_predict_post with input: {image}\n"
    

if __name__ == '__main__':
    # it won't be accessible at http://localhost:5000/
    # 0.0.0.0 is just a placeholder. We need to figure out which address it is actually listening on. If we are trying to reach a server from the same machine, try http://localhost:5000/
    app.run(debug=True, host='0.0.0.0')