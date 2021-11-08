from flask import request
from flask import jsonify
from flask import Flask, render_template
from sentiment import get_sentiment

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    label = get_sentiment(text)
    return(render_template('index.html', variable=label))

if __name__ == "__main__":
    app.run(port='9999', threaded=False, debug=True)