from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    # load current count
    with open("static/count.txt", "r") as f:
        count = int(f.read())

    # increment count
    count += 1

    # overwrite the count
    with open("static/count.txt", "w") as f:
        f.write(str(count))

    # render the HTML with count variable
    return render_template("index.html", count=count)


if __name__ == "__main__":
    app.run(debug=True)

