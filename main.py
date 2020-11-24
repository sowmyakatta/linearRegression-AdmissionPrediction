from flask import Flask, render_template, request
import pickle
from flask_cors import cross_origin, CORS

app = Flask(__name__)
CORS(app)

@app.route("/", methods =["GET", "POST"])
@cross_origin()
def index():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        GRE = float(request.form["GRE"])
        URAT = float(request.form["URAT"])
        TOEFL = float(request.form["TOEFL"])
        SOP = float(request.form["SOP"])
        LOR = float(request.form["LOR"])
        CGPA = float(request.form["CGPA"])
        Research = float(request.form["Research"])
    model = pickle.load(open("my_lr.pickle", "rb"))
    res =model.predict([[GRE, TOEFL, URAT, SOP,LOR,CGPA, Research]])
    r ="Admission prediction " + str(res[0])
    return ("<h1>" + r +"</h1>")


if __name__ == "__main__":
    app.run(port=8000, debug=True)
