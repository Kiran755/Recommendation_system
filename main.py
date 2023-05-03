from flask import Flask
from recommend import give_recommendations
from flask import jsonify
app = Flask(__name__)


@app.route('/book/<string:name>')
def recommend(name):
    # name = "Computer Algorithms"
    return jsonify(give_recommendations(name,True))


if __name__=="__main__":
    app.run(debug=True,host="192.168.0.104",port=5000)

