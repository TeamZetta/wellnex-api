
from flask import Flask, request, jsonify
# from pymongo import MongoClient
import aiutils
import os
import json

app = Flask(__name__)

config_file = 'config.json'
if not os.path.isfile(config_file):
    app.logger.error(
        "Your config.json file is missing." +
        "You need to create one in order to run this app."
    )
    exit()
else:
    config = json.load(open(config_file))
    # print(config)


@app.route("/test", methods=['GET'])
def test():
    return {'message': 'working /test route'}

# @app.route("/login", method=["POST"])
# def login():
#     data = request.json


@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    usertext = data['text']
    report = aiutils.diagnosis(usertext)
    return report

if __name__ == '__main__':
    try:
        # Set up MongoDB connection
        # client = MongoClient(config['MONGODB_URI'])
        # db = client['wellnex']
        # collection = db['users']

        # print("Mongodb connected")

        app.run(debug=True, port=5000)
    except Exception as e:
        app.logger.error(str(e))
        exit()
