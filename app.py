from flask import Flask, request, jsonify, make_response
from bson.json_util import dumps
import json
from pymongo import MongoClient
from flask_cors import CORS
from .model_outputs.model_price_output import predict_price
from .model_outputs.model_product_output import find_adjacent_items
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins":"*"}})

# MongoDB connection settings
MONGO_URI = "mongodb+srv://agropuls:agropuls@cluster0.xwnwqwm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client.get_database('agropulse')
logIncoll = db['agropulse']
farmerColl = db['farmerData']

#Sign up data for the users 
@app.route('/signUp', methods = ["POST","GET"])
def signUp():
    if request.method =="POST":
        data = {
                "_id": request.json['_id'],
                "name": request.json['name'],
                "email": request.json['email'],
                "password": request.json['password']
            }
        users = logIncoll.find_one({"name": request.json['name']})
        if users == None:
            logIncoll.insert_one(data)
            return jsonify(status_code=200)
        else:
            if users['email'] == request.json['email']:
                return jsonify(status_code=401)
            elif users['name'] == request.json['name']:
                return jsonify(status_code=404)
        return jsonify(status_code=200)
    return "Error on the sign up route"

#Login data for the users
@app.route('/logIn', methods=["POST"])
def logIn():
    if request.method == "POST":
        users = logIncoll.find_one({"name": request.json['name']})
        if users != None:
            if users['password'] != request.json['password']:
                return jsonify(status_code=401)
            return json.loads(dumps(users))
        else:
            return jsonify(status_code=404)
    return "Error on the log in route"


#Post method for the farmer data
@app.route('/farmerData', methods=["POST"])
def farmerData():
    if request.method == "POST":
        farmerdata = {
                "_id": request.json["id"],
                "country": request.json['country'],
                "area": request.json['area'],
                "crop": request.json['crop']
            }
        print(farmerColl.insert_one(farmerdata))
        return "Farmer data send"
    return "Data error"
@app.route('/farmerDataGet', methods=["GET"])
def farmerDataGet():
    if request.method == "GET":
        users = farmerColl.find()
        return json.loads(dumps(users))
@app.route('/predictions', methods = ["GET"])
def predictions():
    if request.method == "GET":
        items = find_adjacent_items(2,3542,1445, 11, 20.2)
        price = str(predict_price([500], [32], str(items[0]), [0],2025,[1]))
        return json.loads(dumps({"price": price, "items": items}))
    return "Data error"
        
if __name__ == '__main__':
    app.run(debug=True)