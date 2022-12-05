from flask import Flask
from flask_cors import CORS
from flask_restx import Api
from APIs.model import model_api
from APIs.advideo import advideo_api

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='API 문서', description='Swagger 문서', doc="/api-docs")

api.add_namespace(model_api, '/model')
api.add_namespace(advideo_api, '/advideo')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)

