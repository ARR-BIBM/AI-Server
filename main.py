from flask import Flask, jsonify
from flask_cors import CORS
from flask_restx import Namespace, Resource, fields, Api
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='API 문서', description='Swagger 문서', doc="/api-docs")

IMAGE_NUM_FOR_MODEL = 4
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

make_model_api = Namespace('makemodel', description='광고 이미지로 모델 만들기')
make_model_parser = make_model_api.parser()
for i in range(1, IMAGE_NUM_FOR_MODEL + 1):
    make_model_parser.add_argument('image'+str(i), location='files', type=FileStorage, required=True)

make_model_field = make_model_api.model('make_model', {
    'make_model': fields.String
})

@make_model_api.route('/')
@make_model_api.expect(make_model_parser)
class Make_Model(Resource):

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @make_model_api.response(201, 'Success', make_model_field)
    def post(self):
        args = make_model_parser.parse_args()
        images = []
        for i in range(1, IMAGE_NUM_FOR_MODEL + 1):
            file = args['image'+str(i)]
            if file and self.allowed_file(file.filename):
                images.append(file)

        # upload_db(images)

        return jsonify({"make_model": "success"}), 201

api.add_namespace(make_model_api)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port='5000', threaded=True)