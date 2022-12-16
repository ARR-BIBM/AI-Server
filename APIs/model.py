from flask import jsonify, abort
from werkzeug.datastructures import FileStorage
from flask_restx import Namespace, Resource
import os
import shutil
from AIModels.augmentation import augmentation, yolo_train
from AIModels.bbox import bbox

IMAGE_NUM_FOR_MODEL = 8

ALLOWED_EXTENSIONS_IMG = ['png']
ALLOWED_EXTENSIONS_VIDEO = ['mp4']

# api name
model_api = Namespace('model_api', description='광고 이미지 받아 모델 생성')

# api request rule
model_parser = model_api.parser()
model_parser.add_argument('uid', location='form', type=str, required=True)
model_parser.add_argument('model_idx', location='form', type=int, required=True)
for i in range(1, IMAGE_NUM_FOR_MODEL+1):
    model_parser.add_argument('image'+str(i), location='files', type=FileStorage, required=True)

# api function
@model_api.route('/')
@model_api.expect(model_parser)
class ModelAPI(Resource):

    @model_api.response(200, 'model create success')
    def post(self):
        args = model_parser.parse_args()
        uid = args['uid']
        model_idx = args['model_idx']
        model_name = uid + '_' + str(model_idx)

        self.make_directory(model_name)

        filepath = "AIModels/yolov7/data/" + model_name + '/test/images/'
        for i in range(1, IMAGE_NUM_FOR_MODEL + 1):
            if not 'image'+str(i) in args:
                abort(400, "Missing file : image"+str(i))
            file = args['image'+str(i)]
            file.save(filepath + 'image'+str(i)+'.PNG')

        bbox(model_name, IMAGE_NUM_FOR_MODEL)
        augmentation(model_name, IMAGE_NUM_FOR_MODEL)
        yolo_train(model_name)
        self.remove_directory(model_name)

        return jsonify()


    def make_directory(self, model_name):
        filepath = "AIModels/yolov7/data/" + model_name
        ttvs = ['train', 'test', 'val']
        for ttv in ttvs:
            os.makedirs(filepath + '/' + ttv, exist_ok=True)
            os.makedirs(filepath + '/' + ttv + '/images', exist_ok=True)
            os.makedirs(filepath + '/' + ttv + '/labels', exist_ok=True)

    def remove_directory(self, model_name):
        filepath = "AIModels/yolov7/data/" + model_name
        shutil.rmtree(filepath)
        os.remove(filepath+'.yaml')

