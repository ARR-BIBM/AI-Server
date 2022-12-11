from flask import jsonify, abort
from werkzeug.datastructures import FileStorage
from flask_restx import Namespace, Resource
from AIModels.augmentation import yolo_detect
from AIModels.total_detect import total_detect
import os

advideo_api = Namespace('upload_video', description='광고 동영상 받아서 detect')

advideo_parser = advideo_api.parser()
advideo_parser.add_argument('video', location='files', type=FileStorage, required=True)
advideo_parser.add_argument('uid', location='form', type=str, required=True)
advideo_parser.add_argument('model_idx', location='form', type=int, required=True)
advideo_parser.add_argument('ad_idx', location='form', type=int, required=True)

# api function

@advideo_api.route('/')
@advideo_api.expect(advideo_parser)
class Upload_video(Resource):
    """
        유저에게 video를 받는 api
    """
    @advideo_api.response(200, 'upload Success')
    def post(self):
        args = advideo_parser.parse_args()
        if not 'video' in args:
            abort(400, "Missing file : video")
        video = args['video']
        uid = args['uid']
        model_idx = args['model_idx']
        ad_idx = args['ad_idx']

        ad_name = uid + '_' + str(ad_idx)
        model_name = uid + '_' + str(model_idx)

        filename = 'AIModels/yolov7/' + ad_name +'.mp4'
        video.save(filename)

        yolo_detect(ad_name, model_name)
        cnt, fps = total_detect(ad_name)
        # os.remove(filename)

        return jsonify({'cnt':int(cnt//fps)})
