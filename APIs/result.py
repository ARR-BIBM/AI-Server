from flask import send_file, abort
from flask_restx import Namespace, Resource
import os
# api name
result_api = Namespace('result_api', description='결과 동영상 반환. mode : detect or total')

# api function
@result_api.route('/<uid>/<mode>/<ad_num>')
class ModelAPI(Resource):

    @result_api.response(200, 'model create success')
    def get(self, uid, mode, ad_num):
        ad_name = uid+"_"+str(ad_num)
        file_name = f"runs/"+mode+"/" + ad_name+'_'+mode+'.mp4'
        if not os.path.isfile(file_name):
            abort(400, "no such file")
        return send_file(file_name, mimetype='video/mp4', as_attachment=True)

