from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/ai', methods=['POST'])
def submit():
    files = request.files
    images = []
    for i in range(1, 5):
        images.append(files['image'+str(i)])

    # model = learning(images)
    # result = predict(model)
    result = None
    return jsonify({"result": result})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port='5000', threaded=True)