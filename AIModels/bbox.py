import cv2
import torch
import numpy as np
from numpy import random
from AIModels.PretrainedModel.models.experimental import attempt_load
from AIModels.PretrainedModel.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from AIModels.PretrainedModel.utils.plots import plot_one_box
from AIModels.PretrainedModel.utils.torch_utils import select_device, time_synchronized

# letterbox
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratio, (dw, dh)


# set configuration
def set_config(classes_to_filter):
    opt = {

        "weights": "AIModels/PretrainedModel/weights/yolov7.pt",  # Path to weights file default weights are for nano model
        "yaml": "data/coco.yaml",
        "img-size": 640,  # default image size
        "conf-thres": 0.75,  # confidence threshold for inference.
        "iou-thres": 0.75,  # NMS IoU threshold for inference.
        "device": '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
        "classes": classes_to_filter  # list of classes to filter or None
    }
    return opt


# resize the original image
def resize_img(img_pth, model_name, size=(640, 640)):
    img_path = []
    for i, pic in enumerate(img_pth):
        base_pic = np.zeros((size[1], size[0], 3), np.uint8)
        base_pic.fill(248.)
        pic1 = cv2.imread(pic)
        h, w = pic1.shape[:2]
        ash = size[1] / h
        asw = size[0] / w
        if asw < ash:
            sizeas = (int(w * asw), int(h * asw))
        else:
            sizeas = (int(w * ash), int(h * ash))
        pic1 = cv2.resize(pic1, dsize=sizeas)
        base_pic[int(size[1] / 2 - sizeas[1] / 2):int(size[1] / 2 + sizeas[1] / 2),
        int(size[0] / 2 - sizeas[0] / 2):int(size[0] / 2 + sizeas[0] / 2), :] = pic1
        name = 'AIModels/yolov7/data/' + model_name + '/test/images/image_re'+str(i+1)+'.PNG'
        img_path.append(name)
        cv2.imwrite(name, base_pic)
    return img_path


# get bbox label
def get_box_label(model_name, IMAGE_NUM):
    opt = set_config(['person'])

    # 원본 이미지 경로
    source_image_path = []
    for i in range(IMAGE_NUM):
        source_image_path.append('AIModels/yolov7/data/'+model_name+'/test/images/image'+str(i+1)+'.PNG')

    # 리사이징 된 이미지 경로
    source_image_path = resize_img(source_image_path, model_name=model_name)

    # 박스 좌표
    box_label = []
    img_arr = []

    for img_path in source_image_path:
        with torch.no_grad():
            weights, imgsz = opt['weights'], opt['img-size']
            set_logging()
            device = select_device(opt['device'])
            half = device.type != 'cpu'
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size
            if half:
                model.half()

            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

            img0 = cv2.imread(img_path)
            img = letterbox(img0, imgsz, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]
            # Apply NMS
            classes = None
            if opt['classes']:
                classes = []
                for class_name in opt['classes']:
                    classes.append(names.index(class_name))

            if classes:
                classes = [i for i in range(len(names)) if i in classes]

            pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
            t2 = time_synchronized()
            for i, det in enumerate(pred):
                s = ''
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                if len(det):

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, color=colors[0], line_thickness=3)
                        img_arr.append(img0)
                        box_label.append((xyxy, names[int(cls)]))  # 바운딩 박스 좌표 저장
    return box_label, img_arr


def normalize(box_label):
    for i, label in enumerate(box_label):
        x_mid, y_mid = (label[0][0] + label[0][2]) / 2, (label[0][1] + label[0][3]) / 2
        x_mid, y_mid = torch.round(x_mid) / 640, torch.round(y_mid) / 640
        width, height = (label[0][2] - label[0][0]), (label[0][3] - label[0][1])
        width, height = width / 640, height / 640
        box_label[i] = ([x_mid, y_mid, width, height], 'person')
    return box_label


def write_into_txt(box_label, model_name):
    # writedata.py
    f = open('AIModels/yolov7/data/' + model_name + '/test/labels/image_re'+'.txt', 'w')
    for i, label in enumerate(box_label):
        f = open(f'AIModels/yolov7/data/' + model_name + '/test/labels/image_re'+str(i+1)+'.txt', 'w')
        x_mid, y_mid, width, height = label[0][0], label[0][1], label[0][2], label[0][3]
        data = f'0 {x_mid} {y_mid} {width} {height}\n'
        f.write(data)
        f.close()

def bbox(model_name, IMAGE_NUM):
    bbox_label, img_arr = get_box_label(model_name, IMAGE_NUM)
    bbox_label = normalize(bbox_label)
    write_into_txt(bbox_label, model_name)

# for i in img_arr:
#   cv2_imshow(i)
