import albumentations as A
import cv2
import numpy as np
import os

train_transform = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p = 0.5),
        A.RandomGamma(gamma_limit = (90,110)),
        A.ShiftScaleRotate(scale_limit = 0.1, rotate_limit = 10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Transpose(),
        A.RandomRotate90(),
        A.OneOf([A.NoOp(),A.MultiplicativeNoise(),A.GaussNoise(),A.ISONoise()]),
        A.OneOf(
            [
                A.NoOp(p=0.7),
                A.HueSaturationValue(hue_shift_limit=10,sat_shift_limit=10),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
            ],
            p=0.3
        )
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[])
,)


def startaug(image, bboxes, train_index, valid_index, num, model_path):
    j = 0
    for j in range(num):

        # image augmentation 완료
        transformed = train_transform(image=image, bboxes=bboxes)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        # convert RGB to BGR, bounding box 재배치

        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        transformed_bboxes = np.roll(transformed_bboxes, 1, axis=1).tolist()

        if (j >= 0.8 * num):
            dataset_type = "val"
            index = valid_index + j - int(0.8*num)
        else:
            dataset_type = "train"
            index = train_index + j

        # 이미지 저장
        save_img_path = "AIModels/yolov7/data/" + model_path + dataset_type + "/images/"
        img_name = "augmented_img" + str(int(index)) + ".PNG"
        save_img_path = save_img_path + img_name
        cv2.imwrite(save_img_path, transformed_image)

        # label 저장
        text_name = "AIModels/yolov7/data/" + model_path + dataset_type + "/labels/" + \
                    "augmented_img" + str(int(index)) + ".txt"
        with open(text_name, mode="w") as f:
            flag = 0
            for item in transformed_bboxes[0]:
                if flag != 0:
                    f.write(' ')
                    f.write(str(item))
                else:
                    f.write(str(int(item)))
                flag = flag + 1
            f.write('\n')
            f.close


def augmentation(model_name, IMAGE_NUM):
    model_path = model_name + f"/"
    img_path = "AIModels/yolov7/data/" + model_path + "test/images/"
    label_path = "AIModels/yolov7/data/" + model_path + "test/labels/"

    images = []
    bboxes = []

    for i in range(IMAGE_NUM):
        file_name = 'image_re'+str(i+1)
        image = cv2.imread(img_path + file_name + '.PNG')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = np.loadtxt(fname=label_path + file_name + '.txt', delimiter=" ", ndmin=2)
        bbox = np.roll(bbox, 4, axis=1).tolist()
        images.append(image)
        bboxes.append(bbox)

    train_idx = 0
    num = 200
    val_idx = 8 * int(num * 0.8)
    for i in range(IMAGE_NUM):
        startaug(images[i], bboxes[i], train_idx, val_idx, num, model_path)
        train_idx = train_idx + int(num * 0.8)
        val_idx = val_idx + int(num * 0.2)

    print("image augmentation done")


def yolo_train(model_name):
    # 결과 : best.pt // name 옵션으로 결과 폴더 이름 바꾸기
    data_yaml_file_name = "data/"+model_name+".yaml"
    with open("AIModels/yolov7/" + data_yaml_file_name, mode="w") as f:
        f.write('train: AIModels/yolov7/data/' + model_name + '/train\n')
        f.write('val: AIModels/yolov7/data/' + model_name + '/val\n')
        f.write('nc: 1\n')
        f.write('names: [ \'target\' ]\n')
        f.close

    os.system(
        'python AIModels/yolov7/train.py --device 0 --batch 16 --epochs 25 --name ' + model_name + ' --data ' + data_yaml_file_name + ' --img 640 640 --cfg AIModels/yolov7/cfg/training/yolov7_custom.yaml --weights \'AIModels/yolov7_training.pt\'')


def yolo_detect(model_name):
    # 결과 : .mp4
    os.system(
        'python AIModels/yolov7/detect.py --weights AIModels/yolov7/runs/train/' + model_name + '/weights/best.pt --conf 0.6 --iou 0.6 --source ' + model_name + '.mp4 -- name ' + model_name)

if __name__ == '__main__':
    augmentation("example_0", 8)
    yolo_train('example_0')

