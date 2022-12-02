import albumentations as A
import cv2
import numpy as np
import os
import torch

IMAGE_NUM = 8

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


def startaug(image, bboxes, train_index, valid_index, num):
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
            index = valid_index
        else:
            dataset_type = "train"
            index = train_index

        # 이미지 저장
        save_img_path = "yolov7/data/" + dataset_type + "/images/"
        img_name = "augmented_img" + str(j + index) + ".PNG"
        save_img_path = save_img_path + img_name
        cv2.imwrite(save_img_path, transformed_image)

        # label 저장
        text_name = "yolov7/data/"+dataset_type+"/labels" + \
                    "augmented_img" + str(j + index) + ".txt"
        with open(text_name, "w") as f:
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


def augmentation():
    img_path = "yolov7/data/test/images/"
    label_path = "yolov7/data/test/labels/"

    images = []
    bboxes = []

    for i in range(IMAGE_NUM):
        file_name = 'image'+str(i+1)+'.PNG'
        image = cv2.imread(img_path + file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = np.loadtxt(fname=label_path + file_name, delimiter = " ", ndmin = 2)
        bbox = np.roll(bbox, 4, axis=1).tolist()
        images.append(image)
        bboxes.append(bbox)

    train_idx = 0
    num = 200
    val_idx = 8 * int(num * 0.8)

    for i in range(IMAGE_NUM - 1):
        startaug(image[i], bboxes[i], train_idx, val_idx, num)
        train_idx = train_idx + int(num * 0.8)
        val_idx = val_idx + int(num * 0.2)
    startaug(image[-1], bboxes[-1], train_idx, val_idx, num)

    os.system('python yolov7/train.py --device 0 --batch 16 --epochs 25 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7_custom.yaml --weights')
    os.system('python yolov7/detect.py --weights runs/train/exp35/weights/best.pt --conf 0.6 --iou 0.6 --source test_new_cloth.mp4 --name new_cloth')

if __name__ == '__main__':
    augmentation()
