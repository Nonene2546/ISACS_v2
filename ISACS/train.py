import cv2
import numpy as np
import pandas as pd
import os

train_path = "./ISACS/img_features_label/"
# input_path = './'
label_file = './ISACS/labels.csv'
django_label_file = './ISACS_django/statics/labels.csv'


img_pixel = (96,96)

def save_labels(people):
    df = pd.DataFrame(people, columns=['name'])
    df.to_csv(label_file)
    df.to_csv(django_label_file)

def get_images(path, size):
    class_id = 0
    images, labels = [], []
    people = []
    for subdir in os.listdir(path):
        for image in os.listdir(path + subdir):
            img = cv2.imread(path+os.path.sep+subdir+os.path.sep+image, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, size)
            images.append(np.asarray(img, dtype = np.uint8))
            labels.append(class_id)
        people.append(subdir)
        class_id += 1
    return [images, labels, people]

def train_model():
    [images, labels, people] = get_images(train_path, img_pixel)
    labels = np.asarray(labels, dtype=np.int32)
    face_model = cv2.face.LBPHFaceRecognizer_create()
    face_model.train(images, labels)
    return [face_model, people]

if __name__ == "__main__":
    face_model, people = train_model()
    face_model.write('./ISACS/facemodel.xml')
    save_labels(people)
