import cv2
import os

output_path = "./ISACS/img_features_label/"
GD_output_path = './ISACS_django/statics/labeled_images/' #output path for the django website

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error to create directory " + directory)

def cropImage(img,box,name):
    [p,q,r,s] = box
    write_img_color = img[q:q+s,p:p+r]
    saveCropped(write_img_color,name)

img_pixel = 96
numFaces = 1
def saveCropped(img,name):
    global numFaces
    img = cv2.resize(img,(img_pixel,img_pixel),interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path + name + '/' + str(numFaces) + ".jpg",img)
    numFaces += 1

numU = 1
def saveGDimg(img,name):
    global numU
    cv2.imwrite(GD_output_path + name + '/' + str(numU) + '.jpg',img)
    numU += 1

name = "Jaksawat2"
directory = "./ISACS/img_features_label/" + name
directory2 = "./ISACS_django/statics/labeled_images/" + name
createFolder(directory)
createFolder(directory2)

casc_file = "./ISACS/haarcascade_frontalface_default.xml"
frontal_face = cv2.CascadeClassifier(casc_file)

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    _,frame = cap.read()
    cv2.imshow("frame",frame)
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    boxes = frontal_face.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30),)
    for box in boxes:
        cropImage(gray_img, box, name)
        if(numU<11):
            saveGDimg(frame,name)
    if(numFaces==31):
        break
    if cv2.waitKey(10) & 0xFF == 27: 
        break
cap.release()
cv2.destroyAllWindows()