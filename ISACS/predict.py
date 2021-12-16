import cv2
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
import datetime
import csv

scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("ISACS/creds.json",scope)
client = gspread.authorize(creds)
sheet = client.open("test").sheet1
sheet2 = client.open("test").get_worksheet(1)
data = sheet2.get_all_records()

nameDict = {} # name list
come = {} # check if come
currentRow = 2
mycsv = csv.reader(open("ISACS/labels.csv"))
for row in mycsv:
    text = row[1]
    if(text == "name"):
        continue
    id = row[0]
    nameDict[text] = id
    come[id] = [0, currentRow]
    currentRow += 1

# update date
nextcell = int(sheet.cell(1,1).value)
lateName = ""
now = datetime.datetime.now()
today = abs(int(sheet.cell(1,nextcell-1).value[-2:]))
if(today!=now.day):
    date = str(now.year) + '-' + str(now.month) + '-' + str(now.day)
    sheet.update_cell(1,nextcell,date)
    sheet2.update_cell(1,nextcell-1,date)
    for i in range(2,2+len(nameDict)):
        sheet2.update_cell(i,nextcell-1,'-')
        sheet.update_cell(i,nextcell,0)
        come[i] = 0
    nextcell += 1
    sheet.update_cell(1,1,nextcell)

face_model = cv2.face.LBPHFaceRecognizer_create()
face_model.read('ISACS/facemodel.xml')
casc_file = "ISACS/haarcascade_frontalface_default.xml"
scale = 0.5
feature_size = (96,96)
label_file = 'ISACS/labels.csv'
df = pd.read_csv(label_file)
y_label = df.name
frontal_face = cv2.CascadeClassifier(casc_file)
faces = []
cap = cv2.VideoCapture(0)
if(cap.isOpened()==False):
    print("Failed to open the webcam")

def detect_faces(image):
    boxes = frontal_face.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
    return boxes
frame_counter = 0
while(cap.isOpened()):
    _, frame = cap.read()
    if _ == True:
        color_image = frame
        frame_counter += 1
        if frame_counter >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        color_image = cv2.resize(color_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if(frame_counter % 2) == 0:
            gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            boxes = detect_faces(gray_frame)
            for box in boxes:
                (p,q,r,s) = box
                cv2.rectangle(color_image, (p,q), (p+r,q+s), (25,255,25), 2)
                crop_image = gray_frame[q:q+s, p:p+r]
                crop_image = cv2.resize(crop_image, feature_size)
                [pred_label, pred_conf] = face_model.predict(crop_image)
                print(pred_conf)
                if(pred_conf<=100):
                    box_text = y_label[pred_label][:10]
                    txt_color = (100,0,215)
                    cv2.putText(color_image, box_text, (p+4, q-4), cv2.FONT_HERSHEY_PLAIN, 0.8, txt_color, 1)
                    if(box_text==lateName):
                        reFrame += 1
                    else:
                        reFrame = 0
                    if(reFrame >= 3):
                        currentId = int(nameDict[box_text])
                        if(come[str(currentId)][0] == 0):
                            sheet.update_cell(come[str(currentId)][1], nextcell-1, 1)
                            come[str(currentId)][0] = 1
                    lateName = box_text
                else:
                    cv2.putText(color_image, "Unknown", (p+4, q-4), cv2.FONT_HERSHEY_PLAIN, 0.8, (100,0,215), 1)
            cv2.imshow("ISACS", color_image)
        if cv2.waitKey(50) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


