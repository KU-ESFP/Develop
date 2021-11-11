import os
import cv2

# Face detection: Load classifiers stored in XML format
face_cascade = cv2.CascadeClassifier('../trained_models/detection_models/haarcascade_frontalface_default.xml')

direction = './neutral_face'
for image_name in os.listdir(direction):
    img = cv2.imread(os.path.join(direction, image_name))

    # Face detection within the image
    face = face_cascade.detectMultiScale(img, 1.3, 5)

    list_face_detected = []
    for (x, y, w, h) in face:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_boundary = img[y:y + h, x:x + w]

        img_48x48 = cv2.resize(face_boundary, (48, 48), interpolation=cv2.INTER_LINEAR)
        img_gray = cv2.cvtColor(img_48x48, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('./new_angry/%s' % 'ii'+ image_name, img_gray)
        print(image_name)


# music_test data 700-730개 정도는 있어야함
# angry(403), happy(751), neutral(412), sad(331), surprised(421)
path = 'test/sad'
file_list = os.listdir(path)
print(len(file_list))