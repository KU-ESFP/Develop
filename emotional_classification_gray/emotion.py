import sys

import cv2
import torch
import torchvision.transforms as transforms

from PIL import Image
from src.models.model import LeNet

# Face detection: Load classifiers stored in XML format
face_cascade = cv2.CascadeClassifier('./trained_models/detection_models/haarcascade_frontalface_default.xml')

# Model load
classes = ['angry', 'happy', 'neutral', 'sad', 'surprised']
# emotion_classifier = torch.load(emotion_model_path)
model = LeNet(num_classes=5)
path = "trained_models/emotion_models/new_model3.pth"
model.load_state_dict(torch.load(path))
model.eval()

# Image to detect face (RGB to Gray)
img = cv2.imread('input_images/img_1.jpg')
if img is None:
    print('[ERROR] IMAGE IS NONE')
    exit()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Face detection within the image
face = face_cascade.detectMultiScale(img_gray, 1.3, 5)

# 얼굴이 검출되면 검출된 얼굴 감정들 담기
list_face_detected = []
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face_boundary = img_gray[y-50:y+h+50, x-50:x+w+50]

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    img_pil_face = Image.fromarray(face_boundary)
    img_trans = transform(img_pil_face)
    img_trans = img_trans.unsqueeze(0)                      # Add batch dimension

    output = model(img_trans)                               # Forward pass
    pred = torch.argmax(output, 1)                          # Get predicted class if multi-class classification
    # print('Image predicted as', classes[pred])              # Output: Emotion class for the face
    list_face_detected.append(classes[pred])
    cv2.putText(img, classes[pred], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# 이미지 확인할 때 사용
# cv2.imshow('detect', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 얼굴 검출에 따른 결과
if len(list_face_detected) == 1:
    print('[SUCCESS] ONE FACE DETECTED')
    print(list_face_detected[0])
elif len(list_face_detected) == 0:
    print('[ERROR] FACE IS NOT DETECTED')
else:
    print('[ERROR] A LOT OF FACES ARE DETECTED')

