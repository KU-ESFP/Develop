import cv2
import torch
import torchvision.transforms as transforms

from src.models.cnn import MiniXception
from src.train_emotion_classifier import num_classes

# Face detection: Load classifiers stored in XML format
face_cascade = cv2.CascadeClassifier('./trained_models/detection_models/haarcascade_frontalface_default.xml')

# Model load
emotion_labels = ['angry', 'happy', 'neutral', 'sad', 'surprised']
# emotion_classifier = torch.load(emotion_model_path)
model = MiniXception(num_classes)
path = "trained_models/emotion_models/new_model1.pth"
model.load_state_dict(torch.load(path))
model.eval()

# Image to detect face (RGB to Gray)
img = cv2.imread('input_images/img_1.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Face detection within the image
face = face_cascade.detectMultiScale(img_gray, 1.3, 5)

# 얼굴이 검출되었다면 좌표 정보를 리턴받고, 없다면 오류 표출
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face_boundary = img_gray[y:y+h, x:x+w]      # 얼굴 영역
    face_boundary = cv2.cvtColor(face_boundary, cv2.COLOR_GRAY2RGB)
    face_boundary = cv2.resize(face_boundary, (48, 48), interpolation=cv2.INTER_LINEAR)     # PATCH_SIZE: (48, 48)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    img_trans = transform(face_boundary)           # Preprocess image
    img_trans = img_trans.unsqueeze(0)             # Add batch dimension

    output = model(img_trans)                               # Forward pass
    pred = torch.argmax(output, 1)                          # Get predicted class if multi-class classification
    print('Image predicted as ', emotion_labels[pred])      # Output: Emotion class for the face
    cv2.putText(img, emotion_labels[pred], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('detect', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

