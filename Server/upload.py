from flask import jsonify
import os
from flask import Flask, request
from Server.emotional_classification_gray_gpu import Emotion
from Server.youtube import music_info
import random

UPLOAD_FOLDER = 'emotional_classification_gray_gpu/input_images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
emotion_list = ['angry', 'happy', 'neutral', 'sad', 'surprised']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 가능한 파일 확장자 지정
def allowedFile(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

root_dir = os.path.dirname(os.path.realpath(__file__))
images_path = root_dir + '/emotional_classification_gray_gpu/input_images/'

# 클라이언트가 보내준 이미지 삭제
def deleteInputImage():
    for input_image in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, input_image)):
            os.remove(os.path.join(images_path, input_image))

# 받은 emotion에 맞는 음악 list 보여주기
def showMusic(emotion):
    f = open('music_classification/output_top100/' + emotion + '.txt', mode='r', encoding='utf-8')
    music_list = []
    count_music = 0
    for line in f:
        #print("line : " + line)
        music_list.append(line)
        count_music += 1
        #if count_line == 2:
            #break;
    f.close()
    return [count_music, music_list]

#미리보기로 보여줄 2개 음악 랜덤설정
def makeRandNum(count_music):
    random_number = random.sample(range(1, count_music), 2)
    return random_number

#감정 별 해당 유튜브 id 얻기
def getId(index):
    f = open('youtube/playlist_url/playlist_id.txt', mode='r')
    id_list = []
    for line in f:
        id_list.append(line)
    f.close()
    return id_list[index]

# 서버 켜기
@app.route('/', methods=['POST'])
def fileUpload():

    if request.method == "POST":
        file = request.files['file']
        #file = request.files.get('file')
        if file and allowedFile(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fileType = '.' + filename.rsplit('.', 1)[1]
            fileSize = os.path.getsize(UPLOAD_FOLDER + file.filename)
            emotion = Emotion.runEmotion(filename)
            if emotion == None:
                return "NULL"
            else:
                info_list = showMusic(emotion)
                music_list = info_list[1]
                count_music = info_list[0]
                num1, num2 = makeRandNum(count_music)
                index = emotion_list.index(emotion)
                playlist_id = getId(index)
                return jsonify({"file": {"emotion" : emotion, "fileName": filename, "fileType": fileType, "fileSize": fileSize},
                                "playlist_id" : playlist_id,
                                "music1": music_info.search_music(music_list[num1]),
                                "music2": music_info.search_music(music_list[num2])})
    return

if __name__ == '__main__':
    deleteInputImage()
    app.run(host="0.0.0.0", port="8000", debug=True)
