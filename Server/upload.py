from flask import jsonify
import os
from flask import Flask, request
from emotional_classification_gray import Emotion
from youtube import music_info
import random

UPLOAD_FOLDER = 'emotional_classification_gray/input_images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

root_dir = os.path.dirname(os.path.realpath(__file__))
images_path = root_dir + '/emotional_classification_gray/input_images/'

def deleteInputImage():
    for input_image in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, input_image)):
            os.remove(os.path.join(images_path, input_image))


def show_music(emotion):
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

def makeRandNum(count_music):

    random_number = random.sample(range(1, count_music), 2)

    return random_number



@app.route('/', methods=['POST'])
def fileUpload():

    if request.method == "POST":
        file = request.files['file']
        #file = request.files.get('file')
        if file and allowed_file(file.filename):

            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fileType = '.' + filename.rsplit('.', 1)[1]
            fileSize = os.path.getsize(UPLOAD_FOLDER + file.filename)
            emotion = Emotion.runEmotion(filename)
            if emotion == None:
                return "NULL"
            else:
                info_list = show_music(emotion)
                music_list = info_list[1]
                count_music = info_list[0]
                num1, num2 = makeRandNum(count_music)
                return jsonify({"file": {"emotion" : emotion, "fileName": filename, "fileType": fileType, "fileSize": fileSize},
                                "music1": music_info.search_music(music_list[num1]),
                                "music2": music_info.search_music(music_list[num2])})
    return

if __name__ == '__main__':
    deleteInputImage()
    app.run(host="0.0.0.0", port="8000", debug=True)
