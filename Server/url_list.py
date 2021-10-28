import os
from Server.youtube import music_info

emotion_list = ['angry', 'happy', 'neutral', 'sad', 'surprised']

root_dir = os.path.dirname(os.path.realpath(__file__))
top100_path = root_dir + '\music_classification\output_top100\\'
output_path = root_dir + '\youtube\output_url\\'

def makeOutputFile():
    for file_name in emotion_list:
        file_path = top100_path + file_name + '.txt'
        f = open(file_path, 'rt', encoding='UTF-8')
        lines = f.readlines()
        for line in lines:
            print(line)
            line = line.strip()
            url = music_info.search_url(line)
            url_path = output_path + file_name + '.txt'
            f = open(url_path, 'a+', encoding='UTF-8')
            f.write(url + '\n')
        f.close()
    return

def deleteOutputUrl():
    for url in os.listdir(output_path):
        if os.path.isfile(os.path.join(output_path, url)):
            os.remove(os.path.join(output_path, url))

deleteOutputUrl()
makeOutputFile()





