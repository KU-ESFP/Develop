import os
import torch
import random
import operator

from konlpy.tag import Kkma
from src.models.model import GRU
from torchtext.legacy.data import Field, Example, Dataset, Iterator

# angry, happy, neutral, sad
kkma = Kkma()


# 파일 저장 함수
def saveOutputSong(song_info, emotion):
    output_path = "./output_top100"
    f = open(os.path.join(output_path, emotion + '.txt'), 'a', encoding='UTF8')
    f.write(song_info + '\n')
    f.close()


# 이전 top100 정보 지우기
def deleteOutputSong():
    output_path = "./output_top100"
    for emotion in os.listdir(output_path):
        if os.path.isfile(os.path.join(output_path, emotion)):
            os.remove(os.path.join(output_path, emotion))

# 자연어 토큰화
def custom_tokenizer(text):
    tag_word = kkma.pos(text)

    stopwords = []
    for word, tag in tag_word:
        if tag in ['SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'UN', 'OL', 'OH', 'ON', 'NR', 'NNA']:
            stopwords.append(word)

    tmp_word = [word for word, tag in tag_word if not word in stopwords]
    return tmp_word


# Model load
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

vocab_size = 17407      # src/train_model_classifier: len(TEXT.vocab)
n_classes = 4

print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)
model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
path = 'trained_models/music_models/txt_classification_02.pth'
model.load_state_dict(torch.load(path))
model.eval()


'''
    멜론 TOP100 곡 감정 예측 알고리즘
'''
# path name
top100_path = 'input_top100'
# top100_path = 'input_test'
save_path = './csv_datasets/'
input_dataset = 'input_test_data.csv'


# 데이터
BATCH_SIZE = 64
classes = ['angry', 'happy', 'neutral', 'sad']


'''
    [top100 갱신할 때만 사용]
    exec(open('src/crawlings/input_crawling.py').read())
    
    - 기능: input_top100 폴더에 1-100 순위의 노래 update
    - 특이사항: 하루에 한 번만 실행 (오전 12시)
'''
exec(open('src/crawlings/input_crawling.py', encoding='utf8').read())


''' 
    output_top100 갱신
    - angry, happy, neutral, sad: 감정에 맞는 노래 추천
    - surprised: 1-100개의 노래 중 랜덤으로 7개 노래 추천
'''
# 이전 output_top100 데이터 지워주기
deleteOutputSong()

# top100 정보를 담은 리스트
list_top100 = []

# output_top100 데이터 갱신
for song_info in os.listdir(top100_path):
    if os.path.isfile(os.path.join(top100_path, song_info)):
        list_top100.append(song_info)
        fread = open(os.path.join(top100_path, song_info), 'r', encoding='UTF8')
        song = fread.readline()
        if not song:
            continue
        line = kkma.sentences(song)

        # 1. Field 정의
        TEXT = Field(sequential=True, use_vocab=True, tokenize=custom_tokenizer, batch_first=True)
        fields = [('text', TEXT)]

        # 2. torchtext.data.Example 생성
        sequences = []
        for s in line:
            # print(s)
            sequences.append(Example.fromlist([s], fields))

        # 3. Dataset생성 (word data)
        test_data = Dataset(sequences, fields)

        # 4. vocab 생성
        TEXT.build_vocab(test_data, min_freq=1)
        # print(TEXT.vocab.stoi)
        if len(TEXT.vocab) == 2:           # {'<unk>': 0, '<pad>': 1}: 만들어진 단어가 없다는 것은 token 후 남은 문자가 없다는 것
            continue

        # 5. Dataset생성 (id로 변환된 data)
        # padding 이 되면서, 같은 길이로 만들어진다. 따라서 문장의 가장 긴 것에 맞춰서 가장 긴게 잘리지 않도록 하자
        test_loader = Iterator(dataset=test_data, batch_size=BATCH_SIZE)

        model.eval()

        classes_dic = {'angry': 0, 'happy': 0, 'neutral': 0, 'sad': 0}
        for batch in test_loader:
            x = batch.text.to(DEVICE)
            logit = model(x)
            # print(logit.data)
            # print('logit_max', logit.max(1))
            # print('predict', logit.max(1)[1])
            for emotion_idx in logit.max(1)[1]:
                # print('emotion_idx', emotion_idx)
                classes_dic[classes[emotion_idx]] += 1
                # print('classes_dix', classes_dic[classes[emotion_idx]])
        # print(classes_dic)  # {'angry': 0, 'happy': 3, 'neutral': 7, 'sad': 3} -> Result is neutral
        best_emotion = max(classes_dic.items(), key=operator.itemgetter(1))[0]
        print('SONG NAME: {}, EMOTION: {}'.format(song_info.rstrip('.txt'), best_emotion))
        saveOutputSong(song_info.rstrip('.txt'), best_emotion)    # new top100 info
        fread.close()


# 감정이 surprise 일 때는 top1-100 중 7곡 랜덤으로 구해주기
list_surprised = random.sample(list_top100, 7)
for data in list_surprised:
    saveOutputSong(data.rstrip('.txt'), 'surprised')
