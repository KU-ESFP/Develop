import os

from gensim.models import Word2Vec
from konlpy.tag import Kkma


# 데이터 전처리
kkma = Kkma()

song_path = '../../song_datasets/happy_songs'
# song_path = 'input_test'

# 노래 한 곡 단위로 수집한 노래 가사의 단어를 불러와 리스트 형태로 구현하기 위해 아래와 같이 토큰화 진행
# 평서형 종결 어미인 'EFN' 기준으로 문장을 나눔
tagging_kkma = []
for song_name in os.listdir(song_path):
    if os.path.isfile(os.path.join(song_path, song_name)):
        fread = open(os.path.join(song_path, song_name), 'r', encoding='UTF8')
        song = fread.readline()
        song_sentences = kkma.sentences(song)
        # print('song_name', song_name)
        for line in song_sentences:
            tagging_kkma.append(kkma.pos(line))
        fread.close()
print('tagging_kkma', tagging_kkma)
print("tagging_kkma size", len(tagging_kkma))

# 불용어 처리
stopwords = []
for index in range(len(tagging_kkma)):
    for word, tag in tagging_kkma[index]:
        if tag in ['SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'UN', 'OL', 'OH', 'ON', 'NR', 'NNA']:
            stopwords.append(word)
# print("stopwords", stopwords)

# 태깅한 데이터들을 담은 변수
tagging = []
tagging_data = []
for index in range(len(tagging_kkma)):
    tmp_word = [word for word, tag in tagging_kkma[index] if not word in stopwords]
    tagging_data.append(tmp_word)
# print('tagging data', tagging_data)
print('The total number of samples: {}'.format(len(tagging_data)))

model = Word2Vec(tagging_data, min_count=1, vector_size=150, epochs=10, sg=0, batch_words=2048, window=5)
# model.save('./word2vec_models/w2v_sad_10.model')
model.save('../../trained_models/word2vec_models/w2v_happy_01.model')
