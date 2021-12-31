import os
from pydub import AudioSegment

genres = 'fast slow'
genres = genres.split()

'''
    [pydub 패키지의 audiosegment를 사용하여 오디오 파일 분할]
    - 노래의 하이라이트 부분 추출 (36 ~ 51 sec, 66 ~ 81 sec)
    - 한 노래당 총 30초 추출 (3초씩 10개)
'''

if not os.path.exists('../../content/train_datasets/audio_3_sec'):
    os.makedirs('../../content/train_datasets/audio_3_sec')

for g in genres:
    j = 0
    print(f"{g}")

    for filename in os.listdir(os.path.join('../../content/audio_compress', f"{g}")):                    # 여기에 기존 장르별 본래 노래(wav) 저장
        song = os.path.join(f'../../content/audio_compress/{g}', f'{filename}')
        j = j + 1

        for w in range(12, 17):     # 36 - 51 sec
            t1 = 3 * (w) * 1000
            t2 = 3 * (w + 1) * 1000
            newAudio = AudioSegment.from_wav(song)
            new = newAudio[t1:t2]
            new.export(f'../../content/audio_3_sec/{g + "_" + str(j) + str(w)}.wav', format="wav")       # 30초 짜리를 3초로 10개 만든걸 넣기

        for w in range(22, 27):     # 66 - 81 sec
            t1 = 3 * (w) * 1000
            t2 = 3 * (w + 1) * 1000
            newAudio = AudioSegment.from_wav(song)
            new = newAudio[t1:t2]
            new.export(f'../../content/audio_3_sec/{g + "_" + str(j) + str(w)}.wav', format="wav")


'''
 TEST 용도
'''
# for g in genres:
#     j = 0
#     print(f"{g}")
#
#     for filename in os.listdir(os.path.join('../../content/audio_origin_sec', f"{g}")):                      # 여기에 기존 장르별 본래 노래(wav) 저장
#         song = os.path.join(f'../../content/audio_origin_sec/{g}', f'{filename}')
#         j = j + 1
#
#         for w in range(12, 13):     # 36 - 51 sec
#             t1 = 3 * (w) * 1000
#             t2 = 3 * (w + 1) * 1000
#             newAudio = AudioSegment.from_wav(song)
#             new = newAudio[t1:t2]
#             new.export(f'../../content/test/{g + "_" + str(j) + str(w)}.wav', format="wav")                  # 30초 짜리를 3초로 10개 만든걸 넣기
