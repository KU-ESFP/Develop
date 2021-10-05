import os
import requests as req

from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver import ActionChains

import time

'''
    [데이터 크롤링]
    site: Melon

    input data
        - 인기차트 1-100 
'''

url = 'https://www.melon.com/chart/index.htm'

# input_top100 path
root_dir = os.path.dirname(os.path.realpath(__file__))
top100_path = root_dir + '/input_top100/'


# top100 정보 갱신하기 위해서 이전 데이터 삭제
def deleteInputSong():
    for song_info in os.listdir(top100_path):
        if os.path.isfile(os.path.join(top100_path, song_info)):
            os.remove(os.path.join(top100_path, song_info))

# 파일 저장 함수
def saveInputSong(song_name, artist_name, lyric):
    song_text = []
    for a in song_name:
        song_text.append(a.text.strip())
    real_song = song_text[0].replace("\t", "").replace("\n", "").replace("곡명", "").replace("?", "").replace("/", " ")

    artists = ""
    for a in artist_name:
        artists = artists + ', ' + a.text.strip()
    artists = artists.lstrip(', ')
    # print(artists)

    full_lyric = []
    for i in lyric:
        full_lyric.append(i.text.strip())

    song_path = real_song + "-" + artists + ".txt"
    real_path = top100_path + song_path
    a = str(real_path)
    if not full_lyric:
        return
    f = open(a, 'w', -1, 'utf-8')
    f.write(full_lyric[0])
    f.close()


# 페이지 이동 후 이전페이지 이동 함수
def pageInputMove(path):
    driver.find_element_by_xpath(path).click()
    source = driver.page_source
    soup = bs(source, 'html.parser')
    lyric = soup.find_all('div', class_='lyric')
    song_name = soup.find_all('div', class_='song_name')
    artist_name = soup.select('#downloadfrm > div > div > div.entry > div.info > div.artist > a')
    time.sleep(1)
    driver.back()  # 이전 페이지 이동
    saveInputSong(song_name, artist_name, lyric)  # 파일 저장 함수


header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko'}
melon = req.get(url, headers=header)
melon_html = melon.text
melon_parse = bs(melon_html, 'html.parser')

driver = webdriver.Chrome(executable_path='C:\chromedriver\chromedriver.exe')
driver.get(url)
time.sleep(1)
action = ActionChains(driver)

# 이전 top100 정보 삭제
deleteInputSong()

# 새로운 top100 정보 crawling
for i in range(1, 101):  # 인기차트 1-100
    time.sleep(1)
    pageInputMove('//*[@id="frm"]/div/table/tbody/tr[' + str(i) + ']/td[5]/div/a')

# close browser
driver.close()
