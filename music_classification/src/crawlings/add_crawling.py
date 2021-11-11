import requests as req
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver import ActionChains

import time

url = 'https://www.melon.com/genre/song_list.htm?gnrCode=GN0200#params%5BgnrCode%5D=GN0200&params%5BdtlGnrCode%5D=&params%5BorderBy%5D=NEW&params%5BsteadyYn%5D=N&po=pageObj&startIndex=1'

# 파일 저장 함수
def saveAddSong(song_name, lyric):
    song_text = []
    for a in song_name:
        song_text.append(a.text.strip())
    real_song = song_text[0].replace("\t", "").replace("\n", "").replace("곡명", "").replace("?", "").replace("/", " ")
    full_lyric = []
    for i in lyric:
        full_lyric.append(i.text.strip())
    path = "../../input_test/"
    song_path = real_song + ".txt"
    real_path = path + song_path
    a = str(real_path)
    if not full_lyric:
        return
    f = open(a, 'w', -1, 'utf-8')
    f.write(full_lyric[0])
    f.close()


# 페이지 이동 후 이전페이지 이동 함수
def pageAddMove(path):
    driver.find_element_by_xpath(path).click()
    source = driver.page_source
    soup = bs(source, 'html.parser')
    lyric = soup.find_all('div', class_='lyric')
    song_name = soup.find_all('div', class_='song_name')
    time.sleep(1)
    driver.back()  # 이전 페이지 이동
    saveAddSong(song_name, lyric)  # 파일 저장 함수


header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko'}
melon = req.get(url, headers=header)
melon_html = melon.text
melon_parse = bs(melon_html, 'html.parser')

driver = webdriver.Chrome(executable_path='C:\chromedriver_95\chromedriver.exe')
driver.get(url)
time.sleep(1)
action = ActionChains(driver)

for j in range(1, 11):  # 페이지 번호
    for i in range(1, 51):  # 노래 번호
        time.sleep(5)
        pageAddMove('//*[@id="frm"]/div/table/tbody/tr[' + str(i) + ']/td[4]/div/a')
    driver.find_element_by_xpath('//*[@id="pageObjNavgation"]/div/span/a[' + str(j) + ']').click()

# close browser
driver.close()
