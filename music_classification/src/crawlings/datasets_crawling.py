import requests as req
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver import ActionChains

import time

'''
    [데이터 크롤링]
    site: Melon
    
    테마에 따라서 분류
    - Melon > 멜론DJ > #테마장르
'''

## happy
url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=478102836#params%5BplylstSeq%5D=478102836&po=pageObj&startIndex=1'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=496795342'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=456210363'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=444988300'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=496490405'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=445289879'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=499139357'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=475486287#params%5BplylstSeq%5D=475486287&po=pageObj&startIndex=51'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=494385023'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=490394549'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=481825287'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=481790930'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=415051158'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=499588050'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=490168534'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=456337994'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=456210319'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=439331275#params%5BplylstSeq%5D=439331275&po=pageObj&startIndex=1'
## sad
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=403014132#params%5BplylstSeq%5D=403014132&po=pageObj&startIndex=1'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=483878595'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=430177186#params%5BplylstSeq%5D=430177186&po=pageObj&startIndex=1'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=448382538'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=408429497#params%5BplylstSeq%5D=408429497&po=pageObj&startIndex=1'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=489509078'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=498007628'
## angry
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=457914982'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=490101618'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=405024652'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=498229662'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=442578455'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=494715709#params%5BplylstSeq%5D=494715709&po=pageObj&startIndex=51'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=443038858'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=475425914'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=480884431'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=457914982'
## neutral
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=426612354#params%5BplylstSeq%5D=426612354&po=pageObj&startIndex=501'
# url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=436119495#params%5BplylstSeq%5D=436119495&po=pageObj&startIndex=1'
# url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=457882592#params%5BplylstSeq%5D=457882592&po=pageObj&startIndex=1'
# url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=445557345#params%5BplylstSeq%5D=445557345&po=pageObj&startIndex=1'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=470374673'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=476424903'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=464776889'
# url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=475811466'
## url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=479643659'


# 파일 저장 함수
def saveDatasetsSong(song_name, lyric):
    song_text = []
    for a in song_name:
        song_text.append(a.text.strip())
    real_song = song_text[0].replace("\t", "").replace("\n", "").replace("곡명", "").replace("?", "").replace("/", " ")
    full_lyric = []
    for i in lyric:
        full_lyric.append(i.text.strip())
    path = "../../song_datasets/happy_songs/"
    song_path = real_song + ".txt"
    real_path = path + song_path
    a = str(real_path)
    if not full_lyric:
        return
    f = open(a, 'w', -1, 'utf-8')
    f.write(full_lyric[0])
    f.close()


# 페이지 이동 후 이전페이지 이동 함수
def pageDatasetsMove(path):
    driver.find_element_by_xpath(path).click()
    source = driver.page_source
    soup = bs(source, 'html.parser')
    lyric = soup.find_all('div', class_='lyric')
    song_name = soup.find_all('div', class_='song_name')
    time.sleep(1)
    driver.back()               # 이전 페이지 이동
    saveDatasetsSong(song_name, lyric)  # 파일 저장 함수


header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko'}
melon = req.get(url, headers=header)
melon_html = melon.text
melon_parse = bs(melon_html, 'html.parser')

driver = webdriver.Chrome(executable_path='C:\chromedriver\chromedriver.exe')
driver.get(url)
time.sleep(1)
action = ActionChains(driver)

for j in range(1, 11):          # 페이지 번호
    for i in range(1, 51):      # 노래 번호
        time.sleep(5)
        pageDatasetsMove('//*[@id="frm"]/div/table/tbody/tr['+str(i)+']/td[4]/div/a')
    driver.find_element_by_xpath('//*[@id="pageObjNavgation"]/div/span/a['+str(j)+']').click()

# close browser
driver.close()
