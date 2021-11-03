from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager # webdriver-manager 패키지 다운로드

import time
## Webdirver option 설정

options = webdriver.ChromeOptions()
options.add_argument('headless') # 크롬 띄우는 창 없애기
options.add_argument('window-size=1920x1080') # 크롬드라이버 창크기
options.add_argument("disable-gpu") #그래픽 성능 낮춰서 크롤링 성능 쪼금 높이기
options.add_argument("lang=ko_KR") # 사이트 주언어
driver = webdriver.Chrome(ChromeDriverManager().install(),chrome_options=options)

#찾으려는 음악 정보 가져오기(android에게 전해줄 response)
def search_music(search_name):
    youtubeUrl = (f"https://www.youtube.com/results?search_query={search_name}")
    driver.get(youtubeUrl)
    time.sleep(0.1)

    box_list = driver.find_elements_by_css_selector("#contents > ytd-video-renderer")
    box = box_list[0]
    title = box.find_element_by_css_selector('#video-title')
    link = title.get_attribute("href")
    title_name = box.find_element_by_css_selector('#video-title').text
    img_ = box.find_element_by_css_selector('#img')
    img = img_.get_attribute('src')
    name = box.find_element_by_css_selector('#text > a').text
    music = {'id': link[-11:],
            'title': search_name[:-1],
            'thumbnail': img}
    return music

# 감정 별로 분류하기 위한 유튜브 음악 url 가져오기
def search_url(search_name):
    search_name = '[MV] ' + search_name
    youtubeUrl = (f"https://www.youtube.com/results?search_query={search_name}")
    driver.get(youtubeUrl)
    time.sleep(0.1)

    box_list = driver.find_elements_by_css_selector("#contents > ytd-video-renderer")
    box = box_list[0]
    title = box.find_element_by_css_selector('#video-title')
    link = title.get_attribute("href")
    return link