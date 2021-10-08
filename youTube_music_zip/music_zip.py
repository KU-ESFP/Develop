import time
import chromedriver_autoinstaller
import subprocess
import shutil
import pyautogui
import requests as req

from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver import ActionChains

from youTube_music_zip.private_key import ID, PASSWORD
try:
    shutil.rmtree(r"C:\chrometemp")  # remove Cookie, Cache files
except FileNotFoundError:
    pass

try:
    subprocess.Popen(r'C:\Program Files\Google\Chrome\Application\chrome.exe --remote-debugging-port=9222 '
                     r'--user-data-dir="C:\chrometemp"')  # Open the debugger chrome
except FileNotFoundError:
    subprocess.Popen(r'C:\Users\binsu\AppData\Local\Google\Chrome\Application\chrome.exe --remote-debugging-port=9222 '
                     r'--user-data-dir="C:\chrometemp"')

option = Options()
option.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0]
try:
    driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe', options=option)
except:
    chromedriver_autoinstaller.install(True)
    driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe', options=option)
driver.implicitly_wait(10)


# YouTube Login - Chrome
driver.get(
    url='https://accounts.google.com/signin/v2/identifier?service=youtube&uilel=3&passive=true&continue=https%3A%2F%2Fwww.youtube.com%2Fsignin%3Faction_handle_signin%3Dtrue%26app%3Ddesktop%26hl%3Dko%26next%3Dhttps%253A%252F%252Fwww.youtube.com%252Fplaylist%253Flist%253DPL0ZFmU0U5awl1Dnw9N7IYM4LnipcNeQC5&hl=ko&ec=65620&flowName=GlifWebSignIn&flowEntry=ServiceLogin'
)

pyautogui.write(ID)                 # Fill in your ID or E-mail
pyautogui.press('tab', presses=3)   # Press the Tab key 3 times
pyautogui.press('enter')
time.sleep(3)                       # wait a process
pyautogui.write(PASSWORD)           # Fill in your PW
pyautogui.press('enter')


# Make YouTube Zip
time.sleep(5)
list_songs = ['https://www.youtube.com/watch?v=mrBwXXj0p34', 'https://www.youtube.com/watch?v=WPdWvnAAurg', 'https://www.youtube.com/watch?v=0-q1KafFCLU']  # song_url list

for song_url in list_songs:
    print(song_url)

    driver.find_element_by_xpath('/html/body/ytd-app/div/ytd-page-manager/ytd-browse/ytd-playlist-sidebar-renderer/div/ytd-playlist-sidebar-primary-info-renderer/div[4]/ytd-menu-renderer/yt-icon-button/button').send_keys(Keys.ENTER)    # 3개 점 버튼
    driver.find_element_by_xpath('/html/body/ytd-app/ytd-popup-container/tp-yt-iron-dropdown/div/ytd-menu-popup-renderer/tp-yt-paper-listbox/ytd-menu-service-item-renderer[1]').send_keys(Keys.ENTER)                                      # 동영상 추가 버튼
    time.sleep(2)

    iframes = driver.find_elements_by_tag_name('iframe')                # iframe 에 해당하는 모든 것 구하기
    driver.switch_to.frame(iframes[len(iframes)-1])                     # '재생목록에 동영상 추가'라는 frame open하고, driver의 화면을 현재 frame으로 전환
    # print(driver.page_source)
    time.sleep(3)

    driver.find_element_by_xpath("//input[@type='text']").send_keys(song_url)   # music url 작성

    driver.find_element_by_xpath('//*[@id="doclist"]/div/div[3]/div[2]/div/div[2]/div/div/div[1]/div[1]/div[1]/div/div/div').send_keys(Keys.ENTER)  # 검색 버튼 선택
    time.sleep(2)
    driver.find_element_by_xpath('//*[@id=":p"]/div').click()                       # 노래 선택
    time.sleep(2)
    driver.find_element_by_xpath('//*[@id="picker:ap:2"]').send_keys(Keys.ENTER)    # 동영상 추가 버튼 선택
    driver.switch_to.default_content()  # frame 화면으로 전환됐었던 걸 다시 원래대로 돌려놓기
    time.sleep(7)



