import time
import chromedriver_autoinstaller
import subprocess
import shutil
import pyautogui
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait

from Server.youtube.private_key import ID, PASSWORD

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

emotion_list = ['angry', 'happy', 'neutral', 'sad', 'surprised']
url_list = []


output_path = "./playlist_url"
for emotion in os.listdir(output_path):
    if os.path.isfile(os.path.join(output_path, emotion)):
        os.remove(os.path.join(output_path, emotion))

driver.get(url='https://studio.youtube.com/channel/UCWXHfwdEQ0mMALIOj0_gDfQ/playlists')
time.sleep(3)
pyautogui.write(ID)  # Fill in your ID or E-mail
pyautogui.press('tab', presses=3)  # Press the Tab key 3 times
pyautogui.press('enter')
time.sleep(3)  # wait a process
pyautogui.write(PASSWORD)  # Fill in your PW
pyautogui.press('enter')
time.sleep(5)
for emotion in emotion_list:
    driver.find_element_by_xpath('/html/body/ytcp-app/ytcp-entity-page/div/div/main/div/ytcp-animatable[1]/div[1]/ytcp-button').send_keys(Keys.ENTER)
    driver.find_element_by_xpath('/html/body/ytcp-playlist-creation-dialog/ytcp-dialog/tp-yt-paper-dialog/div[2]/div/div[1]/ytcp-form-textarea/div/textarea').send_keys(Keys.ENTER)
    time.sleep(2)
    pyautogui.write(emotion)

    #pyautogui.write('angry')
    driver.find_element_by_xpath('/html/body/ytcp-playlist-creation-dialog/ytcp-dialog/tp-yt-paper-dialog/div[2]/div/div[2]/ytcp-button[2]').send_keys(Keys.ENTER)
    time.sleep(5)
    box = driver.find_element_by_css_selector('#playlists-section > ytcp-playlist-section-content ')
    id = box.find_element_by_css_selector('#row-container > div:nth-child(1) > div > a').get_attribute("href")

    output_path = "./playlist_url"
    f = open(os.path.join(output_path, 'playlist_id.txt'), 'a', encoding='UTF8')
    f.write(id[38:] + '\n')
    f.close()

    #url_list.append(id[38:])
    #print(url_list)

    file_path = './playlist_url/playlist_id.txt'
    f = open(file_path, 'rt', encoding='UTF-8')
    url_list = f.readlines()


for index, emotion in enumerate(emotion_list):
    time.sleep(3)
    # YouTube Login - Chrome
    driver.get(
        url='https://accounts.google.com/signin/v2/identifier?service=youtube&uilel=3&passive=true&continue=https%3A%2F%2Fwww.youtube.com%2Fsignin%3Faction_handle_signin%3Dtrue%26app%3Ddesktop%26hl%3Dko%26next%3Dhttps%253A%252F%252Fwww.youtube.com%252Fplaylist%253Flist%253D' + url_list[index] + '&hl=ko&ec=65620&flowName=GlifWebSignIn&flowEntry=ServiceLogin'
    )

    pyautogui.write(ID)                 # Fill in your ID or E-mail
    pyautogui.press('tab', presses=3)   # Press the Tab key 3 times
    pyautogui.press('enter')
    time.sleep(3)                       # wait a process
    pyautogui.write(PASSWORD)           # Fill in your PW
    pyautogui.press('enter')



    # Make YouTube Zip
    time.sleep(5)
    #list_songs = ['https://www.youtube.com/watch?v=mrBwXXj0p34', 'https://www.youtube.com/watch?v=WPdWvnAAurg', 'https://www.youtube.com/watch?v=0-q1KafFCLU']  # song_url list
    #list_songs = ['https://www.youtube.com/watch?v=MjCZfZfucEc', 'https://www.youtube.com/watch?v=tL7HKKEoW1Y']



    file_path = './output_url/' + emotion + '.txt'
    f = open(file_path, 'rt', encoding='UTF-8')
    lines = f.readlines()

    time.sleep(5)
    for song_url in lines:

        print(song_url)
        song_url = song_url[:-1]
        driver.find_element_by_xpath('/html/body/ytd-app/div/ytd-page-manager/ytd-browse/ytd-playlist-sidebar-renderer/div/ytd-playlist-sidebar-primary-info-renderer/div[4]/ytd-menu-renderer/yt-icon-button/button').send_keys(Keys.ENTER)    # 3개 점 버튼
        driver.find_element_by_xpath('/html/body/ytd-app/ytd-popup-container/tp-yt-iron-dropdown/div/ytd-menu-popup-renderer/tp-yt-paper-listbox/ytd-menu-service-item-renderer[1]').send_keys(Keys.ENTER)                                      # 동영상 추가 버튼
        time.sleep(1)

        iframes = driver.find_elements_by_tag_name('iframe')                # iframe 에 해당하는 모든 것 구하기
        driver.switch_to.frame(iframes[len(iframes)-1])                     # '재생목록에 동영상 추가'라는 frame open하고, driver의 화면을 현재 frame으로 전환
        # print(driver.page_source)
        time.sleep(2)

        driver.find_element_by_xpath("//input[@type='text']").send_keys(song_url)   # music url 작성

        driver.find_element_by_xpath('//*[@id="doclist"]/div/div[3]/div[2]/div/div[2]/div/div/div[1]/div[1]/div[1]/div/div/div').send_keys(Keys.ENTER)  # 검색 버튼 선택
        time.sleep(1)

        driver.find_element_by_xpath('//*[@id=":p"]/div').click()                       # 노래 선택
        time.sleep(1)
        driver.find_element_by_xpath('//*[@id="picker:ap:2"]').send_keys(Keys.ENTER)    # 동영상 추가 버튼 선택
        driver.switch_to.default_content()  # frame 화면으로 전환됐었던 걸 다시 원래대로 돌려놓기
        time.sleep(3)
