from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import json
import time
import random
from fake_useragent import UserAgent

options = Options()
ua = UserAgent()
user_agent = ua.random
options.add_argument(f'user-agent={user_agent}')

with open('C://Users/Administrator/Desktop/TextGen/concept/concept_pool.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

key_list = []
flag = False
for key in data.keys():
    if key == 'sprinkler':
        flag = True
    if not flag:
        continue
    key_list.append(key)

driver = webdriver.Chrome(options=options)
for key in key_list:
    # query = f"https://www.google.com/search?q={key}"
    query = f"https://translate.google.com/?hl=zh-CN&sl=en&tl=zh-CN&text={key}&op=translate"
    driver.get(query)
    time.sleep(2)

driver.quit()
