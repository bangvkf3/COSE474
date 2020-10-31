from bs4 import BeautifulSoup
from selenium import webdriver
import time
import re
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

driver = webdriver.Chrome("./chromedriver_win32/chromedriver")
driver.implicitly_wait(3)

driver.get('https://www.imdb.com/title/tt6751668/reviews?ref_=tt_ql_3')

driver.find_element_by_xpath('//*[@id="load-more-trigger"]').click()
time.sleep(10)
driver.find_element_by_xpath('//*[@id="load-more-trigger"]').click()
time.sleep(10)
driver.find_element_by_xpath('//*[@id="load-more-trigger"]').click()
time.sleep(10)
# # #
click_list = driver.find_elements_by_xpath("//div[@class='expander-icon-wrapper show-more__control']")
for click in click_list:
    if click.is_displayed():
        click.click()
        time.sleep(1)
# # # #
time.sleep(1)
req = driver.page_source
# # #
bs = BeautifulSoup(req, 'lxml')
#
title_list = bs.findAll('a', 'title')
review_list = bs.findAll('div', 'text show-more__control')
score_list = bs.findAll('span', 'rating-other-user-rating')

for title in title_list:
    print(title.getText())

for i, content in enumerate(review_list):
    print(i, content.getText()+"\n")

for score in score_list:
    print(score.span.getText())

print(len(title_list))
f = open("./data/review.txt", "w", encoding='UTF8')
for i in range(len(title_list)):
    f.write(clean_str(title_list[i].getText())+" "+clean_str(review_list[i].getText())+"\n")
f.close()

f = open("./data/score.txt", "w", encoding='UTF8')
for i in range(len(score_list)):
    f.write(score_list[i].span.getText()+"\n")
f.close()
