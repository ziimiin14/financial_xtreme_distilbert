import re
import pandas as pd
from collections import defaultdict,deque
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def process(sentence,fil):
  return fil.sub(' ',sentence)

def run():
    # Declare variable for chromedriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome('chromedriver',options=options)

    # Declare variable for selenium scraping
    stock_codes={'AAPL':'(Apple|AAPL)','MSFT':'(Microsoft|MSFT)','AMZN':'(Amazon|AMZN)','META':'(Meta|META)','GOOGL':'(Alphabet|Google|GOOGL)','BRK.B':'(Berkshire|BRK.B)','TSLA':'(Tesla|TSLA)','NVDA':'(Nvidia|NVDA)','JPM':'(JPMorgan|JPM)'}
    title = []
    href = []
    tc = []
    mention = []

    # Ask for user input for specific stock news from stock_code list
    print(200*'-')
    sc = input(f'Choose a stock: \n{list(stock_codes.keys())} \n-->')

    # Scrape specific news title and href
    print(200*'-')
    print(f'-->Scraping News for {sc} stock<--\n')
    url = f'https://www.cnbc.com/quotes/{sc}?tab=news'
    driver.get(url)
    try:
      element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "LatestNews-newsTabButton")))
      element.click()

      element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "LatestNews-newsTabButton")))
      element.click()
    except:
      print(f"Cannot find 2nd View More button for {sc}")

    d = driver.page_source
    soup = BeautifulSoup(d,'html.parser')
    group = soup.find_all('div',attrs={'data-analytics':'QuotePage-QuoteNews-1'})
    headlines = deque(group[0].find_all('a'))

    final_headlines = []

    # Check whether it's a legit headline for that news. Video/Club/Pro news are all being removed.
    while headlines:
      temp = headlines.popleft()

      if temp['class'][0] != 'LatestNews-headline':
        headlines.popleft()
      else:
        if not temp.img:
          final_headlines.append(temp)

    for x in final_headlines:
      tc.append(sc)
      mention.append(stock_codes[sc])
      title.append(x['title'])
      href.append(x['href'])

    # Declare a dataframe
    df = pd.DataFrame({'Symbol':tc,'Mention':mention,'title':title,'link':href})

    text_list = []

    # Open each href link and read the keypoints text or body texts
    for m,url_new in zip(df['Mention'],df['link']):
      driver.get(url_new)
      filter = re.compile(f'{m}')
      soup_news = BeautifulSoup(driver.page_source,'html.parser')
      keyPoint = soup_news.find_all('div',attrs={'data-test':'keyPoints-1'})

      # If there is no keypoints to be concatenated into sentence
      if not keyPoint:
        body = soup_news.find_all('div',attrs={'data-module':'ArticleBody'})
        group = body[0].find_all('div',attrs={'class':'group'})
        paragraphs = []
        for g in group:
          p_temp = g.find_all('p')
          for p in p_temp: 
            paragraphs.append(p)

        text = ''
        count = 0

        # For each paragraph, apply regex filter to find the specific companies mentioned and concatenate the paragraph and next paragraph into sentence
        for p in paragraphs:
          if filter.search(p.text):
            text += p.text.strip()
            count += 1
          if count == 2:
            break

        # If there is no specific companies mentioned in paragraphs. Just choose 1st and 2nd paragraphs and concatenate into sentence
        if not text:
          if len(paragraphs) > 1:
            text = paragraphs[0].text.strip() + paragraphs[1].text.strip()
          else:
            text = paragraphs[0].text.strip()

        text_list.append(text)

      # Concatenate all keyspoint into sentence
      else:
        li = keyPoint[0].find_all('li')
        text = ''
        for t in li:
          text += t.text.strip()
        text_list.append(text)

    df['sentence'] = text_list

    # Apply regex for sentence to remove \xa0 symbol
    f = re.compile(f'\xa0')
    df['sentence'] = df['sentence'].apply(process,args=(f,))
    driver.quit()
    print('-->Done<--\n')

    print(df)

if __name__=="__main__":
    run()
