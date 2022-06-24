import numpy as np
import pandas as pd
import os
import torch
import time
import argparse
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler
from datasets import load_metric,load_dataset,Dataset
from sklearn.metrics import confusion_matrix, classification_report,roc_curve,auc
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from collections import defaultdict,deque
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def process(sentence,fil):
  return fil.sub(' ',sentence)

def tokenize_function(examples,tokenizer):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

def inference():
    # Declare model/tokenizer/device variables
    tokenizer_path = './tokenizer_last/'
    model_path = './model_last/'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load Model and Tokenized
    print('Loading Model and Tokenizer \n')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    ################################################################################################################################################################
    # Declare variable for chromedriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
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

    df = pd.DataFrame({'Symbol':tc,'Mention':mention,'title':title,'link':href})

    text_list = []

    # Open each href link and read the keypoints text or body texts
    for m,url_new in zip(df['Mention'],df['link']):
      driver.get(url_new)
      filter = re.compile(f'{m}')
      soup_news = BeautifulSoup(driver.page_source,'html.parser')
      keyPoint = soup_news.find_all('div',attrs={'data-test':'keyPoints-1'})

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

        for p in paragraphs:
          if filter.search(p.text):
            text += p.text.strip()
            count += 1
          if count == 2:
            break
        if not text:
          if len(paragraphs) > 1:
            text = paragraphs[0].text.strip() + paragraphs[1].text.strip()
          else:
            text = paragraphs[0].text.strip()

        text_list.append(text)

      else:
        li = keyPoint[0].find_all('li')
        text = ''
        for t in li:
          text += t.text.strip()
        text_list.append(text)

    df['sentence'] = text_list
    f = re.compile(f'\xa0')
    df['sentence'] = df['sentence'].apply(process,args=(f,))
    driver.quit()
    print('-->Done<--\n')


    ################################################################################################################################################
    # Setup tokenized dataset
    datasets = Dataset.from_dict({'sentence':df['sentence'].tolist()})
    tokenized_datasets = datasets.map(tokenize_function,batched=True,fn_kwargs={'tokenizer':tokenizer})
    tokenized_datasets = tokenized_datasets.remove_columns(['sentence'])
    # tokenized_sentence = tokenizer(df['sentence'].tolist(),padding="max_length", truncation=True)
    # tokenized_datasets = Dataset.from_dict(tokenized_sentence)
    tokenized_datasets.set_format("torch")

    # Dataloader
    data_loader = DataLoader(tokenized_datasets,batch_size=tokenized_datasets.num_rows)

    # Predict sentiment for the news
    model.to(device)
    model.eval()
    print(200*'-')
    print('-->Predicting sentiment for news<--\n')
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)


    df['sentiment'] = predictions.cpu().detach().numpy()
    print('-->Done<--\n')
    print(df)



def eval():
    # Declare variables and path
    model_path = './model_last/'
    tokenizer_path = './tokenizer_last/'
    data_path = './data/'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    metric = load_metric("accuracy")

    # Load datasets
    dataset = load_dataset('csv',data_files={'train':data_path+'train.csv','val':data_path+'val.csv','test':data_path+'test.csv'})

    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Setup tokenized dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True,fn_kwargs={'tokenizer':tokenizer})
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    small_test_dataset = tokenized_datasets["test"]

    # Dataloader
    test_dataloader = DataLoader(small_test_dataset,batch_size=small_test_dataset.num_rows)

    # Evaluation test set
    model.eval()
    start = time.time()

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        y_pred = predictions.cpu().detach().tolist()
        y_true = batch['labels'].cpu().detach().tolist()
        metric.add_batch(predictions=predictions, references=batch["labels"])
        losses = loss.cpu().detach().tolist()

    end = time.time()
    total_time = end-start
    test_acc,test_loss = metric.compute()['accuracy'],np.mean(losses)
    prob = logits.cpu().detach().numpy()

    fp,tp,th = roc_curve(y_true,prob[:,0],pos_label=0)
    fp1,tp1,th1 = roc_curve(y_true,prob[:,1],pos_label=1)
    fp2,tp2,th2 = roc_curve(y_true,prob[:,2],pos_label=2)
    area_uc = round(auc(fp,tp),3)
    area_uc1 = round(auc(fp1,tp1),3)
    area_uc2= round(auc(fp2,tp2),3)

    print('\n')
    print(f'Test Accurary: {test_acc}, Test Loss: {test_loss}')
    print('\n')
    print('Confusion Matrix:\n', confusion_matrix(y_true,y_pred))
    print('\n')
    print('Classification Report:\n', classification_report(y_true,y_pred,target_names=['Neg','Neu','Pos']))
    print('\n')
    print(f'Neg AUC: {area_uc}, Neu AUC: {area_uc1}, Pos AUC: {area_uc2}')
    print('\n')
    print(f'Total time used: {total_time}\nSamples/Sec for {device}: {small_test_dataset.num_rows/total_time}') 


def train():
    # Declare variables and path
    model_name = 'microsoft/xtremedistil-l6-h256-uncased'
    model_best_path = './model_best/'
    model_path = './model_last/'
    tokenizer_path = './tokenizer_last/'
    data_path = './data/'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    metric = load_metric("accuracy")
    
    # Load datasets
    dataset = load_dataset('csv',data_files={'train':data_path+'train.csv','val':data_path+'val.csv','test':data_path+'test.csv'})

    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,do_lower_case=True,model_max_length=256)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=3)

    # Setup tokenized dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True,fn_kwargs={'tokenizer':tokenizer})
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_datasets["val"].shuffle(seed=42)
    small_test_dataset = tokenized_datasets["test"].shuffle(seed=42)

    # Dataloader and optimizer
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=64)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=small_eval_dataset.num_rows)
    test_dataloader = DataLoader(small_test_dataset,batch_size=small_test_dataset.num_rows)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Num epochs, set lr_scheduler
    num_epochs = 2
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    model.to(device)

    # Declare progess bar, best accuracy variable and history hashmap 
    progress_bar = tqdm(range(num_training_steps))
    best_accuracy = 0
    history = defaultdict(list)
    print('START\n')

    # Training epoch start
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training mode for training set
        model.train()
        losses = []
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            losses.append(loss.item())
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Obtain training accuracy and loss
        train_acc = metric.compute()['accuracy']
        train_loss = np.mean(losses)

        # Eval mode for validation set
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        # Obtain validation accuracy and loss
        val_acc = metric.compute()['accuracy']
        val_loss = loss.item()

        print(f'Train loss {train_loss} accuracy {train_acc}')
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
          # torch.save(model.state_dict(), 'best_model_state.bin')
          model.save_pretrained(model_best_path)
          best_accuracy = val_acc

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)


if __name__=="__main__":
    my_parser = argparse.ArgumentParser(description= 'Run train/eval/inference mode')
    my_parser.add_argument('--train',action='store_true')
    my_parser.add_argument('--eval',action='store_true')
    my_parser.add_argument('--inference',action='store_true')
    args = my_parser.parse_args()
    if args.train:
        train()

    if args.eval:
        eval()

    if args.inference:
        inference()

    print('\nDONE')
