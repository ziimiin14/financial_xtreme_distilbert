import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments
import os
import torch
import pandas as pd
from datasets import load_metric,load_dataset
from sklearn.metrics import confusion_matrix, classification_report,roc_curve,auc
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from collections import defaultdict
import time
import argparse

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def inference():
    pass


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

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    # Setup tokenized dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
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

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    # Setup tokenized dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
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
