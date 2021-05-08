#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:40:03 2020

@author: sarroutim2
"""


import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim

import numpy as np
import pandas as pd
import os
from collections import defaultdict
from textwrap import wrap
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,classification_report

class BERTClassifier(nn.Module):
  def __init__(self, model_spec, model_dir, device):
        
    """Constructor for BERTClassifier class.

    Args:
      model_spec: A str to pass into the `pretrained_model_name_or_path`
        argument of `transformers.BertForSequenceClassification.from_pretrained`
        (e.g. `"bert-base-uncased"` or a path to a previously trained model) 
      model_dir: str, directory to save and load model checkpoints.
      device: `torch.device` on which the model should be run.
    """
    super(BERTClassifier, self).__init__()
    if isinstance(model_spec, str):
      self._model = transformers.BertForSequenceClassification.from_pretrained(
          model_spec, num_labels = 3
      )
    elif isinstance(model_spec, transformers.BertConfig):
      self._model = transformers.BertForSequenceClassification(model_spec, num_labels = 3)
    else:
      raise ValueError("model_spec should be a string or T5Config.")
    
    self._model_dir = model_dir
    self._device = device
    if self._device.type == "cuda":
      self._model.cuda()
    self.load_latest_checkpoint(model_dir)
  
  def load_latest_checkpoint(self, model_dir):
      if len(os.listdir(model_dir))!=0:
          self._model.load_state_dict(torch.load(model_dir+'best_model_state.bin'))
  def model(self):
    return self._model
  

  def train(
        self,
        train_data_loader,
        val_data_loader,
        optimizer,
        scheduler,
        epochs,
        checkpoint,
        df_train,
        df_eval
        ):
    def train_epoch(
          self,
          train_data_loader,
          optimizer,
          device,
          scheduler,
          n_examples
                    ):
          model=self._model.train()
          losses = []
          correct_predictions = 0
          for d in train_data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              labels=targets
            )
            
            _, preds = torch.max(outputs[1], dim=1)
            loss = outputs[0]            
            correct_predictions += torch.sum(preds == targets)
        
            losses.append(loss.item())
        
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
    
          return correct_predictions.double() / n_examples, np.mean(losses)
    
    def eval_model(self, data_loader, device, n_examples):
          model = self._model.eval()
          losses = []
          correct_predictions = 0
          with torch.no_grad():
            for d in data_loader:
              input_ids = d["input_ids"].to(device)
              attention_mask = d["attention_mask"].to(device)
              targets = d["targets"].to(device)
              outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels = targets
              )
              _, preds = torch.max(outputs[1], dim=1)
              loss = outputs[0]
              print(preds)
              #print(targets)
              correct_predictions += torch.sum(preds == targets)
              losses.append(loss.item())
          return correct_predictions.double() / n_examples, np.mean(losses)  
    history = defaultdict(list)
    best_accuracy = 0
    
    for epoch in range(epochs):
      print(f'Epoch {epoch + 1}/{epochs}')
      print('-' * 10)
      train_acc, train_loss = train_epoch(
        self,
        train_data_loader,
        optimizer,
        self._device,
        scheduler,
        df_train
      )
      print(f'Train loss {train_loss} accuracy {train_acc}')
      val_acc, val_loss = eval_model(
        self,
        val_data_loader,
        self._device,
        df_eval
      )
      print(f'Val   loss {val_loss} accuracy {val_acc}')
      print()
      history['train_acc'].append(train_acc)
      history['train_loss'].append(train_loss)
      history['val_acc'].append(val_acc)
      history['val_loss'].append(val_loss)
      if val_acc > best_accuracy:
        torch.save(self._model.state_dict(), checkpoint+'best_model_state.bin')
        best_accuracy = val_acc   
    
  def evaluate(
        self,
        test_data_loader,
        checkpoint,
        device,
        n_examples
        ):
          model=self._model.eval()
          losses = []
          correct_predictions = 0
          with torch.no_grad():
            for d in test_data_loader:
              input_ids = d["input_ids"].to(device)
              attention_mask = d["attention_mask"].to(device)
              targets = d["targets"].to(device)
              outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels = targets
              )
              _, preds = torch.max(outputs[1], dim=1)
              loss = outputs[0]
              print(preds)
              #print(targets)
              correct_predictions += torch.sum(preds == targets)
              losses.append(loss.item())
          return correct_predictions.double() / n_examples, np.mean(losses)  

  def evaluate_2(
        self,
        test_data_loader,
        checkpoint,
        device,
        n_examples
        ):
          model=self._model.eval()
          losses = []
          correct_predictions = 0
          preds_eval=[]
          target_eval=[]
          claims_texts = []
          evidences_texts = []
          target_names=['Supports','Refutes', 'Neutral']
          with torch.no_grad():
            for d in test_data_loader:
              texts_a = d["text_a"]
              texts_b = d["text_b"]
              input_ids = d["input_ids"].to(device)
              attention_mask = d["attention_mask"].to(device)
              targets = d["targets"].to(device)
              outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels = targets
              )
              _, preds = torch.max(outputs[1], dim=1)
              loss = outputs[0]
              #print(preds)
              #print(preds.cpu())
              #print(preds.cpu().numpy().tolist())
              #print(targets.cpu().numpy().tolist())
              preds_eval.extend(preds.cpu().numpy().tolist())
              target_eval.extend(targets.cpu().numpy().tolist())
              claims_texts.extend(texts_b)
              evidences_texts.extend(texts_a)
              #print(targets)
              
              correct_predictions += torch.sum(preds == targets)
              losses.append(loss.item())
          #print(preds_eval)
          #print(target_eval)
          file_writer=open(self._model_dir+'predictions.txt','w')
          #file_writer_error=open(self._model_dir+'error_analysis.txt','w')
          for evid, claim, gold_label, pred_label in zip (evidences_texts,claims_texts,target_eval,preds_eval):
              file_writer.write(evid+'\t'+claim+'\t'+target_names[gold_label]+'\t'+target_names[pred_label]+'\n')
          
          return accuracy_score(preds_eval,target_eval), f1_score(preds_eval,target_eval,average='macro'),precision_score(preds_eval,target_eval,average='macro'), recall_score(preds_eval,target_eval,average='macro'), classification_report(preds_eval,target_eval,target_names=target_names)
  def predict_row(
        self,
        encoded_claim,
        checkpoint,
        device,
        ):
          model=self._model.eval()

          input_ids = encoded_claim['input_ids'].to(device)
          attention_mask = encoded_claim['attention_mask'].to(device)
          output = model(input_ids, attention_mask)
          #print(output)
          _, prediction = torch.max(output[0], dim=1)
          #labels = [model.config.id2label[label_id] for label_id in prediction.tolist()]
          #print(labels)
          #print(self._model.config.id2label[prediction[0].item()])
          return prediction
      
  def get_predictions(self, data_loader, device):
      model = self._model.eval()
      claims_texts = []
      evidences_texts = []
      predictions = []
      prediction_probs = []
      real_values = []
      with torch.no_grad():
        for d in data_loader:
          texts_a = d["text_a"]
          texts_b = d["text_b"]
          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_mask"].to(device)
          targets = d["targets"].to(device)
          outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
          )
          _, preds = torch.max(outputs[0], dim=1)
          claims_texts.extend(texts_b)
          evidences_texts.extend(texts_a)
          predictions.extend(preds)
          prediction_probs.extend(outputs)
          real_values.extend(targets)
      predictions = torch.stack(predictions).cpu()
      #prediction_probs = torch.stack(prediction_probs).cpu()
      real_values = torch.stack(real_values).cpu()
      return evidences_texts,claims_texts, predictions, real_values