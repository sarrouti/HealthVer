#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:23:32 2020

@author: sarroutim2
"""
import functools

import itertools
import time
from tools import get_dataset
from tools import create_task
from tools import create_mixture

import logging
import argparse
import torch
import transformers
from models import T5Classifier
from models import BERTClassifier

from tools import MisinfoDataset
from tools import create_data_loader

import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import os
from sklearn.utils import shuffle

def train(args):
    label_to_int = {"Refutes": 0, "Supports": 1, "Neutral": 2} ############ adding this line to avoid TypeError: new(): invalid data type 'str'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if args.model_type=='t5':
        if os.path.exists(args.checkpoint+args.model_spec_b)==False:
            os.makedirs(args.checkpoint+args.model_spec_b)
        checkpoint=args.checkpoint+args.model_spec+"/"
        logging.info('Creating T5 model...')
        model = T5Classifier(args.model_spec, checkpoint, device)
        ##create tasks:
        create_task(args.data, 'HealthVer')
        #create mixture of tasks:
        #mixture=['MEDIQA','RQE']
        #create_mixture(mixture,'MEDNLP')
        #train
        ##### total steps
        tokenizer = BertTokenizer.from_pretrained(args.model_spec_b)
        df = pd.read_csv(args.data+"healthver_train.csv")
        df=df.replace({"label": label_to_int})                      #######################  replacing the str values with int values
        train_data_loader = create_data_loader(df, tokenizer, args.sequence_length_inputs, args.batch_size)
        total_steps = len(train_data_loader) * args.epochs
        print(total_steps)
        #####
        model.train(
        mixture_or_task_name="HealthVer",
        steps=total_steps,
        save_steps=args.save_steps,
        sequence_length={"inputs": args.sequence_length_inputs, "targets": args.sequence_length_targets_t5},
        split="train",
        batch_size=args.batch_size,
        optimizer=functools.partial(transformers.AdamW, lr=1e-4),)
    elif args.model_type=='bert':
        if os.path.exists(args.checkpoint+args.model_spec_b)==False:
            os.makedirs(args.checkpoint+args.model_spec_b)
        checkpoint=args.checkpoint+args.model_spec_b+"/"
        #WebVer_train_emnlp
        df = pd.read_csv(args.data+"healthver_train.csv")
        df=df.replace({"label": label_to_int}) ########### replace here as well
        #df = df [:4000] 
        #df = shuffle(df) 
              
        #df.to_csv(args.data+"train_shuffle.csv")
        df_dev = pd.read_csv(args.data+"healthver_dev.csv")
        df_dev=df_dev.replace({"label": label_to_int})   ############### this as well
        
        logging.info('Creating BERT model...')
        model = BERTClassifier(args.model_spec_b, checkpoint, device)
        tokenizer = BertTokenizer.from_pretrained(args.model_spec_b)
        
        
        
        
        
        '''
        encoding = tokenizer.encode_plus(
          'text text',
          'text text',
          add_special_tokens=True,
          max_length=10,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          #return_tensors='pt',
        )
        print("Multi segment token (str): {}".format(tokenizer.convert_ids_to_tokens(encoding['input_ids'])))
        print("Multi segment token (str): {}".format(encoding['input_ids']))'''
        
        
        
        
        
        train_data_loader = create_data_loader(df, tokenizer, args.sequence_length_inputs, args.batch_size)
        
        val_data_loader = create_data_loader(df_dev, tokenizer, args.sequence_length_inputs, args.batch_size)
        
        total_steps = len(train_data_loader) * args.epochs
        
        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=True)
        model.train(
                train_data_loader,
                val_data_loader,
                optimizer = optimizer,
                scheduler = get_linear_schedule_with_warmup(
                                                              optimizer,
                                                              num_warmup_steps=0,
                                                              num_training_steps=total_steps
                                                            ),
                epochs=args.epochs,
                
                checkpoint=checkpoint,
                df_train = len(df),
                df_eval =len (df_dev)
                )
    else:
        raise ValueError("model_type should be either T5 or BERT")
if __name__== '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--model-type', type=str, default='t5',
                        help='model type: bert or T5')
    parser.add_argument('--data', type=str, default='./data/',
                        help='data for each task')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints-healthver/',
                        help='Path for saving trained models')
    parser.add_argument('--model-spec', type=str, default='t5-base',
                        help='--model-spec: A str to pass into the pretrained_model_name_or_path'
                        'argument of `transformers.T5ForConditionalGeneration.from_pretrained'
                        '(e.g. `"t5-base"` "t5_3b_covid" or a path to a previously trained model) or an'
                        'instance of the `transformers.configuration_t5.T5Config` class to use'
                        'to directly construct the `transformers.T5ForConditionalGeneration object.')
    parser.add_argument('--model-spec-b', type=str, default='bert-base-uncased',
                        help='--model-spec: A str to pass into the pretrained_model_name_or_path'
                        'e.g. lordtt13/COVID-SciBERT, allenai/scibert_scivocab_uncased'
                        'mrm8488/scibert_scivocab-finetuned-CORD19'
                        'monologg/biobert_v1.0_pubmed_pmc'      
                        'monologg/biobert_v1.1_pubmed'
                        'lordtt13/COVID-SciBERT'
                        'bert-base-uncased, bert-large-uncased, distilbert-base-cased'
                        '(e.g. `"bert-base-cased, scibert_scivocab_uncased"` ')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Step size for saving trained models')
    parser.add_argument('--steps', type=int, default=100,
                        help='Step size for saving trained models')
    parser.add_argument('--save-steps', type=int, default=2000,
                        help='Step size for saving trained models')
    parser.add_argument('--batch-size', type=int, default=8, 
                        help='16 for bert, 8 for t5')
    parser.add_argument('--sequence-length-inputs', type=int, default=300)#400
    parser.add_argument('--sequence-length-targets', type=int, default=3)
    parser.add_argument('--sequence-length-targets-t5', type=int, default=5)
    args = parser.parse_args()
    train(args)

    
    
    
    
    
