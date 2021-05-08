#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:57:13 2020

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
from transformers import BertModel, BertTokenizer 
from tools import create_data_loader
import pandas as pd

def predict(args):

    inputs = [
   "Recent research results suggest that bats or pangolins might be the original hosts for the virus based on comparative studies using its genomic sequences.",
   'The coronavirus may have originated in a Chinese laboratory'
]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if args.model_type=='t5':

        model = T5Classifier(args.model_spec, args.checkpoint, device)
    
        model.predict(
        inputs,
        sequence_length={"inputs": args.sequence_length_inputs},
        batch_size=args.batch_size,
        output_file=args.output_file,
        )
    elif args.model_type=='bert':
        
        checkpoint=args.checkpoint+args.model_spec_b+"/"
        model = BERTClassifier(args.model_spec_b, checkpoint, device)
        tokenizer = BertTokenizer.from_pretrained(args.model_spec_b)
        '''
        Get predictions for a simple text
        '''
        encoded_claim = tokenizer.encode_plus(
                  inputs[0],
                  inputs[1],
                  max_length=args.sequence_length_inputs,
                  add_special_tokens=True,
                  return_token_type_ids=False,
                  pad_to_max_length=True,
                  return_attention_mask=True,
                  return_tensors='pt',
                )

        prediction = model.predict_row(
                encoded_claim = encoded_claim,
                checkpoint = checkpoint,
                device = device
                                  )
        labels=['SUPPORTED', 'REFUTED','NOINFO']
        print(f'Claim: {inputs[1]}')
        print(f'Evidence: {inputs[1]}')
        print(f'Label  : {labels[prediction[0]]}')
        '''
        Get predictions for test set 
        This is similar to the evaluation function, except that we’re storing the text of the claims, evidences 
        and the predicted probabilities:
        '''

        df = pd.read_csv(args.data+"dina_test.csv")
        test_data_loader = create_data_loader(df, tokenizer, args.sequence_length_inputs, args.batch_size)

        y_evidences_texts,y_claims_texts, y_pred, y_test = model.get_predictions(
                                                                      test_data_loader,
                                                                      device
                                                                    )
        #print(y_claims_texts)   
        #print(y_evidences_texts)  
        #print(y_pred)  
        #print(y_test)
        for claim, evidence, pred_label, gold_label in zip(y_evidences_texts,y_claims_texts,y_pred,y_test):
            if labels[pred_label.item()]!=labels[gold_label.item()]:
                print('CLAIM: ', claim)
                print('EVIDENCE: ', evidence)
                print('pred_label: ', labels[pred_label.item()])
                print('gold_label: ', labels[gold_label.item()])
            
                                                                  
    else:
        raise ValueError("model_type should be either T5 or BERT")
        
if __name__== '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
     # Session parameters.
    parser.add_argument('--model-type', type=str, default='bert',
                        help='model type: BERT or T5')
    parser.add_argument('--data', type=str, default='./data/',
                        help='data for each task')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints-healthver/',
                        help='Path for saving trained models')
    parser.add_argument('--model-spec', type=str, default='t5-base',
                        help='--model-spec: A str to pass into the pretrained_model_name_or_path'
                        'argument of `transformers.T5ForConditionalGeneration.from_pretrained'
                        '(e.g. `"t5-base"` or a path to a previously trained model) or an'
                        'instance of the `transformers.configuration_t5.T5Config` class to use'
                        'to directly construct the `transformers.T5ForConditionalGeneration object.')
    parser.add_argument('--model-spec-b', type=str, default='allenai/scibert_scivocab_uncased',
                        help='--model-spec: A str to pass into the pretrained_model_name_or_path'
                        'argument of `transformers.T5ForConditionalGeneration.from_pretrained'
                        '(e.g. `"t5-base"` or a path to a previously trained model) or an'
                        'instance of the `transformers.configuration_t5.T5Config` class to use'
                        'to directly construct the `transformers.T5ForConditionalGeneration object.')
    parser.add_argument('--checkpoint-steps', type=str, default='all',
                        help='Step size for saving trained models')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--sequence-length-inputs', type=int, default=300)
    parser.add_argument('--sequence-length-targets', type=int, default=3)
    args = parser.parse_args()
    predict(args)