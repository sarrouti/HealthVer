import functools

import itertools
import time
from tools import get_dataset
from tools import create_task
from tools import create_mixture

import logging
import argparse
import torch
from models import T5Classifier
from models import BERTClassifier
from tools import create_data_loader
from transformers import BertModel, BertTokenizer 

import numpy as np
import pandas as pd

def evaluate(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if args.model_type=='t5':
        checkpoint=args.checkpoint+args.model_spec+"/"
        model = T5Classifier(args.model_spec, checkpoint, device)
        ##create tasks:
        create_task(args.data, 'HealthVer')
        #create mixture of tasks:
        #mixture=['MEDIQA','RQE']
        #create_mixture(mixture,'MEDNLP')
        #train
        
        model.eval(
        "HealthVer",
        checkpoint_steps=args.checkpoint_steps,
        sequence_length={"inputs": args.sequence_length_inputs, "targets": args.sequence_length_targets_t5},
        split="test",
        batch_size=args.batch_size,
        )
    elif args.model_type=='bert':

        checkpoint=args.checkpoint+args.model_spec_b+"/"
        df = pd.read_csv(args.data+"healthver_test.csv")
        logging.info('Creating BERT model...')
        model = BERTClassifier(args.model_spec_b, checkpoint, device)
        tokenizer = BertTokenizer.from_pretrained(args.model_spec_b)
        
        test_data_loader = create_data_loader(df, tokenizer, args.sequence_length_inputs, args.batch_size)

        test_acc, f1_score, precision, recall, classification_report = model.evaluate_2(
                test_data_loader,
                checkpoint = checkpoint,
                device = device,
                n_examples = len(df)
                )
        
        print('Accuracy: ',test_acc)
        print('F1-score: ',f1_score)
        print('Precision: ',precision)
        print('Recall: ',recall)
        print('classification_report: \n',classification_report)



        
    else:
        raise ValueError("model_type should be either T5 or BERT")


if __name__== '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--model-type', type=str, default='t5',
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
    parser.add_argument('--model-spec-b', type=str, default='bert-base-uncased',
                        help='--model-spec: A str to pass into the pretrained_model_name_or_path'
                        'argument of `transformers.T5ForConditionalGeneration.from_pretrained'
                        '(e.g. `"t5-base"` or a path to a previously trained model) or an'
                        'allenai/scibert_scivocab_uncased'
                        'monologg/biobert_v1.1_pubmed'
                        'instance of the `transformers.configuration_t5.T5Config` class to use'
                        'to directly construct the `transformers.T5ForConditionalGeneration object.')
    parser.add_argument('--checkpoint-steps', type=str, default='all',
                        help='Step size for saving trained models')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--sequence-length-inputs', type=int, default=300)
    parser.add_argument('--sequence-length-targets', type=int, default=3)
    parser.add_argument('--sequence-length-targets-t5', type=int, default=6)
    args = parser.parse_args()
    evaluate(args)