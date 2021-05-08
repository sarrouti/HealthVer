#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:14:39 2020

@author: sarroutim2
"""

import argparse

import functools

import gin

import t5
import torch
import transformers

import gzip
import json
import os
import tensorflow.compat.v1 as tf
import functools
import tensorflow_datasets as tfds
import itertools

def create_task(data, task_name):
    qa_tsv_path = {
    "train": os.path.join(data, task_name+"_train.tsv"),
    "validation": os.path.join(data, task_name+"_dev.tsv"),
    "test": os.path.join(data, task_name+"_test.tsv")
    }

    def qa_dataset_fn(split, shuffle_files=False):
      # We only have one file for each split.
      del shuffle_files
    
      # Load lines from the text file as examples.
      ds = tf.data.TextLineDataset(qa_tsv_path[split])
      # Split each "<question>\t<answer>" example into (question, answer) tuple.
      ds = ds.map(
         functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                            field_delim="\t", use_quote_delim=False,select_cols=[1,2]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      # Map each tuple to a {"question": ... "answer": ...} dict.
      ds = ds.map(lambda *ex: dict(zip(["inputs", "label"], ex)))
      return ds
    
    def mediqa_preprocessor(ds):
      def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
        return text
    
      def to_inputs_and_targets(ex):
        """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
        return {
            "inputs":
                 tf.strings.join(
                     [task_name+": ", normalize_text(ex["inputs"])]),
            "targets": normalize_text(ex["label"])
        }
      return ds.map(to_inputs_and_targets, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    t5.data.TaskRegistry.add(
    task_name,
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=qa_dataset_fn,
    splits=["train","validation", "test"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[mediqa_preprocessor],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text, 
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    #num_input_examples=200
    )

def create_mixture(tasks, name):
    
    t5.data.MixtureRegistry.remove(name)
    t5.data.MixtureRegistry.add(
    name,
    tasks,
     default_rate=1.0)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--data', type=str, default='data/',
                        help='data for each task')

    args = parser.parse_args()
    create_task(args.data, 'misinfo_task')
    '''nq_task = t5.data.TaskRegistry.get("mediqa_task")
    ds = get_dataset('mediqa_task', {"inputs": 128, "targets": 32}, split="train", batch_size=8)
    c=0
    for d in ds:
        print(d["inputs"])
        c+=1

    print(c)'''
    

 