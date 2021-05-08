#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:48:00 2020

@author: sarroutim2
"""

import mesh_tensorflow.transformer.dataset as transformer_dataset
import tensorflow.compat.v1 as tf
import t5.data
import tensorflow_datasets as tfds
import argparse
from torch.utils.data import Dataset, DataLoader
import torch

class MisinfoDataset(Dataset):
  def __init__(self, texts_a, texts_b, targets, tokenizer, max_len):
    self.texts_a = texts_a
    self.texts_b = texts_b    
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  def __len__(self):
    return len(self.texts_a)
  def __getitem__(self, item):
    text_a = str(self.texts_a[item])
    text_b = str(self.texts_b[item])
    
    target = self.targets[item]
    #print(target)
    #print(item)
    #print(torch.tensor(target, dtype=torch.long))
    encoding = self.tokenizer.encode_plus(
      text_a,
      text_b,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'text_a': text_a,
      'text_b': text_b,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = MisinfoDataset(
    texts_a=df.evidence.to_numpy(),
    texts_b=df.claim.to_numpy(),
    targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8
  )



def tokens_to_batches(dataset, sequence_length, batch_size, output_features):
  """Convert a dataset of token sequences to batches of padded/masked examples.

  Args:
    dataset: tf.data.Dataset containing examples with token sequences.
    sequence_length: dict of int, a dict mapping feature name to length.
    batch_size: int, the number of padded sequences in each batch.
    output_features: list of str, features to include in the dataset.

  Returns:
    A generator that produces batches of numpy examples.
  """
  dataset = transformer_dataset.pack_or_pad(
      dataset,
      sequence_length,
      pack=False,
      feature_keys=output_features,
      ensure_eos=True,
  )

  def _map_fn(ex):
    for key in output_features:
      tensor = ex[key]
      mask = tf.cast(tf.greater(tensor, 0), tensor.dtype)
      ex[key + "_mask"] = mask
    return ex

  dataset = dataset.map(
      _map_fn,
      num_parallel_calls=t5.data.preprocessors.num_parallel_calls()
  )

  dataset = dataset.batch(batch_size, drop_remainder=False)
  return tfds.as_numpy(dataset)

def get_dataset(mixture_or_task_name, sequence_length, split, batch_size):
  """Get a generator of numpy examples for a given Task or Mixture.

  Args:
    mixture_or_task_name: str, the name of the Mixture or Task to train on.
      Must be pre-registered in the global `t5.data.TaskRegistry` or
      `t5.data.MixtureRegistry.`
    sequence_length: dict of int, a dict mapping feature name to length.
    split: str or `tensorflow_datasets.Split`, the data split to load.
    batch_size: int, the number of padded sequences in each batch.

  Returns:
    A generator that produces batches of numpy examples.
  """
  task = t5.data.get_mixture_or_task(mixture_or_task_name)
  ds = task.get_dataset(sequence_length, split)
  return tokens_to_batches(
      ds, sequence_length, batch_size, tuple(task.output_features)
  )


def write_lines_to_file(lines, filename):
  """Write each line to filename, replacing the file if it exists."""
  if tf.io.gfile.exists(filename):
    tf.io.gfile.remove(filename)
  with tf.io.gfile.GFile(filename, "w") as output_file:
    output_file.write("\n".join([str(l) for l in lines]))





