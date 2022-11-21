# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
         
import logging
import sys
import argparse
import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
# Initialize XLA process group for torchrun
import torch_xla.distributed.xla_backend

device = "xla"
torch.distributed.init_process_group(device)
world_size = xm.xrt_world_size() 

# Global constants
WARMUP_STEPS = 2
batch_size = 8
num_epochs = 3

def main():

    dataset = load_from_disk(os.environ["SM_CHANNEL_TRAIN"])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['text'], max_length=128, padding='max_length', truncation=True)
#        return tokenizer(batch['text'], padding='max_length', truncation=True)
    
    # load dataset
    train_dataset = dataset['train'].shuffle().select(range(1000)) # smaller for faster training demo

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)

    
    # set format for pytorch
    train_dataset =  train_dataset.rename_column("label", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Set up distributed data loader
    train_sampler = None
    if world_size > 1: # if more than one core
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas = world_size,
            rank = xm.get_ordinal(),
            shuffle = True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
    )

    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    num_training_steps = num_epochs * len(train_device_loader)
    progress_bar = tqdm(range(num_training_steps))

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
#    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train() 
    for epoch in range(num_epochs):
        for batch in train_device_loader:
            batch = {k: v.to(device) for k, v, in batch.items()}
            outputs = model(**batch)
            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            xm.optimizer_step(optimizer) #gather gradient updates from all cores and apply them
            progress_bar.update(1)
        print(
            "Epoch {}, rank {}, Loss {:0.4f}".format(epoch, xm.get_ordinal(), loss.detach().to("cpu"))
        )
    os.system(f"df -k") # check directory space

    # Save checkpoint for evaluation (xm.save ensures only one process save)
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {'state_dict': model.state_dict()}
    #xm.save(checkpoint,'checkpoints/checkpoint.pt')
    default_dir = os.environ['SM_MODEL_DIR']
    xm.save(checkpoint, f"{default_dir}/checkpoint.pt")
    print('##### Model saved to: ', f"{default_dir}/checkpoint.pt")
    print('----------End Training ---------------')
    
if __name__ == '__main__':
    main()