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

import os
import json
import torch
import torch_neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

JSON_CONTENT_TYPE = 'application/json'

def model_fn(model_dir):
    
    model_dir = '/opt/ml/model/'
    dir_contents = os.listdir(model_dir)
    model_path = next(filter(lambda item: 'model' in item, dir_contents), None)
    
    tokenizer_init = AutoTokenizer.from_pretrained('distilbert-base-uncased', max_length=128)
    model = torch.jit.load(os.path.join(model_dir, model_path))

    
    return (model, tokenizer_init)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return
    

def predict_fn(input_data, models):

    model_bert, tokenizer = models
    
    max_length = 128 
    tokenized_sequence_pair = tokenizer.encode_plus(input_data,
                                                    max_length=max_length,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_tensors='pt')
    
    # Convert example inputs to a format that is compatible with TorchScript tracing
    example_inputs = tokenized_sequence_pair['input_ids'], tokenized_sequence_pair['attention_mask']
    
    with torch.no_grad():
        paraphrase_classification_logits_neuron = model_bert(*example_inputs)
    
    classes = ['negative', 'positive']
    paraphrase_prediction = paraphrase_classification_logits_neuron[0][0].argmax().item()
    out_str = 'BERT predicts that "{}" is {}'.format(input_data, classes[paraphrase_prediction])
    
    return out_str


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
    
