import os
import sys
module_path = os.path.abspath(os.path.join('Thesis/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from transformers import (
    AutoTokenizer,
    AutoModel,
    T5Tokenizer, 
    T5ForConditionalGeneration,
    pipeline,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

from datasets import load_dataset
import evaluate
from evaluate import evaluator

from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

import torch
import sklearn


import nltk
from nltk.sem.logic import LogicParser
from nltk.sem.evaluate import Valuation, Model
from nltk.sem.logic import *
import json
import pandas as pd
import subprocess


from huggingface_hub import login
login(token="")


# Logging
subprocess.run("nvidia-smi", shell=True)
subprocess.run("pip list", shell=True)
torch.cuda.is_available()

# Choose how many datapoints to evaluate. Max: 100k
number_datapoints = 1000

# Utilities
def make_predictions(dataset, model, tokenizer):
    predictions = []
    for datapoint in dataset["FewShot-2"]:
        input_ids = tokenizer(datapoint, return_tensors="pt", padding=True).input_ids.to("cuda")
        outputs = model.generate(input_ids)
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print("Output:" + outputs + "-----Target:" + ref)
        predictions.append(outputs)
    return predictions

def make_predictions_CausalLM(dataset, model, tokenizer):
    # sets max new tokens, needed for CausalLMs
    predictions = []
    for datapoint in dataset["FewShot-2"]:
        input_ids = tokenizer(datapoint, return_tensors="pt", padding=True).input_ids.to("cuda")
        outputs = model.generate(input_ids, max_new_tokens=200) # max_new_tokens set so that not just prompt is returned
        outputs = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:])[0] # so to not return prompt at all: regarding slicing: see: https://github.com/huggingface/transformers/issues/17117
        predictions.append(outputs)
    return predictions

def save_task1_to_json(dataset, predictions, filename):
    target_sats = dataset['Target-sat']
    valuation = dataset['Valuation']
    d = {'Predictions':predictions,'Target-sat':target_sats, 'Valuation': valuation}
    df_task = pd.DataFrame(d)
    df_task.to_json(filename)

def save_task2_to_json(dataset, predictions, filename):
    target_sats = dataset["Target-sat"]
    formulas = dataset["Formulas"]
    d = {'Predictions': predictions,'Target-sat':target_sats, 'Formulas': formulas}
    df_task = pd.DataFrame(d)
    df_task.to_json(filename)

def save_task3_to_json(dataset, predictions, filename):
    references = dataset['Target']
    d = {'Predictions':predictions,'References':references}
    df_task = pd.DataFrame(d)
    df_task.to_json(filename)

##########################
# TASK: 1
##########################

task1_dataset = pd.read_json('Task1-fs2-dataset.json')
task1_dataset = task1_dataset.iloc[0:number_datapoints, :]

##########################
# TASK: 2
##########################

task2_dataset = pd.read_json('Task2-fs2-dataset.json')
task2_dataset = task2_dataset.iloc[0:number_datapoints, :]

##########################
# TASK: 3 
##########################

task3_dataset = pd.read_json('Task3-fs2-dataset.json')
task3_dataset = task3_dataset.iloc[0:number_datapoints, :]





##########################
#### MODEL: Flan-ul2 #####
##########################

model_name = "google/flan-ul2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token
model = T5ForConditionalGeneration.from_pretrained(model_name, load_in_8bit=True)

# Task 1
predictions = make_predictions(task1_dataset, model, tokenizer)
save_task1_to_json(task1_dataset, predictions, "flan-ul2_task1-fs2.json")

# Task 2
predictions = make_predictions(task2_dataset, model, tokenizer)
save_task2_to_json(task2_dataset, predictions, "flan-ul2_task2-fs2.json")

# Task 3
predictions = make_predictions(task3_dataset, model, tokenizer)
save_task3_to_json(task3_dataset, predictions, "flan-ul2_task3-fs2.json")




##########################
#### MODEL: Falcon-7b-instruct #####
##########################

model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, load_in_8bit=False).to("cuda")


# Task 1
predictions = make_predictions_CausalLM(task1_dataset, model, tokenizer)
save_task1_to_json(task1_dataset, predictions, "falcon-7b_task1-fs2.json")

# Task 2
predictions = make_predictions_CausalLM(task2_dataset, model, tokenizer)
save_task2_to_json(task2_dataset, predictions, "falcon-7b_task2-fs2.json")

# Task 3
predictions = make_predictions_CausalLM(task3_dataset, model, tokenizer)
save_task3_to_json(task3_dataset, predictions, "falcon-7b_task3-fs2.json")






##########################
#### MODEL: Llama-2-13b-chat-hf #####
##########################

model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, load_in_8bit=False).to("cuda")

# Task 1
predictions = make_predictions_CausalLM(task1_dataset, model, tokenizer)
save_task1_to_json(task1_dataset, predictions, "Llama-2-13b-chat-hf_task1-fs2.json")

# Task 2
predictions = make_predictions_CausalLM(task2_dataset, model, tokenizer)
save_task2_to_json(task2_dataset, predictions, "Llama-2-13b-chat-hf_task2-fs2.json")

# Task 3
predictions = make_predictions_CausalLM(task3_dataset, model, tokenizer)
save_task3_to_json(task3_dataset, predictions, "Llama-2-13b-chat-hf_task3-fs2.json")
