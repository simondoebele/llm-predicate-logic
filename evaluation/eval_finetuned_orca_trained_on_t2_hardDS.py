import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import subprocess as sp
from threading import Thread, Timer
import sched, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    default_data_collator, 
    DataCollatorForLanguageModeling, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback #get_linear_schedule_with_warmup
)

from peft import (
    get_peft_config, 
    get_peft_model, 
    get_peft_model_state_dict, 
    LoraConfig, 
    TaskType,
    PeftModel,
    PeftConfig,
    prepare_model_for_int8_training
)

from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import wandb
import json
import subprocess

subprocess.run("nvidia-smi", shell=True)

##########################
### Loading base model ###
##########################

model_name_or_path = "psmathur/orca_mini_v3_13b"
checkpoint_name = "qlora_logic-t2_orca-mini-v3-13b/" # t1 = task 1
output_dir = "checkpoints/orca-13b/" + checkpoint_name
cp = "checkpoint-7500"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0, # default
    llm_int8_has_fp16_weight=False, # false when using 4bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

peft_model_id = output_dir + cp
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, torch_dtype=torch.float16, load_in_8bit=False, device_map='auto', quantization_config=bnb_config, trust_remote_code=True)

lora_model = PeftModel.from_pretrained(model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

print(lora_model) # note down the relevant matrices for LoRA


# Choose how many datapoints to evaluate.
number_datapoints = 1000

# Utilities
def make_predictions(dataset, model, tokenizer):
    predictions = []
    for datapoint in dataset["Input"]:
        input_ids = tokenizer(datapoint, return_tensors="pt", padding=True).input_ids.to("cuda")
        outputs = model.generate(input_ids)
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(outputs)
    return predictions

def make_predictions_CausalLM(dataset, model, tokenizer):
    # sets max new tokens, needed for CausalLMs
    predictions = []
    for datapoint in dataset["Input"]:
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
    
    
#choose e.g. 80000 for testing (we did not train on this!)
#task1_dataset = pd.read_json('Task1-dataset-hard.jsonl', lines=True)
#task1_dataset = task1_dataset.iloc[0:0+number_datapoints, :]

task2_dataset = pd.read_json('Task2-dataset-hard.jsonl', lines=True)
task2_dataset = task2_dataset.iloc[0:0+number_datapoints, :]

#task3_dataset = pd.read_json('Task3-dataset-hard.jsonl', lines=True)
#task3_dataset = task3_dataset.iloc[0:0+number_datapoints, :]

##########################
# Generate
##########################

# Task 1
#predictions = make_predictions_CausalLM(task1_dataset, model, tokenizer)
#save_task1_to_json(task1_dataset, predictions, "orca-13b_trained_on_t2_task1_hard.json")

# Task 2
predictions = make_predictions_CausalLM(task2_dataset, model, tokenizer)
save_task2_to_json(task2_dataset, predictions, "orca-13b_trained_on_t2_task2_hard.json")

# Task 3
#predictions = make_predictions_CausalLM(task3_dataset, model, tokenizer)
#save_task3_to_json(task3_dataset, predictions, "orca-13b_trained_on_t2_task3_hard.json")