import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['WANDB_PROJECT'] = 'orca_predicate_logic'
import subprocess as sp
from threading import Thread, Timer
import sched, time

import numpy as np
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


hf_token = ""
wandb_api_key = ""
run_name="orca-trial-1" # change for each run.

from huggingface_hub import login
wandb.login(key=wandb_api_key)
login(token=hf_token)


##########################
### Loading base model ###
##########################

model_name_or_path = "psmathur/orca_mini_v3_13b"
checkpoint_name = "qlora_logic-t1_orca-mini-v3-13b" # t1 = task 1
output_dir = "checkpoints/orca-13b/" + checkpoint_name

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0, # default
    llm_int8_has_fp16_weight=False, # false when using 4bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.float16,
    device_map='auto',
    quantization_config=bnb_config,
    trust_remote_code=True,
    )
model = prepare_model_for_int8_training(model) # change soon to prepare_model_for_kbit_training()

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

print(model) # note down the relevant matrices for LoRA

######################
### Get LoRA model ###
######################

# LoRA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], #maybe also try: "dense", "dense_h_to_4h", "dense_4h_to_h",
    inference_mode=False, 
    r=16,
    lora_alpha=32,
    lora_dropout=0.05
) 

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


#################################
### Load & Preprocess Dataset ###
#################################

data_files = {
    "all_data": "Task1-traintest-dataset.jsonl"
}

ds = load_dataset("json", split='all_data[0:10000]', data_files=data_files)#, cache_dir="/project/OML/bachelor_theses/sdoebele/scripts/llama/cache/")

# for Causal LMs, we feed the whole Input+Target sequence.
new_column = [inp + tar for inp, tar in zip(ds["Input"], ds["Target"])]
ds = ds.add_column("InputTarget", new_column)

ds.train_test_split(test_size=0.6)
train_test_dataset = ds.train_test_split(test_size=0.4, seed = 200)
test_valid = train_test_dataset['test'].train_test_split(test_size=0.5, seed = 200)
train_test_valid_dataset = DatasetDict({
    'train': train_test_dataset['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']
})

print(f"dataset: {train_test_valid_dataset['train'][0]}")

def preprocess_function_for_CausalLMs(element):
    model_inputs = tokenizer(element["InputTarget"])
    return model_inputs

processed_datasets = train_test_valid_dataset.map(
    preprocess_function_for_CausalLMs,
    batched=True,
    num_proc=32,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
dev_dataset = processed_datasets["valid"]

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

################
### Training ###
################


training_args = TrainingArguments(
        output_dir=output_dir,
        eval_steps=20,
        evaluation_strategy='steps',
        save_steps=20,
        save_strategy='steps',
        logging_steps=20,
        logging_strategy='steps',
        logging_dir=f"{output_dir}/logs", 
        learning_rate=2e-5, # try 1e-3, 2e-4
        per_device_train_batch_size=8, # try 4, 16
        per_device_eval_batch_size=8, # try 4, 16
        weight_decay=0.01, # necessary?
        save_total_limit=3, # total number of checkpoints saved, deletes older ones
        num_train_epochs=10, # try 1000
        fp16=True,
        push_to_hub=False,
        auto_find_batch_size=True,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=checkpoint_name
)


trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
)

trainer.train()# resume_from_checkpoint = True

wandb.finish()
