import subprocess
import os
# by default setting to kaggle environment
DEVICE_IDS = [0, 1]
ROOT_DIR = '/kaggle/working/Python-Code-Generation'    
    
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    try:
        subprocess.run(['bash', f"{ROOT_DIR}/start.sh", ROOT_DIR], check=True)
        print("Shell script executed successfully in kaggle environment.")
    except subprocess.CalledProcessError as e:
        print("Error running shell script in kaggle environment:", e)
    print("Running in Kaggle environment")
else:
    DEVICE_IDS = [0, 1, 2, 4]
    ROOT_DIR = "."
    try:
        subprocess.run(['bash', f"{ROOT_DIR}/start.sh", ROOT_DIR], check=True)
        print("Shell script executed successfully in Nvidia DGX A100 environment.")
    except subprocess.CalledProcessError as e:
        print("Error running shell script in Nvidia DGX A100 environemnt:", e)
    print("Running in Nvidia DGX A100 environment")

# Loading required library
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
from tqdm.autonotebook import tqdm


import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import T5Tokenizer, RobertaTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

import nltk
nltk.download('punkt')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge 
from codebleu import calc_codebleu 

# Code Control
MODEL_NAME = "t5-base"
MAX_INPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 512
BATCH_SIZE = 8
EPOCHS = 2
DEVICE_IDS = [0, 1]

if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    print("Running in Kaggle environment")
else:
    DEVICE_IDS = [0, 1, 2, 4]
    print("Not running in Kaggle environment")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        
# helper function
def save_metrics_to_excel(metrics_dict, file_name):
    if os.path.exists(file_name):
        # If file exists, load the existing data
        df_old = pd.read_excel(file_name)
        # Convert new data to DataFrame
        df_new = pd.DataFrame.from_dict(metrics_dict)
        # Append new data to old data
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        # If file doesn't exist, create a new DataFrame
        df = pd.DataFrame.from_dict(metrics_dict)
        
    df.to_excel(file_name, index = False)

# Stage1 : Dataset Preprocessing

data_text_length = 0
data_code_length = 0
data_file_path = f"{ROOT_DIR}/dataset/python_code_merge_input_output.jsonl"
with open(data_file_path, 'r') as f:
    for line in f:
        # Load each line as a JSON object
        data = json.loads(line)
        
        # Extract text and code
        text = data['text_input']
        code = data['python_code']
        
        # Update lengths if necessary
        text_length = len(text)
        code_length = len(code)
        
        if text_length > data_text_length:
            data_text_length = text_length
        
        if code_length > data_code_length:
            data_code_length = code_length

# Count the number of lines in the file
with open(data_file_path, 'r') as f:
    data_length = sum(1 for _ in f)

print("Length of data:", data_length)
print("Maximum length of text:", data_text_length)
print("Maximum length of code:", data_code_length)


class TaskDataset(Dataset):
    def __init__(self, file_path, tokenizer, input_name, output_name, data_text_length, data_code_length):
        self.tokenizer = tokenizer
        self.data_text_length = data_text_length
        self.data_code_length = data_code_length
        self.input_name = input_name
        self.output_name = output_name
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        #task = row['text']
        #code = row['code']
        task = row[self.input_name]
        code = row[self.output_name]

        inputs = self.tokenizer.encode_plus(
            task, 
            max_length=self.data_text_length, 
            padding='max_length', 
            truncation=True, 
            
        )
        outputs = self.tokenizer.encode_plus(
            code, 
            max_length=self.data_code_length, 
            padding='max_length', 
            truncation=True, 
            
        )
        input_ids = torch.tensor(inputs.input_ids)
        output_ids = torch.tensor(outputs.input_ids)
        return input_ids, output_ids
    

# Stage2 : Setting up Mixed Precision Strategy

global device
def set_strategy(model, tpu, gpu):
    if tpu:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        model.to(device)
        print("TPU strategy setup complete.")

    elif gpu:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"GPU strategy setup complete with {gpu_count} GPUs!")
            #model = torch.nn.DataParallel(model, device_ids=DEVICE_IDS)
            torch.distributed.init_process_group(backend='nccl')
            model.to(device)
            model = DistributedDataParallel(model, device_ids=[DEVICE_IDS], output_device=DEVICE_IDS)
            
        elif gpu_count == 1:
            model.to(device)
            print(f"GPU strategy setup complete with {gpu_count} GPU!")
            
        else:
            print(f"CPU strategy setup complete.")
            model.to(device)
    else:
        print(f"CPU strategy setup complete.")
        model.to(device)
    return model, device
model, device = set_strategy(model, tpu = False, gpu = True)
model, device


# Stage3: Split of data
dataset = TaskDataset(data_file_path, tokenizer, "text_input", "python_code", data_text_length = MAX_INPUT_TOKENS, data_code_length = MAX_OUTPUT_TOKENS)
TRAIN_SIZE =  int(0.8 * len(dataset))
VAL_SIZE = int(0.1 * len(dataset))
TEST_SIZE = len(dataset) - TRAIN_SIZE - VAL_SIZE
train_dataset, val_dataset, test_dataset = random_split(dataset, [TRAIN_SIZE, VAL_SIZE, TEST_SIZE])
print(f"Total length dataset :{len(dataset)}")
print(f'Train dataset: {len(train_dataset)} \nValidation dataset: {len(val_dataset)}\nTest dataset:{len(test_dataset)}')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Stage4: Performance Metrics

def calculate_metrics(reference, candidate, tokenizer):
    # Tokenize the reference and candidate
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    
    #This method uses the p-norm method mentioned in the BLEU SmoothingFunction paper
    smoothie = SmoothingFunction().method4
    # Calculate BLEU score
    bleu_score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)

    # Calculate Rouge score
    rouge = Rouge()
    scores = [{ "rouge-1": {"f": 0., "p": 0.,"r": 0.}, "rouge-2": {"f": 0., "p": 0., "r": 0.}, "rouge-l": {"f": 0., "p": 0.0, "r": 0.}}]
    try:
        if len(candidate)>0 and len(reference) > 0:
            scores = rouge.get_scores(candidate, reference)
    except:
        print('can:', candidate)
        print('ref:', reference)
    
    
    # Calculate METEOR score
    meteor = meteor_score([reference_tokens], candidate_tokens)
    
    
    # Calculate CodeBLEU score
    codebleu = calc_codebleu([reference], [candidate], lang="python",tokenizer = tokenizer)

    return bleu_score, scores[0], meteor, codebleu  # scores[0] contains Rouge-1, Rouge-2 and Rouge-L


def calculate_batch_metrics(references, candidates, tokenizer):
    batch_bleu_scores = []
    batch_rouge_scores = []
    batch_meteor_scores = []
    batch_codebleu_scores = []

    for reference, candidate in zip(references, candidates):
        bleu_score, rouge_score, meteor_score, codebleu_score = calculate_metrics(reference, candidate, tokenizer)
        
        batch_bleu_scores.append(bleu_score)
        batch_rouge_scores.append(rouge_score)
        batch_meteor_scores.append(meteor_score)
        batch_codebleu_scores.append(codebleu_score)
        
    return batch_bleu_scores, batch_rouge_scores, batch_meteor_scores, batch_codebleu_scores


# Stage5: Model Training 
def train_model(model, train_dataloader, validation_dataloader, test_dataloader, tokenizer, epochs):
    
    # OPTIMIZER PARAMETER
    learning_rate = 3e-4
    weight_decay = 1e-4
    adam_epsilon = 1e-8
    
    gpu_count = torch.cuda.device_count()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay, eps = 1e-8)
    # Add an LR scheduler and a system to save the best model
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
    best_val_loss = float('inf')
    global model_save_path
    
    # METRICS SAVING 
    metrics_dict = {
        'Epoch': [],
        'Training Loss': [],
        'Validation Loss': [],
        'Training BLEU': [],
        'Validation BLEU': [],
        'Training METEOR': [],
        'Validation METEOR': [],
        'Training CodeBLEU': [],
        'Validation CodeBLEU': [],
        'Training ROUGE-1-f': [],
        'Validation ROUGE-1-f': [],
        'Training ROUGE-1-p': [],
        'Validation ROUGE-1-p': [],
        'Training ROUGE-1-r': [],
        'Validation ROUGE-1-r': [],
        'Training ROUGE-2-f': [],
        'Validation ROUGE-2-f': [],
        'Training ROUGE-2-p': [],
        'Validation ROUGE-2-p': [],
        'Training ROUGE-2-r': [],
        'Validation ROUGE-2-r': [],
        'Training ROUGE-L-f': [],
        'Validation ROUGE-L-f': [],        
        'Training ROUGE-L-p': [],
        'Validation ROUGE-L-p': [],  
        'Training ROUGE-L-r': [],
        'Validation ROUGE-L-r': [],  
    }
    
    
    val_loss = 0
    model_save_path = f"{ROOT_DIR}/output/{MODEL_NAME}best_model_{epochs}"
    
    loss_history = []
    print("Learning Rate Scheduled:", scheduler.get_last_lr())
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        total_train_bleu_score = 0
        total_train_meteor_score = 0
        total_train_codebleu_score = 0
        total_train_codebleu_score_dict = {'codebleu' : 0., 'ngram_match_score': 0.,'weighted_ngram_match_score': 0.,  'syntax_match_score': 0.0, 'dataflow_match_score': 0.0}
        total_train_rouge_scores = {'rouge-1': {'f': 0, 'p': 0, 'r': 0}, 'rouge-2': {'f': 0, 'p': 0, 'r': 0}, 'rouge-l': {'f': 0, 'p': 0, 'r': 0}}
        
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            input_ids, output_ids = batch
            input_ids = input_ids.to(device)
            output_ids = output_ids.to(device)

            outputs = model(input_ids=input_ids, labels=output_ids)
            loss = outputs.loss
            loss = loss.mean().view(1)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            
            if gpu_count > 1:
             # Generate sequences for the entire batch
                generated = model.module.generate(input_ids, max_length=200, temperature=1.0, do_sample=True)
            else:
                generated = model.generate(input_ids, max_length = 200, temperature = 1.0, do_sample = True)
            # Decode all the generated and reference sequences
            candidates = tokenizer.batch_decode(generated, skip_special_tokens=True)
            references = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # Calculate metrics for the entire batch
            bleu_scores, rouge_scores, meteors, codebleu_scores = calculate_batch_metrics(references, candidates, tokenizer)

            # Calculate average scores for the batch and accumulate
            total_train_bleu_score += np.mean(bleu_scores)
            total_train_meteor_score += np.mean(meteors)
            total_train_codebleu_score += np.mean([score['codebleu'] for score in codebleu_scores])
            
            for codebleu_score in codebleu_scores:
                    for key in codebleu_score:
                        total_train_codebleu_score_dict[key] += codebleu_score[key]
                
            
            for rouge_score in rouge_scores:
                for key in rouge_score:
                    for sub_key in rouge_score[key]:
                        total_train_rouge_scores[key][sub_key] += rouge_score[key][sub_key]

        # Normalization of the total scores
        for key in total_train_rouge_scores:
            for sub_key in total_train_rouge_scores[key]:
                total_train_rouge_scores[key][sub_key] /= len(validation_dataloader)
                
        for key in total_train_codebleu_score_dict:
            total_train_codebleu_score_dict[key] /= len(validation_dataloader)
        
        training_bleu = total_train_bleu_score / len(train_dataloader)
        training_rouge = {key: {sub_key: score / len(train_dataloader) for sub_key, score in value.items()} for key, value in total_train_rouge_scores.items()}
        training_meteor = total_train_meteor_score / len(train_dataloader)
        training_codebleu = { key : score/len(train_dataloader) for key, score in total_train_codebleu_score_dict.items()}
        print('Training BLEU:', training_bleu)
        print('Training ROUGE:', training_rouge)
        print('Training METEOR:', training_meteor)
        print('Training CodeBLEU:', training_codebleu)
        
        
        val_loss = 0
        model.eval()
        total_bleu_score = 0
        total_meteor_score = 0
        total_codebleu_score = 0
        
        total_codebleu_score_dict = {'codebleu' : 0., 'ngram_match_score': 0.,'weighted_ngram_match_score': 0.,  'syntax_match_score': 0.0, 'dataflow_match_score': 0.0}
        total_rouge_scores = {'rouge-1': {'f': 0, 'p': 0, 'r': 0}, 'rouge-2': {'f': 0, 'p': 0, 'r': 0}, 'rouge-l': {'f': 0, 'p': 0, 'r': 0}}
        with torch.no_grad():
            for batch in tqdm(validation_dataloader):
                input_ids, output_ids = batch
                input_ids = input_ids.to(device)
                output_ids = output_ids.to(device)

                # Generate sequences for the entire batch
                if gpu_count > 1:
                    generated = model.module.generate(input_ids, max_length=200, temperature=1.0, do_sample=True)
                else:
                    generated = model.generate(input_ids, max_length=200, temperature=1.0, do_sample=True)
                # Decode all the generated and reference sequences
                candidates = tokenizer.batch_decode(generated, skip_special_tokens=True)
                references = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                # Calculate metrics for the entire batch
                bleu_scores, rouge_scores, meteors, codebleu_scores = calculate_batch_metrics(references, candidates, tokenizer)

                # Calculate average scores for the batch and accumulate
                total_bleu_score += np.mean(bleu_scores)
                total_meteor_score += np.mean(meteors)
                total_codebleu_score += np.mean([score['codebleu'] for score in codebleu_scores])
                
                for codebleu_score in codebleu_scores:
                    for key in codebleu_score:
                        total_codebleu_score_dict[key] += codebleu_score[key]
                
                for rouge_score in rouge_scores:
                    for key in rouge_score:
                        for sub_key in rouge_score[key]:
                            total_rouge_scores[key][sub_key] += rouge_score[key][sub_key]

        # Normalization of the total scores
        for key in total_rouge_scores:
            for sub_key in total_rouge_scores[key]:
                total_rouge_scores[key][sub_key] /= len(validation_dataloader)
        for key in total_codebleu_score_dict:
            total_codebleu_score_dict[key] /= len(validation_dataloader)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model
            if gpu_count > 1:
                model.module.save_pretrained(model_save_path)
            else:
                model.save_pretrained(model_save_path)
            
            
        # Store loss history for plotting
        loss_history.append((total_loss / len(train_dataloader), val_loss / len(val_dataloader)))
        
        validation_bleu = total_bleu_score / len(validation_dataloader)
        validation_rouge = {key: {sub_key: score / len(validation_dataloader) for sub_key, score in value.items()} for key, value in total_rouge_scores.items()}
        validation_meteor = total_meteor_score / len(validation_dataloader)
        validation_codebleu = { key : score/len(validation_dataloader) for key, score in total_codebleu_score_dict.items()}
        
        print(f'Epoch: {epoch}, Training Loss: {total_loss / len(train_dataloader)}, Validation Loss: {val_loss / len(val_dataloader)}')
        print('Validation BLEU:', validation_bleu)
        print('Validation ROUGE:', validation_rouge)
        print('Validation METEOR:', validation_meteor)
        print('Validation CodeBLEU:', validation_codebleu)
        
        metrics_dict['Epoch'].append(epoch)
        metrics_dict['Training Loss'].append(total_loss / len(train_dataloader))
        metrics_dict['Validation Loss'].append(val_loss / len(val_dataloader))
        metrics_dict['Training BLEU'].append(training_bleu)
        metrics_dict['Validation BLEU'].append(validation_bleu)
        metrics_dict['Training METEOR'].append(training_meteor)
        metrics_dict['Validation METEOR'].append(validation_meteor)
        metrics_dict['Training CodeBLEU'].append(training_codebleu['codebleu'])
        metrics_dict['Validation CodeBLEU'].append(validation_codebleu['codebleu'])
        
        metrics_dict['Training ROUGE-1-f'].append(training_rouge['rouge-1']['f'])
        metrics_dict['Validation ROUGE-1-f'].append(validation_rouge['rouge-1']['f'])
        metrics_dict['Training ROUGE-1-p'].append(training_rouge['rouge-1']['p'])
        metrics_dict['Validation ROUGE-1-p'].append(validation_rouge['rouge-1']['p'])
        metrics_dict['Training ROUGE-1-r'].append(training_rouge['rouge-1']['r'])
        metrics_dict['Validation ROUGE-1-r'].append(validation_rouge['rouge-1']['r'])
        
        metrics_dict['Training ROUGE-2-f'].append(training_rouge['rouge-2']['f'])
        metrics_dict['Validation ROUGE-2-f'].append(total_rouge_scores['rouge-2']['f'])
        metrics_dict['Training ROUGE-2-p'].append(training_rouge['rouge-2']['p'])
        metrics_dict['Validation ROUGE-2-p'].append(total_rouge_scores['rouge-2']['p'])
        metrics_dict['Training ROUGE-2-r'].append(training_rouge['rouge-2']['r'])
        metrics_dict['Validation ROUGE-2-r'].append(total_rouge_scores['rouge-2']['r'])
        
        metrics_dict['Training ROUGE-L-f'].append(training_rouge['rouge-l']['f'])
        metrics_dict['Validation ROUGE-L-f'].append(validation_rouge['rouge-l']['f'])
        metrics_dict['Training ROUGE-L-p'].append(training_rouge['rouge-l']['p'])
        metrics_dict['Validation ROUGE-L-p'].append(validation_rouge['rouge-l']['p'])
        metrics_dict['Training ROUGE-L-r'].append(training_rouge['rouge-l']['r'])
        metrics_dict['Validation ROUGE-L-r'].append(validation_rouge['rouge-l']['r'])
        
        
        save_metrics_to_excel(metrics_dict, f"{ROOT_DIR}/output/train_val_metrics.xlsx")
    return loss_history

model_history = train_model(model, train_dataloader, val_dataloader, test_dataloader, tokenizer, EPOCHS)
print(f"Best Model: {model_save_path}")
print(pd.read_excel(f"{ROOT_DIR}/output/train_val_metrics.xlsx"))

def plot_training_val_loss(loss_history):
    epochs = range(1, len(loss_history) + 1)
    plt.plot(epochs, [loss[0] for loss in loss_history], 'g', label='Training loss')
    plt.plot(epochs, [loss[1] for loss in loss_history], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{ROOT_DIR}/output/train_val_loss.jpeg")

plot_training_val_loss(model_history)

# Stage6: Model Inference
model = T5ForConditionalGeneration.from_pretrained(model_save_path).to(device)
def generate_code(task_description):
    # Encode the task description
    inputs = tokenizer.encode_plus(
        task_description, 
        max_length=MAX_INPUT_TOKENS, 
        padding='max_length', 
        truncation=True, 
    )
    input_ids = torch.tensor(inputs.input_ids)
    # Move the inputs to the GPU
    input_ids = input_ids.to(device)
    # Add a batch dimension to the tensor
    input_ids = input_ids.unsqueeze(0)

    # Generate the code
    with torch.no_grad():
        # performing beam search with 5
        outputs = model.generate(input_ids, do_sample = True, max_length = MAX_OUTPUT_TOKENS, top_p = 0.95, top_k = 1, repetition_penalty=2., num_return_sequences = 1)

    # Decode the generated IDs to get the generated text
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_code

# Test the function with a task description
task_description = "Write a python function to remove first and last occurrence of a given character from the string."
print(generate_code(task_description))