import os
import sys
from typing import List

import fire
import torch

import transformers
from datasets import load_dataset
import evaluate

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer,GPT2Config 

from utils.prompter import Prompter

from datasets import load_dataset,load_from_disk
import nltk
nltk.download("punkt")

from nltk.tokenize import sent_tokenize

import numpy as np
import json
import random

current_run = 0

rouge_score = evaluate.load("evaluate/metrics/rouge/rouge.py")
class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trained_data = None

    
    def training_step(self, model, inputs):
        ret = super().training_step(model, inputs)
        self.trained_data = []
        # print(inputs.keys(), len(inputs["input_ids"]),type(inputs["input_ids"]), type(inputs["input_ids"][0]), type(inputs["input_ids"][0][0]))
        # self.trained_data = inputs["id"]
        step_num = len(inputs["input_ids"])
        # print("inputs:", type(inputs), inputs["input_ids"].shape, inputs.keys())
        for i in range(len(inputs["input_ids"])):
            mstr = inputs["input_ids"][i].tolist()
            result = ''.join(map(str, mstr))
            self.trained_data.append(result)
            
        return ret
     
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ):
        
        
            
        ret_loss, ret_logits, ret_labels =  super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        # device = "cuda:0"
        # model = model.to(device)
        # tmp = inputs["input_ids"].to(device)
        # inputs["input_ids"] = inputs["input_ids"].to(model.device)
        inputs = self._prepare_inputs(inputs)
        
        # print("devices:",inputs["input_ids"].device, model.device)
        
        outputs = model.generate(**inputs, 
                                 max_new_tokens=100,
                                 do_sample=True,  
                                 top_k=50)
        
        print("outputs:", type(outputs), outputs.shape)
        predictions, labels = outputs, inputs["labels"][:, -100:]
        print("predictions", len(predictions),type(predictions),predictions.shape)
        print("labels", len(labels),type(labels),labels.shape)
        # Decode generated summaries into text
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True,)
        # print(decoded_preds)
        # print("decoded_preds",len(decoded_preds),len(decoded_preds[0]))
        # Replace -100 in the labels as we can't decode them
        # print("labels", type(labels))
        
        labels = np.where(labels.cpu() != -100, labels.cpu(), self.tokenizer.pad_token_id)           
        # Decode reference summaries into text
        # print("labels", type(labels),labels.shape)
        # print(labels)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # print("decoded_labels", len(decoded_labels),len(decoded_labels[0]))
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        print("result",result)
        # return {k: round(v, 4) for k, v in result.items()}
        
            
        eval_loss = float(ret_loss.item())
        with open(f"training_run{current_run}.json",'a') as f:
            mdict = {"train_data_step": self.trained_data, "eval_loss": eval_loss,"rouge": result, "step":self.state.global_step, "epoch": self.state.epoch}
            f.write(json.dumps(mdict) + '\n')
        
        # print("mend")
            
        
        return ret_loss, ret_logits, ret_labels 
        
       
       
    # def prediction_step(
    #     self,
    #     model,
    #     inputs,
    #     prediction_loss_only,
    #     ignore_keys,
    # ):
    #     print("inputs:", type(inputs), inputs["input_ids"].shape)
        
        
    #     if not self.args.predict_with_generate or prediction_loss_only:
            
    #         loss, logits, labels =  super().prediction_step(
    #             model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
    #         )
    #         eval_loss = float(loss.item())
    #         with open(f"training_run{current_run}.json",'a') as f:
    #             mdict = {"train_data_step": self.trained_data, "eval_loss": eval_loss, "step":self.state.global_step, "epoch": self.state.epoch}
    #             f.write(json.dumps(mdict) + '\n')
                
    #         return loss, logits, labels 
            
    #     else:
    #         loss, generated_tokens, labels = super().prediction_step(
    #             model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
    #         )
    #         print("generated_tokens:", type(generated_tokens), generated_tokens.shape)
    #         eval_loss = float(loss.item())
    #         with open(f"training_run{current_run}.json",'a') as f:
    #             mdict = {"train_data_step": self.trained_data, "eval_loss": eval_loss, "step":self.state.global_step, "epoch": self.state.epoch}
    #             f.write(json.dumps(mdict) + '\n')
            
    #         return loss, generated_tokens, labels
            
 

 
def train(
    # model/data params
    base_model: str = "/ssd3/xiaojingwu/gpt2",  # the only required argument
    data_path: str = "",
    output_dir: str = "./gpt2",
    # training hyperparams
    # batch_size: int = 128,
    batch_size: int = 4,
    micro_batch_size: int = 4,
    num_epochs: int = 4,
    learning_rate: float = 3e-4,
    cutoff_len: int = 900,
    max_target_length = 30,
    # val_set_size: int = 2000,
    val_set_size: int = 100,
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    print("gradient_accumulation_steps0: ", gradient_accumulation_steps)
    
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

        print("gradient_accumulation_steps1: ", gradient_accumulation_steps, "world_size:", world_size)
    
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model


    model = GPT2LMHeadModel.from_pretrained('/ssd3/xiaojingwu/gpt2')

    # tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer = GPT2Tokenizer.from_pretrained('/ssd3/xiaojingwu/gpt2')
    # tokenizer = LlamaTokenizer.from_pretrained(base_model)

    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    tokenizer.padding_side = "left"  # Allow batched inference


    new_pad_token = '<pad>'
    assert new_pad_token not in tokenizer.get_vocab()

    # 为分词器添加新的 pad_token，并设置 pad_token_id
    tokenizer.add_special_tokens({'pad_token': new_pad_token})
    pad_token_id = tokenizer.convert_tokens_to_ids(new_pad_token)
    tokenizer.pad_token_id = pad_token_id
    # 更新模型的词汇表大小
    model.resize_token_embeddings(len(tokenizer))
    # 确保模型知道新的 pad_token_id
    model.config.pad_token_id = pad_token_id

    def tokenize(prompt, add_eos_token=False, pad_str = "max_length"):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding = pad_str,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def tokenize_gpt2(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        sepStr = " TL;DR "
        text = prompt["article"] + sepStr + prompt["highlights"]
        # result = tokenizer(
        #     text,
        #     truncation=True,
        #     max_length=cutoff_len,
        #     padding=False,
        #     return_tensors=None,
        # )
        result = tokenize(text, add_eos_token=add_eos_token)
        result["id"] = prompt["id"]
        if not train_on_inputs:
            user_prompt = prompt["highlights"]
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token, pad_str=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            user_prompt_len = len(result["input_ids"]) - user_prompt_len
            # if add_eos_token:
            #     user_prompt_len -= 1
            result["labels"] = [-100] * user_prompt_len + tokenized_user_prompt["input_ids"]  # could be sped up, probably
            
        
        return result

    
    def tokenize_gpt3(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        sepStr = " TL;DR "
        text = prompt["article"] + sepStr
    
        result = tokenizer(
            text,
            max_length=cutoff_len,
            truncation=True,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        labels = tokenizer(
            prompt["highlights"], max_length=max_target_length, truncation=True
        )
        
        result["labels"] = labels["input_ids"]  
        
        return result
    

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        print("predictions", len(predictions),type(predictions),predictions.shape)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        print("decoded_preds",len(decoded_preds),len(decoded_preds[0]))
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    def compute_metrics2(eval_pred):
 
        predictions, labels = eval_pred
        print("predictions", len(predictions),type(predictions),predictions.shape)
        
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        print("decoded_preds",len(decoded_preds),len(decoded_preds[0]))
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)           
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}



    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt


    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=data_path)
    # else:
    #     data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            # set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    ##------原来数据处理
    # random.seed(42)
    total_data = load_from_disk('/ssd3/xiaojingwu/datasets/cnn_dailymail')
    # print("data_begin_test:")
    # print(data)
    # print("data_end:")
    # num_samples = 100
    # num_rows = len(data["train"])
    # random_indices = random.sample(range(num_rows), num_samples)
    # train_data = data["train"].select(random_indices)
    # save_path = "/ssd3/xiaojingwu/alpaca-lora/train_dataset"
    # train_data.save_to_disk(save_path)
    # print("len(train_data)", train_data)
    
    
    data = load_from_disk("/ssd3/xiaojingwu/alpaca-lora/train_dataset")
    print(len(data))
    num_samples = 64
    num_rows = len(data)
    random_indices = random.sample(range(num_rows), num_samples)
    train_data = data.select(random_indices)
    val_data = total_data["validation"].select([100])
    print(train_data)
    print(val_data)
    train_data = train_data.shuffle().map(tokenize_gpt2)
    
    val_data = val_data.map(tokenize_gpt2)
    
      
    print("train_data:\n",train_data)
    
    print("val_data:\n",val_data)
    print("gradient_accumulation_steps: ", gradient_accumulation_steps)
    print("micro_batch_size:", micro_batch_size)
    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     print("model.is_parallelizable = True")
    #     print("model.model_parallel = True")
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.Seq2SeqTrainingArguments(
            # no_cuda=True,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=50,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            # fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            # evaluation_strategy = "no",
            save_strategy="steps",
            eval_steps=1 if val_set_size > 0 else None,
            save_steps=40,
            output_dir=output_dir,
            # save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            # predict_with_generate = True,
            logging_dir='./logs',
            # prediction_loss_only = False,
            
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, return_tensors="pt", padding="max_length", max_length=cutoff_len,
        ),
        # compute_metrics=compute_metrics2,
        #-------
        tokenizer = tokenizer
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    
    
    trainer.train(resume_from_checkpoint=False)
        
    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )



if __name__ == "__main__":
    # torch.cuda.empty_cache()
    
    training_runs = 3
    for train_run in range(training_runs): 
        torch.cuda.empty_cache()
        current_run = train_run
        # fire.Fire(train)
        train()
        print(train_run)
    # train()
