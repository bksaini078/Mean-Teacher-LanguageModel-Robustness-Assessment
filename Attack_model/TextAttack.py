import numpy as np
import torch
import textattack
# model wrapper for attack
from textattack.models.wrappers import ModelWrapper
#attack recipe importing
from textattack.attack_recipes.bert_attack_li_2020 import BERTAttackLi2020
from textattack.attack_recipes import PWWSRen2019
from textattack.attack_recipes.bae_garg_2019 import BAEGarg2019
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack.attack_recipes.textbugger_li_2018 import TextBuggerLi2018

from textattack import Attacker
from transformers import AutoTokenizer
import datetime  
import tensorflow as tf



def attack_model(model,dataset_for_attack,pretrained_weights,max_len,model_name,attack_recipe='TextFoolerJin2019',query_budget=200, num_examples=100):
  ''' 
  Functions to attack the model
  Inputs:
  model : Model to attack
  dataset_for_attack : test dataset for attacking the model
  pretrained_weights: For tokenization of the dataset
  max_len : for tokenizer
  attack_recipe : type of attack recipe, allowed right now [PWWSRen2019, BERTAttackLi2020,BAEGarg2019,TextFoolerJin2019]
  model_name- Name of the model, 
  query budget : Max allowed query. Default= 200
  num_examples: Max number of examples . Default= 100

  OUTPUT:
  Will save file in Result folder. 
  '''
  class CustomTensorFlowModelWrapper(ModelWrapper):
    '''
    This wrapper is required to attack the model using text attack.
    '''
    def __init__(self, model,pretrained_weights):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

    def __call__(self, text_input_list):
        tokens=self.tokenizer(text_input_list,return_tensors='tf', 
                         truncation=True,
                         padding='max_length',
                         max_length=max_len, 
                         add_special_tokens=True)
        
        input_ids= []
        attention_mask=[]
        input_ids.append(tokens.input_ids)
        attention_mask.append(tokens.attention_mask)
        preds = torch.tensor(self.model([input_ids, attention_mask]).numpy())
  
        return preds

  now =datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
  model_wrapper = CustomTensorFlowModelWrapper(model,pretrained_weights)

  if attack_recipe =='PWWSRen2019':
    attack= PWWSRen2019.build(model_wrapper)
  elif attack_recipe=='BERTAttackLi2020':
    attack= BERTAttackLi2020.build(model_wrapper)
  elif attack_recipe=='BAEGarg2019':
    attack= BAEGarg2019.build(model_wrapper)
  elif attack_recipe=='TextFoolerJin2019':
    attack= TextFoolerJin2019.build(model_wrapper)
  elif attack_recipe=='TextBuggerLi2018':
    attack=TextBuggerLi2018.build(model_wrapper)
  else: 
    print('Either model name is incorrect or wrong model is called .\n Allowed model are [PWWSRen2019, BERTAttackLi2020,BAEGarg2019,TextFoolerJin2019]')
  attack_args = textattack.AttackArgs (
    num_examples=num_examples,
    log_to_csv=f'Result/{attack_recipe}/{model_name}_nexp{num_examples}_qb{query_budget}_{now}.csv',
    disable_stdout=False,
    query_budget=query_budget
    )
  with tf.device('/GPU:0'):
    attacker = Attacker(attack, dataset_for_attack,attack_args)
    attacker.attack_dataset()
  
  return