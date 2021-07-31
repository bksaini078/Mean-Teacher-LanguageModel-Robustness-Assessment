
from transformers import AutoTokenizer
import pandas as pd 
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

'''Functions :
- Data Splitter 
- Creating Tokenizer 
- Convert to category 
- Data Tokenization 
- Evaluation 
- Augmentation Evaluation
'''
def label_to_int(row):
    if row=='real':
        return 0
    else:
        return 1

def dataset_split(dataset,test_size):
  '''Split the dataframe into train and test '''
  df_train=dataset.sample(frac=(1-test_size),random_state=200) #random state is a seed value
  df_test=dataset.drop(df_train.index)
  df_train= df_train.reset_index(drop=True)
  df_test=df_test.reset_index(drop=True)
  return df_train,df_test

# Converting to categories 
def convert_to_category(target, n_classes=2):
  ''' 
  Convert to categorical 
  '''
  return to_categorical(target,n_classes)

#Creating tokenizer
def create_tokenizer(pretrained_weights='distilbert-base-uncased'):
  '''Function to create the tokenizer'''

  tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
  return tokenizer

#Tokenization of the data
def data_tokenization(dataset,feature_col,target_col,max_len,tokenizer):
    '''dataset: Pandas dataframe with feature name is column name 
    Pretrained_weights: selected model 
    RETURN: [input_ids, attention_mask]'''

    tokens = dataset[feature_col].apply(lambda x: tokenizer(x,return_tensors='tf', 
                                                            truncation=True,
                                                            padding='max_length',
                                                            max_length=max_len, 
                                                            add_special_tokens=True))
    input_ids= []
    attention_mask=[]
    for item in tokens:
        input_ids.append(item['input_ids'])
        attention_mask.append(item['attention_mask'])
    input_ids, attention_mask=np.squeeze(input_ids), np.squeeze(attention_mask)

    # if we have label column
    if (target_col in dataset.columns):
        y= convert_to_category(dataset[target_col],2)
        return [input_ids,attention_mask], y, tokenizer.vocab_size
    else:
        return [input_ids,attention_mask]

def data_slices(x_train,y_train,x_unlabel,batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices( (x_train[0],x_train[1], y_train) )
    train_dataset = train_dataset.shuffle( buffer_size=1024 ).batch(batch_size)

    unlabel_dataset = tf.data.Dataset.from_tensor_slices( (x_unlabel[0],x_unlabel[1]) )
    unlabel_dataset = unlabel_dataset.shuffle( buffer_size=1024 ).batch(batch_size)
    return train_dataset, unlabel_dataset