import pandas as pd 
import tensorflow as tf 
import numpy as np 
from Utils.utils import label_to_int

def data_loader(Data_loc,target_col, aug_col):
  '''
  Load the dataset and return train,test and unlabel dataframe
  '''
  df_train_loc=Data_loc+'PreprocessedData/pr_train.csv'
  df_test_loc= Data_loc+'PreprocessedData/pr_test.csv' 


  df_train= pd.read_csv(df_train_loc,index_col=[0])
  df_train= df_train.dropna()
  df_train= df_train.drop_duplicates().reset_index(drop=True)
  df_train[target_col]=df_train[target_col].map(lambda row: label_to_int(row))

  # Test data 
  df_test= pd.read_csv(df_test_loc,index_col=[0])
  df_test= df_test.dropna()
  df_test= df_test.drop_duplicates().reset_index(drop=True)
  df_test[target_col]=df_test[target_col].map(lambda row: label_to_int(row))

  # loading unlabel data 
  # Adversarial unlabeled data location 
  df_aug_syn_un_loc=Data_loc+'AugmentedData/aug_synonym.csv'
  df_aug_con_un_loc=Data_loc+'AugmentedData/aug_context.csv'
  df_aug_bt_un_loc=Data_loc+'AugmentedData/aug_synonym.csv'

  # Reading adversarial Unlabel data 
  df_aug_syn_un= pd.read_csv(df_aug_syn_un_loc)
  df_aug_syn_un= df_aug_syn_un.dropna().drop_duplicates().reset_index(drop=True)


  df_aug_con_un= pd.read_csv(df_aug_con_un_loc)
  df_aug_con_un= df_aug_con_un.dropna().drop_duplicates().reset_index(drop=True)

  df_aug_bt_un= pd.read_csv(df_aug_bt_un_loc)
  df_aug_bt_un= df_aug_bt_un.dropna().drop_duplicates().reset_index(drop=True)

  # Combining all together 
  df_aug_unlabel = df_aug_syn_un.append(df_aug_con_un).append(df_aug_bt_un)
  df_aug_unlabel= df_aug_unlabel.sample(frac=1).reset_index(drop=True)
  df_aug_unlabel[target_col]= df_aug_unlabel[aug_col].map(lambda row: label_to_int(row))



  return df_train, df_test, df_aug_unlabel
