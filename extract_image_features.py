import argparse
from transformers import CLIPProcessor, CLIPModel
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

import multimodal_model 
    
def load_data(dataset_path, task, bad_tweets_path):

    dataframe = pd.read_csv(dataset_path)
    bad_tweets = pd.read_csv(bad_tweets_path)
    bad_tweets = bad_tweets["tweet_id"]

    dataframe = dataframe.merge(bad_tweets, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    if task == 'AS':
        y_true = np.array(dataframe.stance)
    else:
        y_true = np.array(dataframe.persuasiveness)
    
    return dataframe, y_true

def handle_IP_approach_1(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images, bad_tweets_path):

    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'IP', bad_tweets_path)
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'IP', bad_tweets_path)
    df_train = pd.concat([A_df_train, GC_df_train], axis=0)
    y_train = np.concatenate((A_y_train, GC_y_train))

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'IP', bad_tweets_path)
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'IP', bad_tweets_path)
    df_dev = pd.concat([A_df_dev, GC_df_dev], axis=0)
    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'IP', bad_tweets_path)
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'IP', bad_tweets_path)
    df_test = pd.concat([A_df_test, GC_df_test], axis=0)
    y_test_true = np.concatenate((A_y_test, GC_y_test))

    X_train = multimodal_model.clip32(df_train, train_dev_images)
    X_dev = multimodal_model.clip32(df_dev, train_dev_images)
    X_test = multimodal_model.clip32(df_test, test_images)

    classifier = SVC(kernel='poly', degree=3, C=1.0, coef0=0.02,  shrinking=False, probability=True)
    classifier.fit(X_train, y_train)
 
    y_dev_pred = classifier.predict(X_dev)
    y_test_pred = classifier.predict(X_test)
    
    pos_label = 'yes'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label), 4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label), 4)
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)

if __name__ == "__main__":        
    GC_train_dataset_path = "data/gun_control_train.csv"
    A_train_dataset_path = "data/abortion_train.csv"
    GC_dev_dataset_path = "data/gun_control_dev.csv"
    A_dev_dataset_path = "data/abortion_dev.csv"
    GC_test_dataset_path = "test/data/gun_control_test.csv"
    A_test_dataset_path = "test/data/abortion_test.csv"

    bad_tweets_path = 'data/bad_tweets.csv'

    train_images = "data/images/image/"
    test_images = 'test/data/images/image/'
    handle_IP_approach_1(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_images, test_images, bad_tweets_path)