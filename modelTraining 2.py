import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

#Exract output where text features were stored
feature_dir = "text_features" 