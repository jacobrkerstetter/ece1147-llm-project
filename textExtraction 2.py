import clip
import os
import torch
from PIL import Image
import pandas as pd

#Load in device
device = "cpu"

#Load in OpenAI CLIP 
model, preprocess = clip.load("ViT-B/32", device=device)

#Load in data
data_path = "data/abortion_train.csv"
data = pd.read_csv(data_path)

#Take in the extracted text from csv files
output = "text_features"
os.makedirs(output, exist_ok=True)

#Process the extracted text features.
for index, row in data.iterrows():
    tweet_text = row['tweet_text'] 

    #Tokenize and encode the extracted texts
    text_tokens = clip.tokenize([tweet_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    #Save the feature for each tweets
    feature_path = os.path.join(output, f"tweet_{index}_text_features.pt")
    torch.save(text_features.cpu(), feature_path)
    #uncomment  when training
    #print(f"Saving feature {index} to path: {feature_path}")