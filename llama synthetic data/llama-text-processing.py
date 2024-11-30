import csv
import pandas as pd

file = open('ollama_responses.txt', 'r')

# get 400 random images to pair wth the new text
# filter abortion training for only support stance
df = pd.read_csv('../data/abortion_train.csv')
filtered_df = df[df['stance'] == 'support']

# select random lines with replacement
sampled_df = filtered_df.sample(n=400, replace=True)
tweet_ids = sampled_df['tweet_id']
tweet_urls = sampled_df['tweet_url']
persuasiveness = sampled_df['persuasiveness']

with open('../data/abortion_train.csv', 'a', newline='') as f:
    writer = csv.writer(f)

    for i, line in enumerate(file):
        stripped_line = line.replace('"', '').strip()
        new_data = [tweet_ids.iloc[0], tweet_urls.iloc[0], stripped_line, 'support', persuasiveness.iloc[i], 'train']

        writer.writerow(new_data)

file.close()