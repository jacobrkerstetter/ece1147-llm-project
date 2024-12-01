import csv
import pandas as pd

# Open the file containing responses
file = open('flan-T5 synthetic data/flan_oppose_responses.txt', 'r')

# Get 400 random images to pair with the new text
# Filter the abortion training data for only 'support' stance
df = pd.read_csv(r'C:\Users\anhqu\OneDrive\Documents\Project_ECE47\ece1147-llm-project\data\abortion_train.csv')
filtered_df = df[df['stance'] == 'oppose']

# Select random lines with replacement
sampled_df = filtered_df.sample(n=400, replace=True)
tweet_ids = sampled_df['tweet_id']
tweet_urls = sampled_df['tweet_url']
persuasiveness = sampled_df['persuasiveness']

# Write to a new file
output_file = r'C:\Users\anhqu\OneDrive\Documents\Project_ECE47\ece1147-llm-project\data\abortion_train_flan.csv'
with open(output_file, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    for i, line in enumerate(file):
        stripped_line = line.replace('"', '').strip()
        new_data = [
            tweet_ids.iloc[i % len(tweet_ids)],  # Cycle through tweet IDs
            tweet_urls.iloc[i % len(tweet_urls)],  # Cycle through tweet URLs
            stripped_line,  # Response text
            'oppose',  # Stance
            persuasiveness.iloc[i % len(persuasiveness)],  # Cycle persuasiveness
            'train'  # Data split
        ]

        writer.writerow(new_data)

file.close()

print(f"New data has been appended to {output_file}")
