from transformers import pipeline

#load FLAN-T5 model
generator = pipeline("text2text-generation", model="google/flan-t5-large")

#define the prompt for tweet generation
prompt = "Write a short message opposing abortion freedom."

tweets = generator(
    prompt,
    max_length=54,
    num_return_sequences=200,
    temperature=0.6,
    top_p=0.9,
    do_sample=True
)

#print out
for i, tweet in enumerate(tweets):
    print(f"{tweet['generated_text']}")
