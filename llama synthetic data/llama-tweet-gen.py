import ollama

OUTPUT_FILE = 'ollama_responses_abortion.txt'
MODEL = 'llama3.2' 
PROMPT = 'Write text for a tweet supporting pro-choice rights with no leading text.'

# generate 400 responses and write them to a text file.
with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
    for i in range(400):
        print(f'Fetching response {i+1}/400...')
        response = ollama.generate(MODEL, PROMPT)
        file.write(response['response'] + '\n')