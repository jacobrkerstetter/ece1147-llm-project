import csv

file = open('ollama_responses.txt', 'r')


with open('../data/abortion_train.csv', 'a', newline='') as f:
    writer = csv.writer(f)

    for line in file:
        stripped_line = line.replace('"', '').strip()
        print(stripped_line)
        new_data = ['n/a', 'n/a', stripped_line, 'support', 'no', 'train']

        writer.writerow(new_data)

file.close()