import os
from os import getenv
import openai
import pandas as pd
import time


def prepare_jsonl(filename, frac):
    df = pd.read_csv('data\\' + filename + '.csv', sep='\t').drop('original', axis=1)
    df = df.sample(frac=frac)
    print(filename + ' shape:', df.shape)

    df1 = df.rename(columns={'question': 'prompt', 'answer': 'completion'})
    result1 = df1.to_json(orient="records", force_ascii=False)[1:-1].replace('},{', '}\n{')
    with open('fine_tuning\\' + filename + '.jsonl', 'w', encoding='utf-8') as f:
        f.write(result1)


def upload_response_id(file):
    upload_response = openai.File.create(
        file=open(file, "rb"),
        purpose='fine-tune'
    )
    return upload_response.id


data_path = '\\'.join(os.path.abspath(__file__).split('\\')[:-2] + ['data\\'])

api_key = getenv("OPENAI_API_KEY")
openai.api_key = api_key

training_file = "train_msgs.jsonl"
validation_file = "test_msgs.jsonl"

prepare_jsonl(training_file[:-6], 0.01)
prepare_jsonl(validation_file[:-6], 0.01)

tr_file_id = upload_response_id('fine_tuning\\' + training_file)
val_file_id = upload_response_id('fine_tuning\\' + validation_file)

model_engine = "davinci"
n_epochs = 3
batch_size = 4
learning_rate = 1e-5
max_tokens = 1024

# Create the fine-tuning job
fine_tuning_job = openai.FineTune.create(
    model=model_engine,
    n_epochs=n_epochs,
    batch_size=batch_size,
    learning_rate_multiplier=learning_rate,
    # max_tokens=max_tokens,
    training_file=tr_file_id,
    validation_file=val_file_id,
)

job_id = fine_tuning_job["id"]
print(f"Fine-tuning job created with ID: {job_id}")

while True:
    fine_tuning_status = openai.FineTune.retrieve(job_id)
    status = fine_tuning_status["status"]
    # print(f"Fine-tuning job status: {status}")

    if status in ["succeeded", "failed"]:
        print(status)
        print(fine_tuning_status)
        break

fine_tuned_model_id = fine_tuning_status["fine_tuned_model"]


# Use the fine-tuned model for text generation
def generate_text(prompt, model_id, max_tokens=50):
    response = openai.Completion.create(
        engine=model_id,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()


prompt = "как вывести деньга карта"
generated_text = generate_text(prompt, fine_tuned_model_id)
print(f"Generated text: {generated_text}")
