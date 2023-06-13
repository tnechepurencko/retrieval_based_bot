import numpy as np
import pandas as pd
# import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import sklearn.metrics.pairwise
import json
from sklearn.metrics.pairwise import cosine_similarity


class Model:
    def __init__(self) -> None:
        module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
        self.model = hub.load(module_url)
        self.embeds = self.load_embeds()
        self.data = pd.read_csv('train_msgs.csv', sep='\t')

    def load_embeds(self):
        with open('Questions_embeddings_USE.json', 'r') as f:
            embeds = json.load(f)
        return embeds

    def embed_text(self, text):
        """
            returning the embeddings of an input text
        """
        return self.model(text)

    def get_top_ans(self, question, n_answer=3):
        """
            This function will try to get the 3 most relevant answers by comparing the question with similar questions.
            The similarity is calculated using cosine_similarity
        """
        question_embedding = self.embed_text(
            question).numpy()  # getting the embeddings of the question
        question_embedding = question_embedding.reshape(1, -1)
        similarity_scores = []
        for _, embed in list(self.embeds.items()):
            embed = np.array(embed).reshape(1, -1)
            similarity_score = cosine_similarity(
                question_embedding, embed)[0][0]
            similarity_scores.append(similarity_score)
        top_indices = np.argsort(similarity_scores)[-3:][::-1]
        top_questions = [list(self.embeds.keys())[index]
                         for index in top_indices]
        top_answers = [self.data.answer.values[index] for index in top_indices]

        return top_questions[:n_answer], top_answers[:n_answer]


# For testing
model = Model()
test_data = pd.read_csv('test_msgs.csv', sep='\t')
for idx, row in test_data.iterrows():
    question = row['question']
    try:
        q, a = model.get_top_ans(question)
        print(row['original'])
        print(q)
        print(a, '\n')
    except Exception as e:
        print(row['original'])
        print('ERROR')
        print(e, '\n')



# question = ['пополнение мир регистрироваться чек день пополнить почти время долгий получаться карта каждый происходить сбербанк этот не']
# q, a = model.get_top_ans(question)
# print(q)
# print(a, '\n')

