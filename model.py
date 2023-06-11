import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
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
        self.data = pd.read_csv('formatted_msgs.csv', sep='\t')

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
        '''
            This function will try to get the 3 most relevent answers by comparing the question with similar questions.
            The similarity is calculated using cosine_similarity
        '''
        question_embeding = self.embed_text(
            question).numpy()  # getting the embedings of the question
        question_embeding = question_embeding.reshape(1, -1)
        similarity_scores = []
        for _, embed in list(self.embeds.items()):
            embed = np.array(embed).reshape(1, -1)
            similarity_score = cosine_similarity(
                question_embeding, embed)[0][0]
            similarity_scores.append(similarity_score)
        top_indices = np.argsort(similarity_scores)[-3:][::-1]
        top_questions = [list(self.embeds.keys())[index]
                         for index in top_indices]
        top_answers = [self.data.answer.values[index] for index in top_indices]

        return top_questions[:n_answer], top_answers[:n_answer]

# For testing
# model = Model()
# question = ['пополнение мир регистрироваться чек день пополнить почти время долгий получаться карта каждый происходить сбербанк этот не']
# model.get_top_3_ans(question)
