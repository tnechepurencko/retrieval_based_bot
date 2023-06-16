import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from keras.models import load_model
import numpy as np
import json
import random
import pandas as pd
import pymorphy2


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# test data preprocessing
# dfd = pd.read_csv('data/dev/Bitext_Sample_Customer_Service_Validation_Dataset.csv')
# test_data = dfd.drop(columns=['entity_type', 'entity_value', 'start_offset', 'end_offset', 'category', 'tags'])
# test_data = test_data.rename({'utterance': 'patterns', 'intent': 'tag'}, axis=1)

test_data = pd.read_csv('data/test_msgs.csv', sep='\t')

lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

msp = 0

for idx, row in test_data.iterrows():
    ints = predict_class(row['question'], model)
    if len(ints) == 0:
        msp += 1
        print('\tcannot predict')
        print('\tquerry:', row['question'])
        continue

    tag = ints[0]['intent']
    # print(ints)
    if tag != row['answer']:
        msp += 1
        print('\tmisprediction:\n\t', tag, '\n\tinstead of\n\t', row['answer'])
        print('\tquerry:', row['question'])
        print('\t', ints, '\n')

print("{:10.2f}% was mispredicted".format(msp / len(test_data) * 100))
