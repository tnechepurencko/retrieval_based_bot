import nltk
from nltk.stem import WordNetLemmatizer
import json
import tensorflow
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import pandas as pd
import string
import re
import pymorphy2


def remove_one_hello_word(word, phrase):
    if word in phrase.lower():
        # print(answer)
        separators = [m.start() for m in re.finditer('!', phrase)]
        separators.extend([m.start() for m in re.finditer('\.', phrase)])
        separators.append(-1)
        separators.sort()
        if separators[-1] != len(phrase) - 1:
            separators.append(len(phrase) - 1)

        sentences = []
        for i in range(len(separators) - 1):
            sentences.append(phrase[(separators[i] + 1):(separators[i + 1] + 1)])

        new_answer = ''
        for x in sentences:
            if not (word in x.lower() and len(x.split(' ')) < 3):
                new_answer += x
        if new_answer[0] == ' ':
            new_answer = new_answer[1:]
        return new_answer
    else:
        return phrase


def remove_two_hello_words(words, phrase):
    if words in phrase.lower():
        # print(answer)
        separators = [m.start() for m in re.finditer('!', phrase)]
        separators.extend([m.start() for m in re.finditer('\.', phrase)])
        separators.append(-1)
        separators.sort()
        if separators[-1] != len(phrase) - 1:
            separators.append(len(phrase) - 1)

        sentences = []
        for i in range(len(separators) - 1):
            sentences.append(phrase[(separators[i] + 1):(separators[i + 1] + 1)])

        new_answer = ''
        for x in sentences:
            if not (words.split(' ')[0] in x.lower() and words.split(' ')[1] in x.lower() and len(x.split(' ')) < 4):
                new_answer += x
        if new_answer[0] == ' ':
            new_answer = new_answer[1:]
        return new_answer
    else:
        return phrase


def answer_preprocessing(phrase):
    answer = ''
    insert = True
    for s in phrase:
        if insert is True:
            if s != '<':
                answer += s
            else:
                insert = False
        elif s == '>':
            insert = True

    answer = answer.replace('\t', ' ')
    answer = answer.replace('\n', ' ')
    answer = answer.replace('\r', ' ')
    answer = remove_one_hello_word('здравствуйте', answer)
    answer = remove_two_hello_words('добрый день', answer)
    answer = remove_two_hello_words('добрый вечер', answer)
    answer = remove_two_hello_words('доброе утро', answer)
    return answer


# train data preprocessing
df = pd.read_csv('support_messages.csv')

df = df.drop(df[df['text'].str.contains('Мы получили Ваше сообщение!')].index)
df = df.drop(df[df['text'].str.contains('Оцените работу поддержки.')].index)
df = df.drop(df[df['text'].str.contains('Мы всё еще можем Вам помочь?')].index)

helper = df[df['text'].str.split().str.len().lt(3)]
helper = helper[helper['text'].str.contains('@')]
helper = helper[helper['messageFrom'].str.contains('CLIENT')]
df = df.drop(df[df['text'].str.split().str.len().lt(3)].index)

df = pd.concat([df, helper], ignore_index=True)

data = {'question': [], 'answer': []}
new_df = pd.DataFrame(data)

# creating formatted db
morph = pymorphy2.MorphAnalyzer()
stop_phrases = ['по какой причине', 'добрый день', 'добрый вечер', 'доброе утро', 'к сожалению', 'в очередной раз',
                'у меня', 'по факту']
stop_words = {'здравствуйте', 'почему', 'уже', 'я', 'есть', 'тут', 'а', 'и', 'нигде', 'снова', 'как', 'мой', 'но',
              'этой', 'в', 'хорошо', 'понял', 'поняла', 'пожалуйста', 'от', 'с', 'подскажите', 'так', 'спасибо'}

for x in df['usedeskChatId'].unique():
    df_usr = df.loc[df['usedeskChatId'] == x]
    query = []
    for idx, row in df_usr.sort_values('createdDate').iterrows():
        if len(query) > 0 and row['messageFrom'] == 'OPERATOR':
            for i in range(len(query)):
                query[i] = morph.parse(query[i])[0].normal_form

            question = ' '.join(list(set(query) - stop_words))
            answer = answer_preprocessing(row['text'])

            new_df.loc[len(new_df)] = [question, answer]
            query = []
        elif row['messageFrom'] == 'OPERATOR':
            continue
        else:
            msg = row['text'].translate(str.maketrans('', '', string.punctuation)).lower()
            for w in stop_phrases:
                if w in msg:
                    msg = msg.replace(w, '')
            query.extend(msg.split())

new_df.to_csv('formatted_msgs.csv', sep='\t', encoding='utf-8', index=False)
