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
from sklearn.model_selection import train_test_split
import string

nltk.download('punkt')
nltk.download('wordnet')

# train data preprocessing
# dft = pd.read_csv('data/train/Bitext_Sample_Customer_Service_Training_Dataset.csv')
# train_data = dft.drop(columns=['entity_type', 'entity_value', 'start_offset', 'end_offset', 'category', 'tags'])
# train_data = train_data.rename({'utterance': 'patterns', 'intent': 'tag'}, axis=1)
#
# tags = train_data.tag.unique()
# answers = {}
# for idx, val in enumerate(tags):
#     answers[val] = idx

# train data preprocessing
dft = pd.read_csv('formatted_msgs.csv', sep='\t')

tags = dft.answer.unique()
answers = {}
for idx, val in enumerate(tags):
    # answers[val] = idx
    b = val.encode('utf-8')
    answers[val] = int.from_bytes(b, 'little')
print('answers:', len(answers))

df_elements = dft.sample(n=1000)
train_data, test_data = train_test_split(df_elements, test_size=0.2)
test_data.to_csv('test_msgs.csv', sep='\t', encoding='utf-8', index=False)
# print(train_data.iloc[1]['question'])
# print(train_data.iloc[1]['answer'])
train_data = train_data.replace({"answer": answers})
test_data = test_data.replace({"answer": answers})

# preprocessing queries
lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']

for key in answers.keys():
    df = train_data.loc[train_data['answer'] == key]
    phrases = df['question'].tolist()

    for phrase in phrases:
        w = nltk.word_tokenize(phrase)
        words.extend(w)

        documents.append((w, key))
        if key not in classes:
            classes.append(key)

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# print(len(documents), "documents")
# print(len(classes), "classes", classes)
# print(len(words), "unique lemmatized words", words)
print(len(documents), "documents")
print(len(classes), "classes")
print(len(words), "unique lemmatized words")

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# training
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data is created")

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model is ready")
