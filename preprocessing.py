import pandas as pd
import string
import re
import pymorphy2
from sklearn.model_selection import train_test_split
from russiannames import parser


# print(parser.NamesParser().)


def remove_hello_words(words, phrase):
    if words in phrase.lower():
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

        if len(words.split(' ')) == 1:
            for x in sentences:
                if not (words in x.lower() and len(x.split(' ')) < 3):
                    new_answer += x
        else:
            for x in sentences:
                if not (words.split(' ')[0] in x.lower() and words.split(' ')[1] in x.lower() and len(
                        x.split(' ')) < 4):
                    new_answer += x

        while new_answer[0] == ' ':
            new_answer = new_answer[1:]
        return new_answer

    else:
        return phrase


def make_one_line(phrase):
    answer = phrase.replace('\t', ' ')
    answer = answer.replace('\n', ' ')
    answer = answer.replace('\r', ' ')
    return answer


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

    answer = make_one_line(answer)
    answer = remove_names(answer, unique_names)
    # for w in ['здравствуйте', 'добрый день', 'добрый вечер', 'доброе утро']:
    #     answer = remove_hello_words(w, answer)

    return ' '.join(answer.split(' '))


def initial_preprocessing():
    df = pd.read_csv('support_messages.csv')
    for p in ['Мы получили Ваше сообщение!', 'Оцените работу поддержки.', 'Мы всё еще можем Вам помочь?']:
        df = df.drop(df[df['text'].str.contains(p)].index)

    helper = df[df['text'].str.split().str.len().lt(3)]
    helper = helper[helper['text'].str.contains('@')]
    helper = helper[helper['messageFrom'].str.contains('CLIENT')]
    df = df.drop(df[df['text'].str.split().str.len().lt(3)].index)

    df = pd.concat([df, helper], ignore_index=True)
    return df


def remove_stop_phrases(phrase, stops):
    for w in stops:
        if w in phrase:
            phrase = phrase.replace(w, '')
    return phrase


def process_query(query, row, stops):
    orig = ' '.join(query)

    for i in range(len(query)):
        query[i] = morph.parse(query[i])[0].normal_form

    question = ' '.join([item for item in query if item not in stops])
    answer = answer_preprocessing(row['text'])
    return [orig, question, answer]


def remove_names(phrase, unique):
    for p in string.punctuation:
        if p in phrase:
            phrase = phrase.replace(p, ' ' + p + ' ')

    words = phrase.split(' ')
    without_names = []
    for w in words:
        if w != '':
            if w in string.punctuation and len(without_names) > 0:
                if without_names[-1][-1] in string.punctuation:
                    without_names[-1] = without_names[-1][:-1] + w
                else:
                    without_names[-1] += w
            elif w not in string.punctuation and w.lower() not in unique:
                without_names.append(w)
    if len(without_names) > 0:
        without_names[0] = without_names[0].title()

    phrase = ' '.join(without_names)
    if 'help@ scan. com. ru' in phrase:
        phrase = phrase.replace('help@ scan. com. ru', 'help@scan.com.ru')

    return phrase


if __name__ == '__main__':
    df = initial_preprocessing()
    names_df = pd.read_csv('ru_names.csv', sep='\t')
    unique_names = names_df['text'].str.lower().unique()

    data = {'original': [], 'question': [], 'answer': []}
    new_df = pd.DataFrame(data)

    # creating formatted db
    morph = pymorphy2.MorphAnalyzer()
    stop_phrases = ['по какой причине', 'добрый день', 'добрый вечер', 'доброе утро', 'к сожалению', 'в очередной раз',
                    'у меня', 'по факту']
    stop_words = {'здравствуйте', 'почему', 'уже', 'я', 'есть', 'тут', 'а', 'и', 'нигде', 'снова', 'мой', 'но',
                  'этой', 'в', 'хорошо', 'понял', 'поняла', 'пожалуйста', 'от', 'с', 'подскажите', 'так', 'спасибо',
                  'почемуто', 'почему-то', 'на', 'тогда', 'ли', 'день', 'вопрос'}

    for x in df['usedeskChatId'].unique():
        df_usr = df.loc[df['usedeskChatId'] == x]
        query = []
        for idx, row in df_usr.sort_values('createdDate').iterrows():
            if len(query) > 0 and row['messageFrom'] == 'OPERATOR':
                new_df.loc[len(new_df)] = process_query(query, row, stop_words)
                query = []
            elif row['messageFrom'] == 'OPERATOR':
                continue
            else:
                msg = row['text'].translate(str.maketrans('', '', r"""!"#$%&'()*+,-./:;=?[\]^_`{|}~""")).lower()
                msg = remove_stop_phrases(msg, stop_phrases)
                query.extend(msg.split())

    new_df.to_csv('formatted_msgs.csv', sep='\t', encoding='utf-8', index=False)

    # new_df = new_df.sample(n=1000)
    train_data, test_data = train_test_split(new_df, test_size=0.2)
    train_data.to_csv('train_msgs.csv', sep='\t', encoding='utf-8', index=False)
    test_data.to_csv('test_msgs.csv', sep='\t', encoding='utf-8', index=False)
