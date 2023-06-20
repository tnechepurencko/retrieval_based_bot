import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def get_pairwise_similarity(corpus):
    vect = TfidfVectorizer(min_df=1, stop_words="english")
    tfidf = vect.fit_transform(corpus)
    pairwise_similarity = tfidf * tfidf.T
    return pairwise_similarity.toarray()


def get_categories(pairwise_similarity, accuracy=0.5):
    w = np.where(pairwise_similarity > accuracy)
    categories = dict()

    for i in range(len(w[0])):
        in_cat = False
        for k in categories.keys():
            if w[1][i] in categories[k]:
                in_cat = True
                break
        if not in_cat:
            in_cat = -1
            for k in categories.keys():
                if w[0][i] in categories[k]:
                    in_cat = k
                    break
            if in_cat != -1:
                categories[in_cat].append(w[1][i])
            else:
                if w[0][i] not in categories.keys():
                    categories[w[0][i]] = []
                categories[w[0][i]].append(w[1][i])
    return categories


def reduce(df, answers, categories):
    reduced_cats = list(filter(lambda x: len(x) < 5, list(categories.values())))  # remove them from db
    print(reduced_cats)
    print('len of reduced_cats:', len(reduced_cats))

    s = set()
    for x in reduced_cats:
        s |= set(x)

    print('set:', s)
    print('len of set of reduced_cats:', len(s))

    # lst = [answers[i] for i in s]
    # print('lst:', lst)

    df = df.drop(list(s))
    # df = df.drop(df[df['answer'] in lst].index)
    return df


data = pd.read_csv('../data/formatted_msgs.csv', sep='\t')
tags = data['answer'].tolist()
print('original len of answers:', len(tags))

arr = get_pairwise_similarity(tags)
cats = get_categories(arr)
# print(cats)
print('num of cats:', len(cats))

# check uniqueness

num_of_elems = 0
for k in cats.keys():
    num_of_elems += len(cats[k])

print('num of elems:', num_of_elems)


# lens = [len(x[1]) for x in list(cats.items())]
# print('lens:', sorted(lens, reverse=True))

# s_lst = list(cats.items())
# s_lst.sort(key=lambda x: len(x[1]), reverse=True)
# print('s_lst:', s_lst)

new_df = reduce(data, tags, cats)
print('new_df.shape:', new_df.shape)
new_df.to_csv('new_df.csv', sep='\t', encoding='utf-8', index=False)

# tags = new_df['answer'].tolist()
#
# arr = get_pairwise_similarity(tags)
# cats = get_categories(arr)
# print('new num of cats:', len(cats))

