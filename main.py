import logging
# import sklearn
from os import getenv
from sys import exit
import string
import pymorphy2
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor
from aiogram.utils.exceptions import BotBlocked
# import nltk
# from nltk.stem import WordNetLemmatizer
import pickle
# from keras.models import load_model

from model import Model
# from test_model import predict_class

bot_token = getenv("BOT_TOKEN")
if not bot_token:
    exit("Error: no token provided")

bot = Bot(token=bot_token)

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
logging.basicConfig(level=logging.INFO)

morph = pymorphy2.MorphAnalyzer()

model = Model()
# model = load_model('chatbot_model.h5')
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))

stop_phrases = ['по какой причине', 'добрый день', 'добрый вечер', 'доброе утро', 'к сожалению', 'в очередной раз',
                'у меня', 'по факту']
stop_words = {'здравствуйте', 'почему', 'уже', 'я', 'есть', 'тут', 'а', 'и', 'нигде', 'снова', 'как', 'мой', 'но',
              'этой', 'в', 'хорошо', 'понял', 'поняла', 'пожалуйста', 'от', 'с', 'подскажите', 'так', 'спасибо'}


def remove_stop_phrases(phrase, stops):
    for w in stops:
        if w in phrase:
            phrase = phrase.replace(w, '')
    return phrase


def preprocess_msg(msg):
    msg = msg.translate(str.maketrans('', '', string.punctuation)).lower()
    msg = remove_stop_phrases(msg, stop_phrases)
    query = msg.split()

    for i in range(len(query)):
        query[i] = morph.parse(query[i])[0].normal_form
    question = ' '.join(list(set(query) - stop_words))
    return question


@dp.message_handler()
async def processing(message: types.Message):
    print(message)
    question = preprocess_msg(message.text)
    try:
        q, a = model.get_top_ans([question])
        # top = 'most relevant answer: ' + a[0] + '\n' + 'other:\n'
        # for i in range(1, len(a)):
        #     top += str(i) + ') ' + a[i] + '\n'

        print(message.text)
        print(q)
        print(a, '\n')
        await message.answer(a[0])

    except Exception as e:
        print(message.text)
        print('ERROR')
        print(e, '\n')
        await message.answer(str(e))


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
