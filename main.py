import logging
from os import getenv
from sys import exit
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor
from aiogram.utils.exceptions import BotBlocked
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from keras.models import load_model
from test_model import predict_class

bot_token = getenv("BOT_TOKEN")
if not bot_token:
    exit("Error: no token provided")

bot = Bot(token=bot_token)

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
logging.basicConfig(level=logging.INFO)

lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


@dp.message_handler()
async def processing(message: types.Message):
    print(message)
    ints = predict_class(message.text, model)
    if len(ints) == 0:
        print('\tcannot predict')
        print('\tquerry:', message.text)
        await message.answer('cannot predict')
    else:
        tag = ints[0]['intent']
        await message.answer(tag)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
