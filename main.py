from typing import Final

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler, Updater

import pandas as pd
from datetime import date

import os

import yfinance as yf
import pickle

TOKEN: Final = os.getenv('BOTAPIKEY')
BOT_USERNAME: Final = '@predscazatelcryptobot'

# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await update.message.reply_text('Hello! Thanks for chatting with me! I hope we will see the future together!')
  
  await menu_command(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await update.message.reply_text('Choose the crypto of your interest in the menu and I will provide a prediction of the closing price for the next day (depending on the crypto it may take some time, so be patient). Use /menu to request a menu')

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

  keyboard = [
    [
      InlineKeyboardButton("BTC", callback_data= 'btc'),
      InlineKeyboardButton("ETH", callback_data= 'eth')
    ],
    [
      InlineKeyboardButton("LTC", callback_data= 'ltc'),
      InlineKeyboardButton("XMR", callback_data= 'xmr')
    ],
  ]

  reply_markup = InlineKeyboardMarkup(keyboard)

  await update.message.reply_text('Please choose:', reply_markup=reply_markup)

# Buttons
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):

  query = update.callback_query #shortcut to access provided CallbackQuery -> part of update that has all the information

  await query.answer()

  days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

  #load data for prediction
  data = yf.download(query.data+'-USD', start="2023-07-01", interval= '1d')
  loadedModel = pickle.load(open(query.data+'.sav', 'rb'))
  last_week = data['Close'].tail(7).to_numpy()
  last_week = last_week.reshape((-1, 7, 1))
  pred_for_next_week = loadedModel.predict(last_week)
  pred_list = pred_for_next_week[0].tolist()
  pred_list = [int(x) for x in pred_list] #now we get list of 7 int pred for next week

  if query.data == 'btc':
    #data = pd.read_csv(query.data+".csv")
    #prediction = data.BTC

    prediction = pred_list #prediction.to_list()

    #now we have to convert it into column of values with green and red arrows
    today = date.today().weekday()
    ans = " "
    for i in range(len(prediction)):
      if today >= 7:
        today = today - 7
      ans += days[today] + '     ' + str(prediction[i]) + '\n '
      today += 1

    await query.edit_message_text(text="Here is your prediction of BTC closing price for the next 7 days: \n\n" + ans)
  
  elif query.data == 'eth':
    #data = pd.read_csv(query.data+".csv")
    #prediction = data.ETH
    prediction = pred_list #prediction.to_list()

    #now we have to convert it into column of values with green and red arrows
    today = date.today().weekday()
    ans = " "
    for i in range(len(prediction)):
      if today >= 7:
        today = today - 7
      ans += days[today] + '     ' + str(prediction[i]) + '\n '
      today += 1

    await query.edit_message_text(text="Here is your prediction of ETH closing price for the next 7 days: \n\n" + ans)

  elif query.data == 'ltc':
    #data = pd.read_csv(query.data+".csv")
    #prediction = data.LTC
    prediction = pred_list #prediction.to_list()

    #now we have to convert it into column of values with green and red arrows
    today = date.today().weekday()
    ans = " "
    for i in range(len(prediction)):
      if today >= 7:
        today = today - 7
      ans += days[today] + '     ' + str(prediction[i]) + '\n '
      today += 1

    await query.edit_message_text(text="Here is your prediction of LTC closing price for the next 7 days: \n\n" + ans)

  elif query.data == 'xmr':
    #data = pd.read_csv(query.data+".csv")
    #prediction = data.XMR
    prediction = pred_list #prediction.to_list()

    #now we have to convert it into column of values with green and red arrows
    today = date.today().weekday()
    ans = " "
    for i in range(len(prediction)):
      if today >= 7:
        today = today - 7
      ans += days[today] + '     ' + str(prediction[i]) + '\n '
      today += 1

    await query.edit_message_text(text="Here is your prediction of XMR closing price for the next 7 days: \n\n" + ans)

  else:
    await query.edit_message_text(text=f"Selected option: {query.data}")

# Responses
def handle_response(text: str) -> str: 
  processed: str = text.lower()

  if 'hello' in processed:
    return 'Hey there !'
  
  return 'I do not understand what you wrote, please use menu buttons for interactions'

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
  message_type: str = update.message.chat.type
  text: str = update.message.text

  print(f'User({update.message.chat.id}) in {message_type}: "{text}"')

  if message_type == 'group': #if bot is tagged in the message (ex somebody is talking to it in a group) 
    if BOT_USERNAME in text:
      new_text: str = text.replace(BOT_USERNAME, '').strip()
      response: str = handle_response(new_text)
    else:
      return
    
  else:
    response: str = handle_response(text)

  print('Bot:', response)
  await update.message.reply_text(response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
  print(f'Update {update} caused error {context.error}')

if __name__ == '__main__':

  print('Starting BOT')
  #app = Application.builder().token(TOKEN).build()
  bot = Bot(token = TOKEN)
  app = Application.builder().token(TOKEN).build()
  updater = Updater(TOKEN, update_queue=True)

  # Commands
  #app.add_handler(CommandHandler('start', start_command))
  #app.add_handler(CommandHandler('help', help_command))
  app.add_handler(CommandHandler("start", start_command))
  app.add_handler(CommandHandler("help", help_command))
  
  #app.add_handler(CommandHandler('menu', menu_command))
  app.add_handler(CommandHandler("menu", menu_command))
  
  # triggered when inline buttons are used by user
  #app.add_handler(CallbackQueryHandler(button))
  app.add_handler(CallbackQueryHandler(button))

  # Messages
  #app.add_handler(MessageHandler(filters.TEXT, handle_message))
  app.add_handler(MessageHandler(filters.TEXT, handle_message))

  # Errors
  #app.add_error_handler(error)
  app.add_error_handler(error)

  # Polls the bot
  print('Polling ....')
  
  #instead of just polling we will do infinite loop with pooling and status update
  #app.run_polling(poll_interval=3)

  PORT = int(os.environ.get('PORT', '3000'))
  HOOK_URL = 'https://predictionbot-yotx.codecapsules.co.za' + '/' + TOKEN
  bot.setWebhook(HOOK_URL)
  updater.start_webhook(listen='0.0.0.0', port=PORT, url_path=TOKEN)
  updater.bot.setWebhook(HOOK_URL)
  updater.idle()

