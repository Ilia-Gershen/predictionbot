from typing import Final

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN: Final = '5660612955:AAHwjnbuOa-PLXv_hR4vrKGT0OKnH-qovx0'
BOT_USERNAME: Final = '@predscazatelcryptobot'

# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await update.message.reply_text('Hello! Thanks for chatting with me! I hope we will see the future together!')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await update.message.reply_text('Choose the crypto of your interest in the menu and I will provide a prediction of the closing price for the next day (depending on the crypto it may take some time, so be patient)')

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
  app = Application.builder().token(TOKEN).build()

  # Commands
  app.add_handler(CommandHandler('start', start_command))
  app.add_handler(CommandHandler('help', help_command))

  # Messages
  app.add_handler(MessageHandler(filters.TEXT, handle_message))

  # Errors
  app.add_error_handler(error)

  # Polls the bot
  print('Polling ....')
  app.run_polling(poll_interval=3)