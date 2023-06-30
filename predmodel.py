import yfinance as yf

import math
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import csv

#will try to launch bot after the prediction is made
import schedule
import time

def getdata(coin: str): #downloading data from yhoo financa and saves it to data.csv file
    #converting input string to capitals
    coin = coin.upper()

    data = yf.download(coin+'-USD', interval= '1d') #we get all the dates available since 2014
    #prints show if everything goes well 
    #print(type(data)) #the data type is the same as in typ
    #print(data.head())
    #print(data.dtypes)
    #empty file before writing
    f = open('predictionbot/data.csv', "w+")
    f.close()
    data.to_csv('predictionbot/data.csv', sep=',', index = True, encoding = 'utf-8')

def DrawingTrainAndValLoss(historyDict): #Function helps to draw the validation and training loss of models, as well as rmsr of training and validation

  loss = historyDict["loss"]
  root_mean_squared_error = historyDict["root_mean_squared_error"]
  valLoss = historyDict["val_loss"]
  val_root_mean_squared_error = historyDict["val_root_mean_squared_error"]

  epochs = range(1, len(loss) + 1)

  fig, (ax1, ax2) = plt.subplots(1, 2)

  fig.set_figheight(5)
  fig.set_figwidth(15)

  ax1.plot(epochs, loss, label = 'Training Loss')
  ax1.plot(epochs, valLoss, label = 'Validation Loss')
  ax1.set(xlabel = "Epochs", ylabel = "Loss")
  ax1.legend()

  ax2.plot(epochs, root_mean_squared_error, label = "Training Root Mean Squared Error")
  ax2.plot(epochs, val_root_mean_squared_error, label = "Validation Root Mean Squared Error")
  ax2.set(xlabel = "Epochs", ylabel = "Loss")
  ax2.legend()

  plt.show()

def CalculateErrors(ETH_Test_Y, ETH_prediction): #Function calculates the evaluation metrics for the models and prints them
  
  mse = mean_squared_error(ETH_Test_Y.reshape(-1,7), ETH_prediction)
  print(f'MSE: {mse}')

  mae = mean_absolute_error(ETH_Test_Y.reshape(-1,7), ETH_prediction)
  print(f'MAE: {mae}')

  rmse = math.sqrt(mean_squared_error(ETH_Test_Y.reshape(-1,7), ETH_prediction))
  print(f'RMSE: {rmse}') 

  mape = np.mean(np.abs(ETH_Test_Y.reshape(-1,7) - ETH_prediction )/np.abs(ETH_Test_Y.reshape(-1,7))) * 100
  print(f'MAPE: {mape}')

def Dataset(Data):

  #Swap date into right format
  Data["Date"] = pd.to_datetime(Data["Date"])

  #Data for Training 
  #From 2018 to 2022
  Train_Data = Data['Close'][Data['Date'] < '2021-12-30'].to_numpy() #taking data befor the given date
  TrainingData = []
  TrainingDataX = []
  TrainingDataY = []

  for i in range(0, len(Train_Data), 7): #putting in close values data in form of weeks aka 7 days
    try:
      TrainingData.append(Train_Data[i : i+7]) #array of arrays of 7 close prices
    except:
      pass

  if len(TrainingData[-1]) < 7: #If last week is less then 7 days we remove last week
    TrainingData.pop(-1)

  TrainingDataX = TrainingData[0 : -1] # taking all but last element
  TrainingDataX = np.array(TrainingDataX)
  TrainingDataX = TrainingDataX.reshape((-1, 7, 1)) #split into groups of 7 

  
  TrainingDataY = TrainingData[1:len(TrainingData)] # Taking all but first element
  TrainingDataY = np.array(TrainingDataY)
  TrainingDataY = TrainingDataY.reshape((-1, 7, 1)) # split into groups of 7 and keep the order

  #Data for Validation
  #From 2022 to 2022-06
  Val_Data = Data['Close'][(Data['Date'] >= '2022-01-01') & (Data['Date'] < '2022-06-01')].to_numpy() #taking data from given date to latest date
  ValData = []
  ValDataX = []
  ValDataY = []

  for i in range(0, len(Val_Data), 7):
    try:
      ValData.append(Val_Data[i : i + 7])
    except:
      pass

  if len(ValData[-1]) < 7:
    ValData.pop(-1)

  ValDataX = ValData[0 : -1]
  ValDataX = np.array(ValDataX)
  ValDataX = ValDataX.reshape((-1, 7, 1))

  ValDataY = ValData[1 : len(ValData)]
  ValDataY = np.array(ValDataY)
  ValDataY = ValDataY.reshape((-1, 7, 1))

  #Data for Testing 
  #From 2022-06 to the end

  Test_Data = Data['Close'][Data['Date'] >= '2022-06-02'].to_numpy()
  TestData = []
  TestDataX = []
  TestDataY = []

  for i in range(0, len(Test_Data), 7):
    try:
      TestData.append(Test_Data[i : i+7])
    except:
      pass

  if len(TestData[-1]) < 7:
    TestData.pop(-1)

  TestDataX = TestData[0 : -1]
  TestDataX = np.array(TestDataX)
  TestDataX = TestDataX.reshape((-1, 7, 1))

  TestDataY = TestData[1 : len(TestData)]
  TestDataY = np.array(TestDataY)
  TestDataY = TestDataY.reshape((-1, 7, 1))
 
  return TrainingDataX, TrainingDataY, ValDataX, ValDataY, TestDataX, TestDataY

def Model():

  model = tf.keras.models.Sequential([ #input shape first number is number of days we will predict and second one is number of purumenters use for prediction
      tf.keras.layers.GRU(200, input_shape = (7,1), activation = tf.nn.leaky_relu, return_sequences = True),
      tf.keras.layers.LSTM(200, activation = tf.nn.leaky_relu), #or make in 150 if it will not work
      tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu),
      tf.keras.layers.Dense(50, activation = tf.nn.leaky_relu),
      tf.keras.layers.Dense(7, activation = tf.nn.leaky_relu)
  ])

  return model

def scheduler(epoch):
  if epoch <= 100:
    lrate = epoch * (10 ** -6)
  else:
    lrate = 0.00001

  return lrate

def predmodel(coin: str):
  
  getdata(coin)
  ETH = pd.read_csv('predictionbot/data.csv')

  ETH_Train_X, ETH_Train_Y, ETH_Val_X, ETH_Val_Y, ETH_Test_X, ETH_Test_Y = Dataset(ETH)

  #graph printing to show the data split
  #plt.figure(figsize = (20, 5))
  #plt.plot(ETH['Date'][ETH['Date'] < '2021-12-30'], ETH['Close'][ETH['Date'] < '2021-12-30'], label = 'Training')
  #plt.plot(ETH['Date'][(ETH['Date'] >= '2022-01-01') & (ETH['Date'] < '2022-06-01')], ETH['Close'][(ETH['Date'] >= '2022-01-01') & (ETH['Date'] < '2022-06-01')], label = 'Validaiton')
  #plt.plot(ETH['Date'][ETH['Date'] >= '2022-06-02'], ETH['Close'][ETH['Date'] >= '2022-06-02'], label = 'Testing')
  #plt.xlabel('Time')
  #plt.ylabel('Closing Price')
  #plt.legend(loc = 'best')

  callback = tf.keras.callbacks.LearningRateScheduler(scheduler) #applying new learning rate to callback that is used as input for model compilation

  ETH_Model = Model()
  ETH_Model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mse', metrics = tf.keras.metrics.RootMeanSquaredError())
  ETH_hist = ETH_Model.fit(ETH_Train_X, ETH_Train_Y, epochs = 200, validation_data = (ETH_Val_X, ETH_Val_Y), callbacks = [callback])

  #DrawingTrainAndValLoss(ETH_hist.history)

  ETH_prediction = ETH_Model.predict(ETH_Test_X)

  # Prediction closeup 
  #in case want to see the prediction in form of graph
  #plt.figure(figsize = (10, 5))

  #plt.plot(ETH['Date'][ETH['Date'] >= '2022-06-02'], ETH['Close'][ETH['Date'] >= '2022-06-02'], label = 'Testing')
  #plt.plot(ETH['Date'][(ETH['Date'] >= '2022-06-02')&(ETH['Date'] <= '2023-06-14')], ETH_prediction.reshape(-1), label = 'Predictions')
  #plt.xlabel('Time')
  #plt.ylabel('Closing Price')
  #plt.legend(loc = 'best')
  #plt.show()

  #in case need to see error values
  #CalculateErrors(ETH_Test_Y, ETH_prediction)

  #now we have to take the last 7 values from ETH and pass it into model for prediction
  last_week = ETH['Close'].tail(7).to_numpy()
  last_week = last_week.reshape((-1, 7, 1))
  pred_for_next_week = ETH_Model.predict(last_week)

  #converting results of prediction to list and saving in correct file
  pred_list = pred_for_next_week[0].tolist() #as result we get list of float values (7)
  pred_list = [int(x) for x in pred_list] #now we get list of 7 int pred for next week
  with open('predictionbot/'+coin+'.csv', 'w', encoding='UTF8') as f:
    coin = coin.upper()
    writer = csv.DictWriter(f, fieldnames = [coin])
    writer.writeheader()
    for elem in pred_list:
      writer.writerow({coin:elem})

def main():
  predmodel('btc')
  predmodel('eth')

if __name__ == '__main__':

  main()

  schedule.every().day.at("15:15").do(main)

  while True:
    schedule.run_pending()
    time.sleep(60)
