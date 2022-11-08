import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from Options import Options

pd.set_option('display.max_columns',None)

start = datetime.datetime(2020,1,1)
end = datetime.datetime.today()
df = web.DataReader('AAPl','yahoo',start,end)['Adj Close']
df.to_csv('df')
sp = df[-1]
expire_date = ['2022,11,11','2022,11,18','2022,11,25','2022,12,2','2022,12,9','2022,12,16','2023,1,20','2023,2,17','2023,3,17','2023,4,21','2023,5,19','2023,6,16','2023,7,21','2023,9,15']

op_end0 = datetime.datetime(2022,11,11)
tau0 = (op_end0 - end).days
op_end1 = datetime.datetime(2022,11,18)
tau1 = (op_end1 - end).days
op_end2 = datetime.datetime(2022,11,25)
tau2 = (op_end2 - end).days
op_end3 = datetime.datetime(2022,12,2)
tau3 = (op_end3 - end).days
op_end4 = datetime.datetime(2022,12,9)
tau4 = (op_end4 - end).days
op_end5 = datetime.datetime(2022,12,16)
tau5 = (op_end5 - end).days
op_end6 = datetime.datetime(2023,1,20)
tau6 = (op_end6 - end).days
op_end7 = datetime.datetime(2023,2,17)
tau7 = (op_end7 - end).days
op_end8 = datetime.datetime(2023,3,17)
tau8 = (op_end8 - end).days
op_end9 = datetime.datetime(2023,4,21)
tau9 = (op_end9 - end).days

# need to calculate tau manually !!!!!!

option = Options(2,tau3)
BSMcall = option.BSM()[2]
strike_price = pd.read_csv('op_data').iloc[:,2]
real_call_price = pd.read_csv('op_data').iloc[:,3]
#print(BSMcall,real_call_price,strike_price)

'''class Collect():
    def __init__(self, ticker, type, start, end):
        self.ticker = ticker
        self.type = type
        self.start = start
        self.end = end

    def Stock(self):
        stock = web.DataReader(self.ticker, 'yahoo', start, end)
        stock.to_csv(self.ticker)

    def Option(self,ex_date):
        option = yf.Ticker(self.ticker)
        op_calls, op_puts = option.option_chain(option.options[ex_date])
        op_calls.to_csv('Option_calls')
        op_puts.to_csv('Option_puts')'''




plt.style.use('bmh')
plt.figure(figsize = (8,5))
plt.figure(1)
plt.title('Option Pricing and Greeks')
plt1 = plt.subplot(212)
#plt1.title('BSM option price vs trading option price')
#plt1.xlabel('Strike Price')
#plt1.ylabel('Call price')
plt1.plot(strike_price, real_call_price, color = 'r', linewidth = 1, label = 'trading call price')
plt1.plot(strike_price, BSMcall, color = 'b', linewidth = 1, label = 'BSM call price')
plt1.legend()


call_delta = option.Greeks()[0]
put_delta = option.Greeks()[1]
gamma = option.Greeks()[2]


plt2 = plt.subplot(221)
#plt2.title('Option Delta')
#plt2.xlabel('Strike price')
#plt2.ylabel('Delta')
plt2.plot(strike_price,call_delta, color = 'b', linewidth = 1, label = 'Call Delta')
plt2.plot(strike_price,put_delta, color = 'g', linewidth = 1, label = 'Put Delta')
plt2.legend()

plt3 = plt.subplot(222)
#plt3.title('Option Gamma')
plt3.plot(strike_price, gamma, color = 'r', linewidth = 1, label = 'Gamma')
plt3.legend()
plt.show()
