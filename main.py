import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from pandas_datareader import Options
import yfinance as yf
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',None)


start = datetime.datetime(2021,11,1)
end = datetime.datetime.today()
df = web.DataReader('AAPl','yahoo',start,end)['Adj Close']
sp = df[-1]
log_re = np.log(df/df.shift(1))
#print(log_re[-243:-1])  #length: 252 trading days/year
an_re = np.log(1.0414) #risk free rate of 1Y bonds in usa (2022,11)
an_vo = log_re.std() #standard deviation of the stock's returns: 0.021283883795994305
'''aapl = yf.Ticker("AAPL")
df_calls, df_puts = aapl.option_chain(aapl.options[1])
# 0: 2022,11,4
# 1: 2022,11,11
# 2: 2022,11,18
# 3: 2022,11,25
# 4: 2022,12,2
# 5: 2022,12,9
# 6: 2022,12,16
# 7: 2023,1,20

df_calls.to_csv('AAPL_calls')
df_puts.to_csv('AAPL_puts')
aapl_calls = df_calls
aapl_puts = df_puts

getinfo = ['contractSymbol', 'strike', 'lastPrice', 'openInterest', 'impliedVolatility']
x = aapl_calls[getinfo]
print(x)
'''


class Options():

    def __init__(self, ex_date ,sp, strike, vol, re, tau):
        # x[1]: strike price
        # x[2]: real call price
        # x[4]: implied volatility
        # re: log annual return
        # vol: annual standard erro of log return
        # tau: rest option horizon
        aapl = yf.Ticker("AAPL")
        self.ex_date = ex_date
        #print(self.ex_date) 是1没错
        # 0: 2022,11,4
        # 1: 2022,11,11
        # 2: 2022,11,18
        # 3: 2022,11,25
        # 4: 2022,12,2
        # 5: 2022,12,9
        # 6: 2022,12,16
        # 7: 2023,1,20
        self.sp = sp
        self.strike = strike
        self.vol = vol
        self.re = re
        self.tau = tau

    def BSM(self):
        aapl = yf.Ticker("AAPL")
        df_calls, df_puts = aapl.option_chain(aapl.options[self.ex_date])
        df_calls.to_csv('AAPL_calls')
        df_puts.to_csv('AAPL_puts')
        aapl_calls = df_calls
        aapl_puts = df_puts
        getinfo = ['contractSymbol', 'strike', 'lastPrice', 'openInterest', 'impliedVolatility']
        x = aapl_calls[getinfo]
        print(x)
        call = []
        put = []
        for i in range(len(x)):
            d1 = (np.log(sp / x.iloc[i, 1]) + (self.re + 0.5 * self.vol ** 2) * self.tau) / (
                        self.vol * np.sqrt(self.tau))
            d2 = d1 - self.vol * np.sqrt(self.tau)
            c = norm.cdf(d1) * sp - norm.cdf(d2) * x.iloc[i, 1] * np.exp(-self.re * self.tau)
            p = norm.cdf(-d2) * x.iloc[i, 1] * np.exp(-self.re * self.tau) - norm.cdf(-d1) * sp
            call.append(c)
            put.append(p)
        return c, p


option = Options(1,1,1,1,1,1)
option.BSM()


'''
tau = []
expire_date = ['2022,11,11','2022,11,18','2022,11,25','2022,12,2','2022,12,9','2022,12,16','2023,1,20','2023,2,17','2023,3,17','2023,4,21','2023,5,19','2023,6,16','2023,7,21','2023,9,15']
pd.DataFrame(expire_date)
#print(len(expire_date)) :14


op_end = []
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
print(tau0,tau1,tau2,tau3,tau4,tau5,tau6,tau7,tau8,tau9)



aapl= yf.Ticker("AAPL")
df_calls, df_puts = aapl.option_chain(aapl.options[1])
# 0: 2022,11,4
# 1: 2022,11,11
# 2: 2022,11,18
# 3: 2022,11,25
# 4: 2022,12,2
# 5: 2022,12,9
# 6: 2022,12,16
# 7: 2023,1,20

df_calls.to_csv('AAPL_calls')
df_puts.to_csv('AAPL_puts')
aapl_calls = df_calls
aapl_puts = df_puts

getinfo = ['contractSymbol','strike','lastPrice','openInterest','impliedVolatility']
op_data = aapl_calls[getinfo]
#print(op_data)

call = []
put = []

def call_option_Pricing(used_data, sp, r, std, tau):
    #x[1]: strike price
    #x[2]: real call price
    #x[4]: implied volatility
    #r: log annual return
    #std: annual standard erro of log return
    #tau: rest option horizon
    used_data = x
    d1 = (np.log(sp/x[1])+(r+0.5*std**2)*tau)/(std*np.sqrt(tau))
    d2 = d1 - std*np.sqrt(tau)
    c = norm.cdf(d1)*sp - norm.cdf(d2)*x[1]*np.exp(-r*tau)
    p = norm.cdf(-d2)*x[1]*np.exp(-r*tau) - norm.cdf(-d1)*sp
    call.append(c)
    put.append(p)
    return c,p

for i in range(len(op_data)):
     x = op_data.iloc[i,:]
     call_option_Pricing(x, sp, an_re, an_vo, tau)



strike_price = np.array(op_data['strike'])
real_call_price = np.array(op_data['lastPrice'])
plt.style.use('bmh')
plt.figure(figsize = (8,5))
plt.xlabel('Strike Price')
plt.ylabel('Call price')
plt.plot(strike_price, real_call_price, color = 'r', linewidth = 1, label = 'real call price')
plt.plot(strike_price, call, color = 'b', linewidth = 1, label = 'BSM call price')
plt.legend()
#plt.show()

'''