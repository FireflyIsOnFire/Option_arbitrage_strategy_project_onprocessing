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
log_re = np.log(df/df.shift(1)).drop(index = '2021-11-01' )
#print(log_re)  #length: 252
an_re = log_re.sum(axis = 0) #annualized risk-free interest rate: 0.016869234533158098
an_vo = log_re.std() #standard deviation of the stock's returns: 0.021283883795994305
op_end = datetime.datetime(2022,11,4)  # it's changing everyday, so need to adjust according to current option contract
tau = (op_end-end).days

aapl= yf.Ticker("AAPL")
df_calls, df_puts = aapl.option_chain(aapl.options[0])
df_calls.to_csv('AAPL_calls')
df_puts.to_csv('AAPL_puts')
aapl_calls = df_calls
aapl_puts = df_puts

getinfo = ['contractSymbol','strike','lastPrice','openInterest','impliedVolatility']
op_data = aapl_calls[getinfo]
print(op_data)
x = op_data.iloc[4, :]
print(x)

def call_option_Pricing(used_data, sp, r, std, tau):
    '''x[1]: strick price
    x[2]: real call price
    x[4]: implied volatility
    r: log annual return
    std: annual standard erro of log return
    tau: rest option horizon'''
    used_data = x
    d1 = (np.log(sp/x[1])+(r+0.5*std**2)*tau)/(std*np.sqrt(tau))
    d2 = d1 - std*np.sqrt(tau)
    c = norm.cdf(d1)*sp - norm.cdf(d2)*x[1]*np.exp(-r*tau)
    p = norm.cdf(-d2)*x[1]*np.exp(-r*tau) - norm.cdf(-d1)*sp
    print(c, p)
    return c,p

call_option_Pricing(x,sp,an_re,an_vo,tau)


















