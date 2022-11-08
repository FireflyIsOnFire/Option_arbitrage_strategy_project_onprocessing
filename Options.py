import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from scipy.stats import norm


df = pd.read_csv('AAPL')['Adj Close']
sp = np.array(df)[-1]

class Options():
    # 0: 2022,11,4
    # 1: 2022,11,11
    # 2: 2022,11,18
    # 3: 2022,11,25
    # 4: 2022,12,2
    # 5: 2022,12,9
    # 6: 2022,12,16
    # 7: 2023,1,20
    log_re = np.log(df / df.shift(1))
    log_re = log_re[-243:-1]  #length: 252 trading days/year
    re = np.log(1.0414)  # risk free rate of 1Y bonds in usa (2022,11)
    vol = log_re.std()  # standard deviation of the stock's returns: 0.021283883795994305

    def __init__(self, ex_date, tau):
        # x[1]: strike price
        # x[2]: real call price
        # x[4]: implied volatility
        # re: log annual return
        # vol: annual standard erro of log return
        # tau: rest option horizon
        self.ex_date = ex_date
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
        x.to_csv('op_data')
        call = []
        put = []
        #print(sp, self.re, self.tau,self.vol, x.iloc[:,1])
        for i in range(len(x)):
            d1 = (np.log(sp / x.iloc[i, 1]) + (self.re + 0.5 * self.vol ** 2) * (self.tau)/365) / (
                        self.vol * np.sqrt((self.tau)/365))
            #print(sp, x.iloc[i,1], self.vol, self.tau)
            d2 = d1 - self.vol * np.sqrt((self.tau)/365)
            c = norm.cdf(d1) * sp - norm.cdf(d2) * x.iloc[i, 1] * np.exp(-self.re * ((self.tau)/365))
            p = norm.cdf(-d2) * x.iloc[i, 1] * np.exp(-self.re * ((self.tau)/365)) - norm.cdf(-d1) * sp
            call.append(c)
            put.append(p)
            #print('Price: ',sp, 'Strike:',x.iloc[i,1], 'TTM: ',self.tau,'Call: ',c,'ImVola:',x.iloc[i,4],'Real call:',x.iloc[i,2])
        return c, p, call, put

    def Greeks(self):
        aapl = yf.Ticker("AAPL")
        df_calls, df_puts = aapl.option_chain(aapl.options[self.ex_date])
        df_calls.to_csv('AAPL_calls')
        df_puts.to_csv('AAPL_puts')
        aapl_calls = df_calls
        aapl_puts = df_puts
        getinfo = ['contractSymbol', 'strike', 'lastPrice', 'openInterest', 'impliedVolatility']
        x = aapl_calls[getinfo]
        x.to_csv('op_data')
        call_delta = []
        put_delta = []
        gamma = []
        x = x.iloc[:,1:]
        #print(x.info)
        #print(x.info, x.iloc[0,0])
        for i in range(len(x)):
            d1 = (np.log(sp / x.iloc[i, 0]) + (self.re + 0.5 * self.vol ** 2) * (self.tau) / 365) / (
                    self.vol * np.sqrt((self.tau)/365))
            c = norm.cdf(d1)
            p = c - 1
            call_delta.append(c)
            put_delta.append(p)
            g =( (1 / np.sqrt(2 * 3.1415926)) * np.exp(-(d1 ** 2) / 2) )/(sp * self.vol * np.sqrt((self.tau)/365))
            gamma.append(g)
        return call_delta, put_delta, gamma




