#region imports
from AlgorithmImports import *
from ou_process import *
from sklearn.linear_model import LinearRegression
import pandas as pd
#endregion


# Your New Python File
class Pair():

    def __init__(self, s1, s2, data_source):
        self.A = s1
        self.B = s2
        self.price_A, self.price_B, self.beta = self.calc_beta(data_source)
        self.theta, self.mu, self.sigma, self.ou_beta = arg_max_B_alloc(self.price_A, self.price_B)
        # self.ou_beta = arg_max_B_alloc(self.price_A, self.price_B)[-1]
        self.position = 0
        # self.update_spread()
        # self.update_ma_spread()
        # self.update_std_spread()
        # self.update_entry_point()
        # self.update_exit_point()
        
    def get_ou_param(self):
        return self.theta, self.mu, self.sigma, self.ou_beta

    def calc_beta(self, data):
        price1 = data[data["symbol"] == self.A]["close"].values
        price2 = data[data["symbol"] == self.B]["close"].values
        price1_reshaped = price1.reshape(-1, 1)
        price2_reshaped = price2.reshape(-1, 1)
        model = LinearRegression(fit_intercept = True)
        model.fit(price2_reshaped, price1_reshaped)
        beta = model.coef_[0][0]
        return pd.Series(price1), pd.Series(price2), beta


    def update_spread(self):
        self.spread = self.price_A - self.beta * self.price_B

    def update_ma_spread(self):
        self.ma_spread = self.spread.rolling(window=20).mean()

    def update_std_spread(self):
        self.std_spread = self.spread.rolling(window=20).std()

    def update_entry_point(self):
        self.lower_enter, self.upper_enter = (self.ma_spread - 2 * self.std_spread, self.ma_spread + 2 * self.std_spread)

    def update_exit_point(self):
        self.lower_exit, self.upper_exit = (self.ma_spread - self.std_spread, self.ma_spread + self.std_spread)

    def update_price_A(self, price):
        self.price_A.append(pd.Series(price))

    def update_price_B(self, price):
        self.price_B.append(pd.Series(price))

