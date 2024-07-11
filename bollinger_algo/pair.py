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
        self.ou_beta = arg_max_B_alloc(self.price_A, self.price_B)[-1]
        self.position = 0
        # self.update_spread()
        # self.update_ma_spread()
        # self.update_std_spread()
        # self.update_entry_point()
        # self.update_exit_point()
        
    def calc_beta(self, data):
        price1 = data[data["symbol"] == self.A]["close"].values
        price2 = data[data["symbol"] == self.B]["close"].values
        price1_reshaped = price1.reshape(-1, 1)
        price2_reshaped = price2.reshape(-1, 1)
        model = LinearRegression(fit_intercept = True)
        model.fit(price2_reshaped, price1_reshaped)
        beta = model.coef_[0][0]
        return pd.Series(price1), pd.Series(price2), beta

