# region imports
from AlgorithmImports import *
from pair_selector import *
from ou_process import *
# from bollinger_band import *
from datetime import datetime
from pair import Pair
from config import *
import time
# endregion


class CryingFluorescentYellowRhinoceros(QCAlgorithm):
    
    def Initialize(self):
        # Set up algo
        CONFIG = Config()
        
        self.SetStartDate(CONFIG.BACKTEST_START)
        self.SetEndDate(CONFIG.BACKTEST_END)
        self.SetBenchmark(CONFIG.BENCHMARK)
        self.formation_start = CONFIG.FORMATION_START
        self.formation_end = CONFIG.FORMATION_END
        self.SetCash(CONFIG.CASH)  
        self.DataNormalizationMode = CONFIG.NORMALIZATION_MODE
        self.universe = CONFIG.UNIVERSE
        self.qb = QuantBook()
        self.port_value = CONFIG.CASH
        self.last_update = CONFIG.BACKTEST_START
        # Get data
        symbols = [self.qb.AddEquity(i, dataNormalizationMode = DataNormalizationMode.Adjusted).Symbol for i in self.universe]
        start = time.time()
        history = self.qb.History(symbols, self.formation_start, self.formation_end, Resolution.Daily)
        end = time.time()
        self.Log("Time for history request")
        self.Log(end - start)
        self.symbol_dict = {}
        symbols_list = history.index.get_level_values(0).unique().tolist()
        for i in range(len(self.universe)):
            self.symbol_dict[symbols_list[i]] = symbols[i]
        # Select pairs
        pair_selector = PairSelector(history)
        start = time.time()
        self.selected_pairs = pair_selector.select_pairs()
        end = time.time()
        self.Log("Time to find suitable pairs")
        self.Log(end - start)
        self.symbol_list = []
        self.price_dict = {}
        for p in self.selected_pairs:
            start = time.time()
            symbol1 = self.symbol_dict[p.A]
            self.AddEquity(symbol1, Resolution.Hour, dataNormalizationMode = DataNormalizationMode.Adjusted)
            symbol2 = self.symbol_dict[p.B]
            self.AddEquity(symbol2, Resolution.Hour, dataNormalizationMode = DataNormalizationMode.Adjusted)
            self.symbol_list.append([symbol1, symbol2])
            self.Log([symbol1.Value, symbol2.Value])
            end = time.time()
            self.Log("Time takes to AddEquity")
            self.Log(end - start)
            # price1 = self.History(symbol1, TimeSpan.FromHours(50*24), Resolution.Hour).loc[:,"close"]
            # price2 = self.History(symbol2, TimeSpan.FromHours(50*24), Resolution.Hour).loc[:,"close"]
            price1 = pd.Series(self.History(symbol1, TimeSpan.FromDays(50), Resolution.Hour)["close"].values)
            price2 = pd.Series(self.History(symbol2, TimeSpan.FromDays(50), Resolution.Hour)["close"].values)
            self.price_dict[symbol1] = price1.tail(140)
            self.price_dict[symbol2] = price2.tail(140)


    def refresh(self, start_date):
        formation_period_start = start_date - relativedelta(months=10)
        formation_period_end = start_date - timedelta(days=1)
        symbols = [self.qb.AddEquity(i, dataNormalizationMode = DataNormalizationMode.Adjusted).Symbol for i in self.universe]
        history = self.qb.History(symbols, formation_period_start, formation_period_end, Resolution.Daily)
        self.symbol_dict = {}
        symbols_list = history.index.get_level_values(0).unique().tolist()
        for i in range(len(self.universe)):
            self.symbol_dict[symbols_list[i]] = symbols[i]
        pair_selector = PairSelector(history)
        self.selected_pairs = pair_selector.select_pairs()
        self.symbol_list = []
        self.price_dict = {}
        for p in self.selected_pairs:
            symbol1 = self.symbol_dict[p.A]
            self.AddEquity(symbol1, Resolution.Hour, dataNormalizationMode = DataNormalizationMode.Adjusted)
            symbol2 = self.symbol_dict[p.B]
            self.AddEquity(symbol2, Resolution.Hour, dataNormalizationMode = DataNormalizationMode.Adjusted)
            self.symbol_list.append([symbol1, symbol2])
            price1 = pd.Series(self.History(symbol1, TimeSpan.FromDays(50), Resolution.Hour)["close"].values)
            price2 = pd.Series(self.History(symbol2, TimeSpan.FromDays(50), Resolution.Hour)["close"].values)
            self.price_dict[symbol1] = price1.tail(140)
            self.price_dict[symbol2] = price2.tail(140)
        # self.last_update = start_date

    def OnData(self, data: Slice):
        # check whether the pairs should be refreshed
        if ((self.Portfolio.TotalPortfolioValue - self.port_value) / self.port_value <= -0.10) or (data.Time > self.last_update + relativedelta(months=6)):
            for i in range(len(self.symbol_list)):
                self.selected_pairs[i].position = 0
                self.Liquidate(self.symbol_list[i][0])
                self.Liquidate(self.symbol_list[i][1])
                self.port_value = self.Portfolio.TotalPortfolioValue
                self.refresh(data.Time)
                self.last_update = data.Time
                self.Log("Pairs refreshed")
        # look for trading opportunies       
        for i in range(len(self.symbol_list)):
            if not (data.ContainsKey(self.symbol_list[i][0]) and data.ContainsKey(self.symbol_list[i][1])):
                return
            # TODO: Store data in the rolling window (DONE)
            # self.History.close / pd.Series().to_frame()
            # manage the rolling window
            for k in [0, 1]:
                new_data = data[self.symbol_list[i][k]].Close
                self.price_dict[self.symbol_list[i][k]] = self.price_dict[self.symbol_list[i][k]].shift(-1).dropna()
                self.price_dict[self.symbol_list[i][k]].loc[len(self.price_dict[self.symbol_list[i][k]])] = new_data
            history_A, history_B = self.price_dict[self.symbol_list[i][0]], self.price_dict[self.symbol_list[i][1]]
            spread = history_A - self.selected_pairs[i].beta * history_B
            self.Log(self.selected_pairs[i].beta)
            ma_spread = spread.mean()
            std_spread = spread.std()
            spread = spread.iloc[-1]
            # ma_spread = ma_spread.iloc[-1]
            # std_spread = std_spread.iloc[-1]
            lower_entry, upper_entry = ma_spread - 2 * std_spread, ma_spread + 2 * std_spread
            # lower_exit, upper_exit = ma_spread - std_spread, ma_spread + std_spread
            # apply the trading logic
            lower_exit, upper_exit = ma_spread, ma_spread
            if self.selected_pairs[i].position != 0:
                holding_1 = self.Portfolio[self.symbol_list[i][0]]
                holding_2 = self.Portfolio[self.symbol_list[i][1]]
                investment_amount = holding_1.Quantity * holding_1.AveragePrice + holding_2.Quantity * holding_2.AveragePrice
                return_1 = holding_1.UnrealizedProfit
                return_2 = holding_2.UnrealizedProfit
                if investment_amount == 0:
                    total_return = 0
                else:
                    total_return = (return_1 + return_2) / investment_amount
                if total_return <= -0.05:
                    self.Liquidate(self.symbol_list[i][0])
                    self.Liquidate(self.symbol_list[i][1])
                    self.selected_pairs[i].position = 0
                    self.Log("Cutting off losses")
            if spread >= upper_entry:
                if not self.Portfolio.Invested:
                    self.selected_pairs[i].position = -1
                    self.SetHoldings(self.symbol_list[i][0], (self.selected_pairs[i].ou_beta - 1)/len(self.selected_pairs))
                    self.SetHoldings(self.symbol_list[i][1], (self.selected_pairs[i].ou_beta)/len(self.selected_pairs))
                    # self.port_value[i] = spread
            elif spread <= lower_entry:
                if not self.Portfolio.Invested:
                    self.selected_pairs[i].position = 1
                    self.SetHoldings(self.symbol_list[i][0], (1 - self.selected_pairs[i].ou_beta)/len(self.selected_pairs))
                    self.SetHoldings(self.symbol_list[i][1], (-self.selected_pairs[i].ou_beta)/len(self.selected_pairs))
                    # self.port_value[i] = spread
            elif spread <= upper_exit and self.selected_pairs[i].position == -1:
                self.Liquidate(self.symbol_list[i][0])
                self.Liquidate(self.symbol_list[i][1])
                self.selected_pairs[i].position = 0
            elif spread >= lower_exit and self.selected_pairs[i].position == 1:
                self.Liquidate(self.symbol_list[i][0])
                self.Liquidate(self.symbol_list[i][1])
                self.selected_pairs[i].position = 0 
    