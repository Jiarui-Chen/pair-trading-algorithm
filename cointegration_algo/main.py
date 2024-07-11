# region imports
from AlgorithmImports import *
from pair_selector import *
from cointegration import *

from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.optimize as spop
import matplotlib.pyplot as plt

# endregion

TICKERS = ['ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB',
           'ARE', 'ALGN', 'ALLE', 'ALL', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC',
           'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'AOS', 'APA', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ANET', 'AJG',
           'AIZ', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AVGO', 'BKR', 'BAC', 'BK', 'BAX', 'BDX', 'BRK.B', 'BBY',
           'BIO', 'BIIB', 'BLK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'BR', 'BF.B', 'BEN', 'CHRW', 'CDNS', 'CZR',
           'CPB', 'COF', 'CAH', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CF', 'CHTR',
           'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'CTSH', 'CL',
           'CMCSA', 'CMA', 'CAG', 'COP', 'COO', 'CPRT', 'CTVA', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'CRM', 'DHI', 'DHR',
           'DRI', 'DVA', 'DE', 'DAL', 'DVN', 'DXCM', 'DLR', 'DFS', 'DISH', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW',
           'DTE', 'DUK', 'DD', 'DXC', 'DGX', 'DIS', 'ED', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ENPH',
           'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'FANG',
           'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FRC', 'FISV', 'FLT', 'FMC', 'F', 'FTNT', 'FTV',
           'FBHS', 'FOXA', 'FOX', 'FCX', 'GOOGL', 'GOOG', 'GLW', 'GPS', 'GRMN', 'GNRC', 'GD', 'GE', 'GIS', 'GM', 'GPC',
           'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HAL', 'HBI', 'HIG', 'HAS', 'HCA', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT',
           'HOLX', 'HD', 'HON', 'HRL', 'HWM', 'HPQ', 'HUM', 'HBAN', 'HII', 'IT', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY',
           'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JKHY', 'J',
           'JBHT', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KMX', 'KO', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC',
           'KR', 'LNT', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LEG', 'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ',
           'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'LUV', 'MMM', 'MO', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM',
           'MAS', 'MA', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK',
           'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA',
           'NWS', 'NEE', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NOV', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'NOW',
           'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE', 'ORCL', 'OTIS', 'O', 'PEAK', 'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC',
           'PYPL', 'PENN', 'PNR', 'PEP', 'PKI', 'PRGO', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL',
           'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PTC', 'PEG', 'PSA', 'PHM', 'PVH', 'PWR', 'QRVO', 'QCOM', 'RE', 'RL',
           'RJF', 'RTX', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SCHW', 'STZ',
           'SJM', 'SPGI', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'SHW', 'SPG', 'SWKS', 'SNA', 'SO', 'SWK', 'SBUX', 'STT',
           'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'T', 'TAP', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TGT', 'TEL', 'TDY',
           'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'UDR',
           'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UHS', 'UNM', 'VLO', 'VTR', 'VRSN', 'VRSK',
           'VZ', 'VRTX', 'VFC', 'VTRS', 'V', 'VNO', 'VMC', 'WRB', 'WAB', 'WMT', 'WBA', 'WM', 'WAT', 'WEC', 'WFC',
           'WELL', 'WST', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WYNN', 'XRAY', 'XOM', 'XEL', 'XYL', 'YUM', 'ZBRA',
           'ZBH', 'ZION', 'ZTS']


class cotest(QCAlgorithm):

    def Initialize(self):
        qb = QuantBook()
        tickers = [qb.AddEquity(t, dataNormalizationMode=DataNormalizationMode.Adjusted).Symbol for t in TICKERS]
        history = qb.History(tickers, datetime(2020, 1, 1), datetime(2020, 12, 31), Resolution.Daily)
        history = pd.DataFrame(history["close"])
        history = history.reset_index()

        self.SetStartDate(2021, 1, 1)  # Set Start Date
        self.SetEndDate(2022, 12, 31)
        # self.SetBenchmark("SPY")
        self.SetCash(100000)  # Set Strategy Cash
        self.DataNormalizationMode = "Adjusted"
        self.pair_selector = PairSelector(TICKERS, datetime(2020, 1, 1), datetime(2020, 12, 31))
        self.selected_pairs = self.pair_selector.select_pairs()
        self.symbol_list = []
        for p in self.selected_pairs:
            symbol1 = self.AddEquity(p.A, Resolution.Hour, dataNormalizationMode=DataNormalizationMode.Adjusted).Symbol
            symbol2 = self.AddEquity(p.B, Resolution.Hour, dataNormalizationMode=DataNormalizationMode.Adjusted).Symbol
            self.symbol_list.append([symbol1, symbol2])

    def OnData(self, data: Slice):

        for i in range(len(self.symbol_list)):
            try:

                history_A = pd.Series(
                    self.History(self.symbol_list[i][0], TimeSpan.FromDays(21), Resolution.Hour)["close"].values)
                history_B = pd.Series(
                    self.History(self.symbol_list[i][1], TimeSpan.FromDays(21), Resolution.Hour)["close"].values)
                raw_data = pd.DataFrame()

                # specifying maximum KPSS statistic (95% critical value)
                KPSS_max = 0.463
                # specifying the KPSS test (one-parameter unbiased or two-parameter)
                unbiased = 1
                # specifying whether to perform beta-loading or not
                beta_loading = 0

                current_return = 0
                tickers = [self.symbol_list[i][0], self.symbol_list[i][0]]
                raw_data[tickers[0]] = history_A
                raw_data[tickers[1]] = history_B

                # for t in range(window, len(history_A)-1):

                # specifying the subsample
                # data = raw_data[t-window:t]
                # stock 2 = a + b*stock 1
                # OLS parameters as starting values
                old_signal = self.selected_pairs[i].position
                raw_data = raw_data.dropna()
                reg = sm.OLS(raw_data[tickers[1]], sm.add_constant(raw_data[tickers[0]]))

                res = reg.fit()
                a0 = res.params[0]
                b0 = res.params[1]
                if unbiased == 1:
                    # defining the KPSS function (unbiased one-parameter forecast)
                    def KPSS(b):
                        a = np.average(np.array(raw_data[tickers[1]]) - np.array(b * raw_data[tickers[0]]))
                        resid = np.array(raw_data[tickers[1]] - (a + b * raw_data[tickers[0]]))
                        cum_resid = np.cumsum(resid)
                        st_error = (np.sum(resid ** 2) / (len(resid) - 2)) ** (1 / 2)
                        KPSS = np.sum(cum_resid ** 2) / (len(resid) ** 2 * st_error ** 2)
                        return KPSS

                    # minimising the KPSS function (maximising the stationarity)
                    res = spop.minimize(KPSS, b0, method='Nelder-Mead')
                    KPSS_opt = res.fun
                    # retrieving optimal parameters
                    b_opt = float(res.x)
                    a_opt = np.average(np.array(raw_data[tickers[1]]) - np.array(b_opt * raw_data[tickers[0]]))
                else:
                    # defining the KPSS function (two-parameter)
                    def KPSS2(kpss_params):
                        a = kpss_params[0]
                        b = kpss_params[1]
                        resid = np.array(raw_data[tickers[1]] - (a + b * raw_data[tickers[0]]))
                        cum_resid = np.cumsum(resid)
                        st_error = (np.sum(resid ** 2) / (len(resid) - 2)) ** (1 / 2)
                        KPSS = np.sum(cum_resid ** 2) / (len(resid) ** 2 * st_error ** 2)
                        return KPSS

                    # minimising the KPSS function (maximising the stationarity)
                    res = spop.minimize(KPSS2, [a0, b0], method='Nelder-Mead')
                    # retrieving optimal parameters
                    KPSS_opt = res.fun
                    a_opt = res.x[0]
                    b_opt = res.x[1]

                    # trading rules
                # if we have investment position

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
                    if total_return <= -0.02:
                        self.Liquidate(self.symbol_list[i][0])
                        self.Liquidate(self.symbol_list[i][1])
                        self.selected_pairs[i].position = 0
                        self.Log("Cutting off losses")
                if np.sign(history_B.iloc[-1] - (a_opt + b_opt * history_A.iloc[-1])) == self.selected_pairs[
                    i].position:
                    # self.selected_pairs[i].position = old_signal
                    self.selected_pairs[i].position = self.selected_pairs[i].position



                else:
                    # only trade if the pair is cointegrated
                    if KPSS_opt > KPSS_max:
                        self.Liquidate(self.symbol_list[i][0])
                        self.Liquidate(self.symbol_list[i][1])

                        self.selected_pairs[i].position = 0
                        # only trade if there are large enough profit opportunities (optimal entry)
                        # elif abs(history_B.iloc[-1]/(a_opt + b_opt*history_A.iloc[-1])-1) < 0.02:

                        self.Liquidate(self.symbol_list[i][0])
                        self.Liquidate(self.symbol_list[i][1])

                        self.selected_pairs[i].position = 0
                    else:
                        self.selected_pairs[i].position = np.sign(
                            history_B.iloc[-1] - (a_opt + b_opt * history_A.iloc[-1]))

                        if self.selected_pairs[i].position == 1:
                            if not self.Portfolio.Invested:
                                self.SetHoldings(self.symbol_list[i][0],
                                                 (-self.selected_pairs[i].ou_beta) / len(self.selected_pairs))
                                self.SetHoldings(self.symbol_list[i][1],
                                                 (1 - self.selected_pairs[i].ou_beta) / len(self.selected_pairs))
                        elif self.selected_pairs[i].position == -1:
                            if not self.Portfolio.Invested:
                                self.SetHoldings(self.symbol_list[i][0],
                                                 (self.selected_pairs[i].ou_beta) / len(self.selected_pairs))
                                self.SetHoldings(self.symbol_list[i][1],
                                                 (self.selected_pairs[i].ou_beta - 1) / len(self.selected_pairs))



            except Exception as e:
                self.Log(e)


