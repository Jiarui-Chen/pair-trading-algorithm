# region imports
from AlgorithmImports import *
from pair_selector import *
from copula import *
from ou_process import *
from datetime import datetime
from scipy.stats import kendalltau
from scipy.integrate import quad
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
# endregion

TICKERS = ['ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALGN', 'ALLE', 'ALL', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'AOS', 'APA', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ANET', 'AJG', 'AIZ', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AVGO', 'BKR', 'BAC', 'BK', 'BAX', 'BDX', 'BRK.B', 'BBY', 'BIO', 'BIIB', 'BLK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'BR', 'BF.B', 'BEN', 'CHRW', 'CDNS', 'CZR', 'CPB', 'COF', 'CAH', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CF', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'COO', 'CPRT', 'CTVA', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'CRM', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'DVN', 'DXCM', 'DLR', 'DFS', 'DISH', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DD', 'DXC', 'DGX', 'DIS', 'ED', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ENPH', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'FANG', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FRC', 'FISV', 'FLT', 'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'FCX', 'GOOGL', 'GOOG', 'GLW', 'GPS', 'GRMN', 'GNRC', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HAL', 'HBI', 'HIG', 'HAS', 'HCA', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HWM', 'HPQ', 'HUM', 'HBAN', 'HII', 'IT', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KMX', 'KO', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LNT', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LEG', 'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'LUV', 'MMM', 'MO', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NOV', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'NOW', 'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE', 'ORCL', 'OTIS', 'O', 'PEAK', 'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PENN', 'PNR', 'PEP', 'PKI', 'PRGO', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PTC', 'PEG', 'PSA', 'PHM', 'PVH', 'PWR', 'QRVO', 'QCOM', 'RE', 'RL', 'RJF', 'RTX', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SCHW', 'STZ', 'SJM', 'SPGI', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'SHW', 'SPG', 'SWKS', 'SNA', 'SO', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'T', 'TAP', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UHS', 'UNM', 'VLO', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VTRS', 'V', 'VNO', 'VMC', 'WRB', 'WAB', 'WMT', 'WBA', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WYNN', 'XRAY', 'XOM', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']

class CryingFluorescentYellowRhinoceros(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2021, 5, 1)  # Set Start Date
        self.SetEndDate(2022, 12, 31) # Set End Date
        self.SetCash(100000)  # Set Strategy Cash
        self.DataNormalizationMode = "Adjusted"
        self.pair_selector = PairSelector(TICKERS, datetime(2020, 7, 1), datetime(2020, 12, 31))
        self.selected_pairs = self.pair_selector.select_pairs()
        self.symbol_list = []
        for p in self.selected_pairs:
            symbol1 = self.AddEquity(p.A, Resolution.Hour, dataNormalizationMode=DataNormalizationMode.Adjusted).Symbol
            symbol2 = self.AddEquity(p.B, Resolution.Hour, dataNormalizationMode=DataNormalizationMode.Adjusted).Symbol
            self.Log(symbol1)
            self.Log(symbol2)
            self.symbol_list.append([symbol1, symbol2])

        

    def OnData(self, data: Slice):
        
        for i in range(len(self.symbol_list)):
            try:
                history_A = pd.Series(self.History(self.symbol_list[i][0], TimeSpan.FromDays(250), Resolution.Daily)["close"].values).to_list() 
                history_B = pd.Series(self.History(self.symbol_list[i][1], TimeSpan.FromDays(250), Resolution.Daily)["close"].values).to_list() #Extract the price data for the past 250 days
                return_A = []
                return_B = []
                for j in range(1, len(history_A)): 
                    return_A.append((history_A[j] / history_A[j-1]) - 1)
                    return_B.append((history_B[j] / history_B[j-1]) - 1) #Calculate the return
                return_A.pop(0)
                return_B.pop(0) #Remove the first one since it's a null
                ecdf_u = ECDF(return_A)
                ecdf_v = ECDF(return_B) 
                u = ecdf_u(return_A[-1])
                v = ecdf_v(return_B[-1]) #Create the quantile function and generate the corresponding values for the returns
                tau = kendalltau(return_A, return_B)[0]

                theta_clayton = 2 * tau / (1 - tau) #clayton theta

                theta_gumbel = 1 / (1 - tau) #gumbel theta

                integrand = lambda t: t / (np.exp(t) - 1)  # generate the integrand
                frank_fun = lambda theta_frank: ((tau - 1) / 4.0  - (quad(integrand, sys.float_info.epsilon, theta_frank)[0] / theta_frank - 1) / theta_frank) ** 2
                theta_frank = minimize(frank_fun, 4, method='BFGS', tol=1e-5).x #frank theta

                #utov = ((np.exp(-theta_frank * u) - 1) * (np.exp(-theta_frank * v) - 1) + (np.exp(-theta_frank * u) - 1)) / ((np.exp(-theta_frank * u) - 1) * (np.exp(-theta_frank * v) - 1) + (np.exp(-theta_frank) - 1))
                #vtou = ((np.exp(-theta_frank * u) - 1) * (np.exp(-theta_frank * v) - 1) + (np.exp(-theta_frank * v) - 1)) / ((np.exp(-theta_frank * u) - 1) * (np.exp(-theta_frank * v) - 1) + (np.exp(-theta_frank) - 1)) #frank function
                
                utov = u ** (-theta_clayton - 1) * (u ** (-theta_clayton) +  v ** (-theta_clayton) - 1) ** (-1 / theta_clayton - 1)
                vtou = v ** (-theta_clayton - 1) * (u ** (-theta_clayton) +  v ** (-theta_clayton) - 1) ** (-1 / theta_clayton - 1) #clayton function


                #A = (-np.log(u)) ** theta_gumbel + (-np.log(v)) ** theta_gumbel
                #c = np.exp(-A ** (1 / theta_gumbel))
                #utov = c * ((-np.log(u)) ** theta_gumbel + (-np.log(v)) ** theta_gumbel) ** ((1 - theta_gumbel) / theta_gumbel) * (-np.log(v)) ** (theta_gumbel - 1) * (1 / v)
                #vtou = c * ((-np.log(u)) ** theta_gumbel + (-np.log(v)) ** theta_gumbel) ** ((1 - theta_gumbel) / theta_gumbel) * (-np.log(u)) ** (theta_gumbel - 1) * (1 / u) #gumbel function

                conditional_probs = [utov, vtou] #Use the function you want and comment the other two

                
            # apply the trading logic
                self.Log(self.selected_pairs[i].position)
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
                        self.Log("Cutting off losses")                              #Loss control algorithm: when the unrealized profit is under the threshold, liquidate all the holdings
                if conditional_probs[0] > 0.95 and conditional_probs[1] < 0.05:
                    if not self.Portfolio.Invested:
                        self.selected_pairs[i].position = -1
                        self.SetHoldings(self.symbol_list[i][0], (self.selected_pairs[i].ou_beta - 1)/len(self.selected_pairs))
                        self.SetHoldings(self.symbol_list[i][1], (self.selected_pairs[i].ou_beta)/len(self.selected_pairs))  #Short the spread
                elif conditional_probs[0] < 0.05 and conditional_probs[1] > 0.95:
                    if not self.Portfolio.Invested:
                        self.selected_pairs[i].position = 1
                        self.SetHoldings(self.symbol_list[i][0], (1 - self.selected_pairs[i].ou_beta)/len(self.selected_pairs))
                        self.SetHoldings(self.symbol_list[i][1], (-self.selected_pairs[i].ou_beta)/len(self.selected_pairs))   #Long the spread
                elif conditional_probs[0] > 0.5 and conditional_probs[1] > 0.5 and self.selected_pairs[i].position == -1:
                    self.Liquidate(self.symbol_list[i][0])
                    self.Liquidate(self.symbol_list[i][1])
                    self.selected_pairs[i].position = 0
                elif conditional_probs[0] > 0.5 and conditional_probs[1] > 0.5 and self.selected_pairs[i].position == 1:
                    self.Liquidate(self.symbol_list[i][0])
                    self.Liquidate(self.symbol_list[i][1])
                    self.selected_pairs[i].position = 0                #Quit the position if both have crossed the mean
            except Exception as e:
                self.Log(e)
