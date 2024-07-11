#region imports
from AlgorithmImports import *

import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from hurst import compute_Hc
from datetime import datetime
from pair import *
#endregion

warnings.filterwarnings("ignore")

# retrieve sp500 tickers
# TICKERS = ['ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALGN', 'ALLE', 'ALL', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'AOS', 'APA', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ANET', 'AJG', 'AIZ', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AVGO', 'BKR', 'BAC', 'BK', 'BAX', 'BDX', 'BRK.B', 'BBY', 'BIO', 'BIIB', 'BLK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'BR', 'BF.B', 'BEN', 'CHRW', 'CDNS', 'CZR', 'CPB', 'COF', 'CAH', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CF', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'COO', 'CPRT', 'CTVA', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'CRM', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'DVN', 'DXCM', 'DLR', 'DFS', 'DISH', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DD', 'DXC', 'DGX', 'DIS', 'ED', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ENPH', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'FANG', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FRC', 'FISV', 'FLT', 'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'FCX', 'GOOGL', 'GOOG', 'GLW', 'GPS', 'GRMN', 'GNRC', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HAL', 'HBI', 'HIG', 'HAS', 'HCA', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HWM', 'HPQ', 'HUM', 'HBAN', 'HII', 'IT', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KMX', 'KO', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LNT', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LEG', 'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'LUV', 'MMM', 'MO', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NOV', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'NOW', 'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE', 'ORCL', 'OTIS', 'O', 'PEAK', 'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PENN', 'PNR', 'PEP', 'PKI', 'PRGO', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PTC', 'PEG', 'PSA', 'PHM', 'PVH', 'PWR', 'QRVO', 'QCOM', 'RE', 'RL', 'RJF', 'RTX', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SCHW', 'STZ', 'SJM', 'SPGI', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'SHW', 'SPG', 'SWKS', 'SNA', 'SO', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'T', 'TAP', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UHS', 'UNM', 'VLO', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VTRS', 'V', 'VNO', 'VMC', 'WRB', 'WAB', 'WMT', 'WBA', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WYNN', 'XRAY', 'XOM', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']

class PairSelector(): 
    def __init__(self, history_data): 
    #     # TODO: pass in  QCAlgorithm object here to avoid using qb in __init__ 
    #     # TODO: Combine the start_data and end_date with the property of QCAlgo.
    #     self.universe = universe
    #     self.start_date = start_date
    #     self.end_date = end_date
        self.history = history_data

    # retrieve historical price
    def extract_price_data(self): 
        history = pd.DataFrame(self.history["close"])
        history = history.reset_index()
        # history_close = history.copy() # before calculating return, keep a copy of original closing price for later usage
        return history

    """
    def extract_pe_data(self, symbols):
        # TODO: Get rid of the AddEquity
        qb = QuantBook()
        tickers = [qb.AddEquity(t, dataNormalizationMode=DataNormalizationMode.Adjusted).Symbol for t in symbols]
        history_pe = qb.GetFundamental(tickers, "ValuationRatios.PERatio", datetime(2022, 3, 1), datetime(2022, 12, 31))
        history_pe.drop_duplicates(inplace=True) # only need monthly resolution
        return history_pe
    """

    """
    def calc_pe_change(self, history_pe):
        history_pe = history_pe.pct_change()
        history_pe.dropna(inplace=True, axis=0)
        history_pe = history_pe.T
        return history_pe
    """


    # calculate return using sliding window
    # TODO: Map the pct_change() to the dataframe on a symbol by symbol basis. 
    def calc_return(self, history):
        history['return'] = history.groupby('symbol')['close'].apply(lambda x: x.pct_change())
        history.drop("close", axis=1, inplace=True)   
        return history


    def preprocess_return_data(self, return_df):
        # format dataframe for clustering usage (row: symbol, column: time, value: return)
        first_date = return_df["time"][0] # the return on the first date is always NaN, so drop it
        return_df = return_df.pivot(index="symbol", columns="time", values="return")
        return_df.drop(first_date, axis=1, inplace=True)
        # normalize the return data for dimensionality reduction later
        scaler = MinMaxScaler()
        return_df = pd.DataFrame(scaler.fit_transform(return_df), columns=return_df.columns, index=return_df.index)
        return return_df


    # regular pca for dimensionality reduction
    def dimensionality_reduction(self, data, threshold):
        pca = PCA(random_state=383)
        data.fillna(0, inplace=True) # pca does not take null value, fill NA with 0 return.
        data_transformed = pd.DataFrame(pca.fit_transform(data), index=data.index)
        try: # when threshold is high, np.argwhere return is in shape [0, 1]
            n_components_retained = np.argwhere(pca.explained_variance_ratio_.cumsum()>threshold)[0][0] # Determine number of component
        except IndexError as e:
            print("[IndexError]: Please lower the EXPLAINED VARIANCE THRESHOLD.")
            return None
        # print(f"Kept {n_components_retained} components.")
        data_reduced = data_transformed.iloc[:,:n_components_retained].copy()
        return (data_reduced, pca)


    # pca visualization
    def visualize_pca(self, model):
        exp_var_pca = model.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
        plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


    def optics_clustering(self, data_reduced, *args, **kwargs):
        clustering = OPTICS(*args, **kwargs)
        clustering.fit(data_reduced)
        labels = clustering.labels_
        # append result back to the dataframe for interpretation
        data_clustered = data_reduced.copy()
        data_clustered["cluster"] = labels
        return (data_clustered, clustering)


    def form_candidate_pair(self, data_clustered, clustering_model):
        num_cluster = len(np.unique(clustering_model.labels_)) - 1 # number of clusters
        pair_pool = []
        for i in range(num_cluster): # iterate through each cluster
            cluster = data_clustered[data_clustered["cluster"] == i]
            stock_list = cluster.index.tolist() # get the list of stock symbols wtihin a cluster
            # start to form pairs:
            for j in range(len(stock_list)-1):
                for k in range(j+1, len(stock_list)):
                    pair_pool.append([stock_list[j], stock_list[k]])
        return pair_pool


    def cointegration_test(self, pair, data):
        price1 = data[data["symbol"] == pair[0]]["close"]
        price2 = data[data["symbol"] == pair[1]]["close"]
        eg_test = sm.tsa.stattools.coint(price1, price2)
        if eg_test[1] <= 0.1: # p-value. The threshold could be adjusted
            return True
        else:
            return False


    def hurst_exponent(self, pair, data):
        price1 = data[data["symbol"] == pair[0]]["close"].values.reshape(-1, 1)
        price2 = data[data["symbol"] == pair[1]]["close"].values.reshape(-1, 1)
        model = LinearRegression(fit_intercept = True)
        model.fit(price2, price1)
        beta = model.coef_[0][0]
        spread = price1 - beta * price2 - model.intercept_
        h = compute_Hc(spread, kind='price', simplified=False)[0]
        if h <= 0.5:
            return True
        else:
            return False


    def half_life(self, pair, data):
        price1 = data[data["symbol"] == pair[0]]["close"].values.reshape(-1, 1)
        price2 = data[data["symbol"] == pair[1]]["close"].values.reshape(-1, 1)
        model = LinearRegression(fit_intercept = True)
        model.fit(price2, price1)
        beta = model.coef_[0][0]
        spread = pd.DataFrame(price1 - beta * price2 - model.intercept_, columns=["spread"])
        # half time calculation reference: https://flare9xblog.wordpress.com/2017/09/27/half-life-of-mean-reversion-ornstein-uhlenbeck-formula-for-mean-reverting-process/
        # lag the spread by -1
        spread["lag"] = spread["spread"].shift(1)
        spread.dropna(axis=0, inplace=True) # adjust the length to the same for spread and spread_lag
        spread["diff"] = spread["spread"] - spread["lag"]
        spread["lag_mean_diff"] = spread["lag"] - spread["lag"].mean()
        # linear regression
        model = LinearRegression(fit_intercept=True)
        model.fit(spread["lag_mean_diff"].values.reshape(-1, 1), spread["diff"].values.reshape(-1, 1))
        coef = model.coef_[0][0]
        half_time = -math.log(2) / coef
        if 1 < half_time < 252: # number of trade day in a year is 252
            return True
        else:
            return False


    def mean_cross(self, pair, data):
        price1 = data[data["symbol"] == pair[0]]["close"].values.reshape(-1, 1)
        price2 = data[data["symbol"] == pair[1]]["close"].values.reshape(-1, 1)
        model = LinearRegression(fit_intercept = True)
        model.fit(price2, price1)
        beta = model.coef_[0][0]
        spread = pd.DataFrame(price1 - beta * price2 - model.intercept_, columns=["spread"])
        spread["above_mean"] = (spread["spread"] > spread["spread"].mean()) + 0
        spread["above_mean_lag"] = spread["above_mean"].shift(1)
        spread.dropna(axis=0, inplace=True)
        spread["cross"] = (np.logical_xor(spread["above_mean"],spread["above_mean_lag"])) + 0
        if spread["cross"].sum() >= 12:
            return True
        else:
            return False


    def apply_selection_rules(self, pair_pool, price_data):
        final_pair = []
        for pair in pair_pool:
            if self.cointegration_test(pair, price_data) and self.hurst_exponent(pair, price_data) and self.half_life(pair, price_data) and self.mean_cross(pair, price_data):
                p = Pair(pair[0], pair[1], price_data)
                final_pair.append(p)
        return final_pair


    def select_pairs(self):
        price_df = self.extract_price_data()
        original_data_source = price_df.copy()
        return_df = self.calc_return(price_df)
        return_df_processed = self.preprocess_return_data(return_df)
        # pe_df = self.extract_pe_data(self.universe)
        # pe_df = self.calc_pe_change(pe_df)
        # feature_df = return_df_processed.join(pe_df, lsuffix="price", rsuffix="pe")
        # feature_df.columns = feature_df.columns.astype(str)
        # feature_df.dropna(inplace=True, axis=0)
        return_reduced, pca_model = self.dimensionality_reduction(return_df_processed, 0.8)
        stock_clustered, optics_model = self.optics_clustering(return_reduced, min_samples = 3)
        pair_pool = self.form_candidate_pair(stock_clustered, optics_model)
        final_pairs = self.apply_selection_rules(pair_pool, original_data_source)
        return final_pairs