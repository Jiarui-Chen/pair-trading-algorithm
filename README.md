# Paris Trading Algorithm

## Project Info
* Educational Institution: University of Rochester
* Course: DSCC 383 - Data Science Capstone
* Project Sponsor: FLX AI
* Project Contributors:
	* Qihang Tang, University of Rochester (Spread Modeling, OU Algorithm)
	* Jiarui Chen, University of Rochester (Pair Selection, Bollinger Algorithm)
	* Rong Fan, University of Rochester (Copula Algorithm)
	* Xuanyu Shen, University of Rochester (Cointegration Algorithm)

## Abstract
The objective of this project is to build a pair trading algorithm with the following three component:
* Pairs Selection
* Spread Modeling
* Trading Strategy and Execution

The success of the project is measured by the annual return of the algorithm. Over the course of the project, we aimed to outperform the SPY ETF Index Fund.  

## How to Run the Code?
1. All code must be run on QuantConnect, an online platform for researching, backtesting, and live trading.
2. To run any algorithm, there must be a project built on QuantConnect, with all the Python files for the algorithm within the same project. Please do not change the file names for the main.py files. The file name must be main.py in order to be recognized as an algorithm trigger on QuantConnect.
3.	A backtest node is needed for backtesting and a live trade node is needed for live trading. If there are any time-out errors (such as exceeding the 5 minutes constrain in the backtesting node), a better node with stronger computation power is needed. This may especially be the case for the OU-based implementation.
4.	For a detailed explanation of what each method does in each Python file, please read the comment around the actual codes of methods.
5.	Here is a link to an introduction to using QuantConnect: https://www.quantconnect.com/docs/v2/lean-engine/getting-started

## Code File Usage
With a group of four, each of the contributor explored one of the following trading strategies, and the corresponding code folders are listed in the brackets
* OU Strategy (ou_algo)
* Bollinger Band Strategy (bollinger_algo)
* Copula Strategy (copula_algo)
* Cointegration Strategy (cointegration_algo)

The code files for pair selection are integrated in four folders respectively but are the same across the board. 

There are two additional folders in the code file:
* Others: the folder that contains the Jupiter notebook files that are used for preliminary testing. These codes are not included in the actual implementation of the algorithm and should not be used as a reference.
* Data: this folder that contains the sample stock price data. On QuantConnect, the algorithms will directly use data provided by the platform with specific codes, and therefore no data is required for running the code. 

## Additional Details
For additional details on the project, please refer to the "Project Report.pdf" file included in the repo. 




