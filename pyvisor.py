'''
Driver
'''
import SymbolFetcher as fetcher
import TechnicalCalculator as tc
import StockChart
import sys
import pandas as pd
import datetime as dt
import os
import time
import StrategyTrainer
import matplotlib.pyplot as plt
def main():
	t = time.time()
	print (sys.argv)
	date = str(dt.date.today())
	symbol = sys.argv[1]
	# data = gather(symbol,window= sys.argv[2])
	# data.to_csv('../Data/{}_{}.csv'.format(symbol,date))
	# # StockChart.plot_dash(symbol,data,40)
	# # os.system('open Charts/{}_{}.pdf'.format(symbol,date))
	# # t = time.time() - t
	# # print('Fetch and Dash Creation took {} seconds.'.format(str(t)))
	# StockChart.plot_charts(symbol,data,40)
	run(symbol, 40)



def run(symbol,window=40):
	date = str(dt.date.today())

	# check for cached data
	if not is_chached(symbol,date):
		try:
			print('Trying to gather data for {}'.format(symbol))
			data = gather(symbol,window)
		except Exception as e:
			print('Error gathering {}'.format(symbol))
			return False

		data.to_csv('../Data/{}_{}.csv'.format(symbol,date))
		StockChart.plot_charts(symbol,data,40)
		print('Done gathering data for {}'.format(symbol))

	return date

def update(symbol,window=40):
	date = str(dt.date.today())
	try:
		print('Trying to gather data for {}'.format(symbol))
		data = gather(symbol,window)
	except Exception as e:
		print('Error gathering {}'.format(symbol))
		return False

	data.to_csv('../Data/{}_{}.csv'.format(symbol,date))
	StockChart.plot_charts(symbol,data,40)
	print('Done gathering data for {}'.format(symbol))
	return date




def gather(symbol,window):
	f = fetcher.SymbolFetcher()
	data = f.fetch_pricing(symbol)
	data['daily_rtrn'] = tc.DailyReturn(data)
	vol = tc.Volatility(data,window=10)
	sma5 = tc.SMA(data,5)
	sma10 = tc.SMA(data,10)
	sma20 = tc.SMA(data,20)
	rsi5 = tc.RSI(data,window=5)
	rsi10 = tc.RSI(data,window=10)
	rsi20 = tc.RSI(data,window=20)
	mom = tc.Momentum(data,window=10)
	macd =  f.fetch_indicator(symbol,'MACD')
	aroon = f.fetch_indicator(symbol,'AROON')
	stoch = f.fetch_indicator(symbol,'STOCH')
	cci = tc.CCI(data,window=10)
	bands = tc.BBands(data,window=20)

	data['Volatility'] = vol
	data['RSI_5'] = rsi5
	data['RSI_10'] = rsi10
	data['RSI_20'] = rsi20
	data['Momentum'] = mom

	# MACD has fewer indicies, so it must be flipped back to original order
	# then appended to data--in the original order--so that the most recent
	# dates line up
	data.index = data.index[::-1]
	data.sort_index(inplace=True)
	macd.index = macd.index[::-1]
	macd.sort_index(inplace=True)
	data['MACD'] = macd['MACD']
	data['MACD_Hist'] = macd['MACD_Hist']
	data['MACD_Signal'] = macd['MACD_Signal']
	if not aroon.shape[0] == 0:
		data['AROON_UP'] = aroon.iloc[:,2]
		data['AROON_DOWN'] = aroon.iloc[:,1]
	# if not adx.shape[0] == 0:
	# 	data['ADX'] = adx.iloc[:,1]
	# Now, return the data order back to oldest first to newest last for both
	# data and macd data
	data.index = data.index[::-1]
	data.sort_index(inplace=True)
	macd.index = macd.index[::-1]
	macd.sort_index(inplace=True)
	data['STOCHD'] = stoch['SlowD']
	data['STOCHK'] = stoch['SlowK']
	data['SMA_5'] = sma5
	data['SMA_10'] = sma10
	data['SMA_20'] = sma20
	data['CCI'] = cci
	data['BBand_Up'] = bands['UPPER_BAND']
	data['BBand_Mid'] = bands['MID_BAND']
	data['BBand_Lower'] = bands['LOWER_BAND']
	ret = data['daily_rtrn']
	data.drop(labels=['daily_rtrn'], axis=1,inplace = True)
	data['daily_rtrn'] = ret
	return data

def get_data(symbol,window=40):
	date = str(dt.date.today())
	if not is_chached(symbol,date):
		print('Fetching new data.')
		data = gather(symbol,window)
	else:
		print('Fetching cached data.')
		data = pd.read_csv('../Data/{}_{}.csv'.format(symbol,date))

	data.to_csv('../Data/{}_{}.csv'.format(symbol,date))
	
	return data
	
def is_chached(symbol,date):
	return os.path.isfile('../Data/{}_{}.csv'.format(symbol,date))

def train_agent(symbol,window,max_epoch=100):
	data = gather(symbol,window).dropna()
	# for col in ['open', 'high','low','SMA_5','SMA_10','SMA_20','BBand_Up',
	#'BBand_Mid','BBand_Lower']:
	# 	data[col] = data[col]/ data['close']
	# data.drop(labels=['SMA_5','SMA_10','SMA_20','MACD','MACD_Signal',
	#'MACD_Hist'], axis=1,inplace = True)

	data = data[['RSI_5','CCI','Momentum','close','STOCHD','STOCHK',
	'Volatility','daily_rtrn']]

	training_data = data.iloc[:-100,:]
	testing_data = data.iloc[-100:,:]

	strat,epoch=StrategyTrainer.train(training_data, testing_data, 
		eps = .50, max_epoch=max_epoch)
	 
	print('It took {} of {} possible epochs to train the agent.'.format(epoch,
		max_epoch))

	# plt.plot(returns)
	# plt.show()
	return strat


if __name__ == '__main__':
	main()