import QLearner
import pyvisor
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np


test_window = -60

def backtest(testing_data,returns,close):
	close = close.iloc[test_window:]
	close.index = range(0,close.shape[0])
	s = testing_data[0,:]
	total = []
	port_val = []
	cash_val = []
	position = 0
	cash = close.iloc[0]
	port = 0
	orders = []
	for state in range(0,testing_data.shape[0]-1):
		a = learner.query(s,opt=True)
		orders.append(a)
		if a == 1:
			if position == 0:
				port = close.iloc[state]
				cash -= close.iloc[state]
				position = 1
			if position == 1:
				port = close.iloc[state]
		elif a == 0:
			if position == 1:
				port -= close.iloc[state]
				cash += close.iloc[state]
				position = 0
			if position == 0:
				port = 0

		
		sprime = testing_data[(state + 1),:]
		sprime[-2] = a
		s = sprime
		total.append((cash + port))
		port_val.append(port)
		cash_val.append(cash)

	total = np.array(total)
	port = np.array(port_val)
	cash = np.array(cash_val)

	actions = orders
	plt.plot((port),label='Port')
	plt.plot((cash),label='Cash')
	plt.legend(loc='best')
	plt.show()

	for x in range(0,len(actions)):
		if actions[x] == 1:
			plt.axvspan(x,(x+1),facecolor='g',alpha=.1)
			# plt.axvline(x,color='g')
		if actions[x] == 0:
			pass
			# plt.axvline(x,color='r')
	plt.plot((close),label='Market')
	plt.plot((total),label='Value')
	plt.legend(loc='best')
	plt.show()



data = pyvisor.get_data('MSFT',40)
data = data.dropna()
data = data[['RSI_10','CCI','Momentum','close','STOCHD','STOCHK',
'Volatility','daily_rtrn']]
returns = data['daily_rtrn']
close = data['close']
data = data[['RSI_10','CCI','Momentum','Volatility']]
data['position'] = 0

training_data = data.iloc[:test_window,:]
testing_data = data.iloc[test_window:,:]
bin_size = 10
encoder = KBinsDiscretizer(n_bins = bin_size, encode='ordinal')
encoder.fit(training_data.values)
training_data = encoder.transform(training_data).astype(int)
testing_data = encoder.transform(testing_data).astype(int)

actions = 2
dims = (bin_size,bin_size,bin_size,bin_size,2,actions)
alpha = .01
gamma = 0.0
eps = .9
eps_end = .005
eps_decay = 'linear'
decay_period = 500
decay_rate = .99
dyna = 300
epochs = 10

learner = QLearner.QLearner(dims,actions,alpha,gamma,eps,eps_end,
	eps_decay,decay_period,decay_rate,dyna)

for epoch in range(0,epochs):
	s = training_data[0,:]
	print('Epoch:',epoch)
	for state in range(0,training_data.shape[0] - 1):
		a = learner.query(s)
		if a == 0:
			mult = -1
		else:
			mult = a
		r = mult * returns.iloc[(state + 1)]
		sprime = training_data[(state+1),:]
		sprime[-1] = a
		learner.learn(s,a,r,sprime)
		s = sprime

backtest(testing_data,returns,close)





