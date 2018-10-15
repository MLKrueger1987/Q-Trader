import numpy as np
import math
import random
class QLearner(object):
	"""docstring for QLearner"""


	def __init__(self,dims,actions,alpha=.1,gamma=.8,eps=.9,eps_end=.05,
		eps_decay='linear',decay_period=100, decay_rate =.998,dyna=0):
		super(QLearner, self).__init__()
		self._init_table(dims)
		self._set_decay(eps_decay,decay_period,eps,eps_end,decay_rate)
		self.alpha = alpha
		self.gamma = gamma
		self.actions = actions
		self.dyna = dyna
		if dyna > 0:
			self.model = dict() 

	def query(self,s,opt=False):
		self.eps = self.decay(self.eps)
		if np.random.random() < self.eps and not opt:
			return np.random.randint(self.actions)
		else:
			s = tuple(s.tolist())
			# if self.model.has_key(s):
			# 	a = self.max_model(s)
			# else:
			a = self.q[s].argmax()
			# print('a:', a)
			return a

	def learn(self,s,a,r,sprime):
		alpha = self.alpha
		gamma = self.gamma
		s = s.tolist()
		s.append(a)
		s = tuple(s)
		sprime = tuple(sprime.tolist())
		q = self.q[s]
		q = (1-alpha * q) + (alpha) * (r + (gamma * 
			self.q[sprime].max())) 
		self.q[s] = q

		if self.dyna > 0:
			s = list(s)
			a = s.pop()
			s = tuple(s)
			if self.model.has_key(s):
				if self.model[s].has_key(a):
					self.model[s][a]= {'sprime':sprime,'r':r}
				else:
					self.model[s] = { a : {'sprime':sprime,'r':r}}
			else:
				self.model[s] = { a : {'sprime':sprime,'r':r}}

			samples = 0

			if self.dyna < len(self.model.keys()):
				samples = self.dyna
			else:
				samples = len(self.model.keys())
			for xp in random.sample(self.model.keys(),samples):
				self.dyna_update(xp,self.model[xp])

	def dyna_update(self,xp,xp_tuple):
		alpha = self.alpha
		gamma = self.gamma
		s = xp
		a = random.sample(self.model[s].keys(),1)[0]
		s = list()
		s.append(a)
		s = tuple(s)
		sprime = xp_tuple[a]['sprime']
		r = xp_tuple[a]['r']
		q = self.q[s]
		q = (1-alpha * q) + (alpha) * (r + (gamma * 
			self.q[sprime].max())) 
		self.q[s] = q

	def _init_table(self, dims):
		self.q = np.random.random(dims) / 1000

	def _set_decay(self,eps_decay,decay_period,eps,eps_end,decay_rate):
		self.eps = eps
		self.eps_end = eps_end
		self.decay_period = decay_period
		self.decay_rate = decay_rate

		if eps_decay == 'linear':
			self.rate = (eps - eps_end) / decay_period
			self.decay =  lambda x: x - self.rate if x > self.eps_end else x
		elif eps_decay == 'expo':
			self.decay = lambda x: self.eps_end + (self.eps - self.eps_end) * math.exp(-1. * x / self.decay_period)
		
		elif eps_decay == 'mult':
			self.decay = lambda x: x * self.decay_rate


		return self.decay

