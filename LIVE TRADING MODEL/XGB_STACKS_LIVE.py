import time
import warnings
from datetime import datetime as dt
import joblib
import numpy as np
import xgboost
import requests
import json
from importlib import reload
from collections import deque
import fxcm_rest_api_token as fxcm_rest_api
result = reload(fxcm_rest_api)

SYMBOL = "GBPJPY"
SYMBOL_FACTOR = 100
STOPLOSS = 15 #in pips
FEATURES = 69
BOOKKEEPING = f'{SYMBOL}_smart_diff_80pct_stack_V2_trail'
SYMBOL_DICT = {'GBPJPY': 'GBP/JPY', 'EURUSD': 'EUR/USD', 'AUDUSD': 'AUD/USD', "GBPUSD": "GBP/USD"}
LOTS = 1 #Equal to 100,000 units
MAX_SPREAD = 5
SPREAD_MEAN_SIZE = 30
MIN_TP = 15
TIME = dt.now().strftime("%Y%m%d_%H%M%S")
SEQ_LEN = 3
TRAIL = 20

xgb_filename = "models\\ID0506_1930_n_est1500_GJ_Smart_Diff_80pct_STACK_2015-19.joblib_cv.dat"
filename = 'C:\\Users\\Michael\\AppData\\Roaming\\MetaQuotes\\Terminal\\3B1F27ECABFCD8AC8EBD6C5EE76ACF30\\MQL4\\Files\\GBPJPY_Smart_Signals.csv'

class forex_env():

	def __init__(self):
		self.order_price = 0
		self.close = 0
		self.orders = ""
		self.ordertype = True
		self.accountID = 0
		self.tradeID = ""
		self.prev_data = np.empty((1,FEATURES))
		self.win_loss = 0
		self.total_win_loss = 0
		self.take_profit = 0
		self.data_deque = deque(maxlen=SEQ_LEN)
		self.full_stack = False
		self.row_stack = np.array([1,222])
		self.trailing = False
		self.trailstop = 0
		self.trailorder = False


	def get_data(self, filename):
        
		wait_for_data = True
		#with warnings.catch_warnings():
			#warnings.simplefilter("ignore", UserWarning)
		while wait_for_data:
				try:
					data = np.genfromtxt(filename, delimiter=',')
						#print(data[3])
					equal_arrays = np.array_equal(self.prev_data, data)
						#print(f'DATA: {data}')
						#print(f'OLD DATA: {self.prev_data}')
					if not equal_arrays and data.shape[0] == FEATURES:
							#print('88')
							#wait_for_data = False
						self.preprocess(data)
						self.prev_data = data
						if self.full_stack:
							return self.row_stack
					elif self.orders != "":
						self.get_tick()
						time.sleep(5)
					else:
						time.sleep(5)
							#print('77')
				except:
					pass
					#print('ERROR PASS!')

	def init_data(self, filename):
		data = np.genfromtxt(filename, delimiter=',')
		while data.shape[0] != FEATURES:
			data = np.genfromtxt(filename, delimiter=',')
			print('No data')
		self.prev_data = data
		self.preprocess(data)
	

	def preprocess(self, df):
		prep_data = np.delete(df,[0,1,2,9,18,19,46,47,66,67,68]) #Adjust time
		
		trend = np.array(self.trend(df)).reshape(1,-1)
		data = np.empty([8])
		data[0] = (df[0] - self.prev_data[0])*SYMBOL_FACTOR #close_diff
		data[1] = (df[1] - df[2])*SYMBOL_FACTOR #high_low_diff
		data[2] = (df[9] - self.prev_data[9])*SYMBOL_FACTOR #ma5_trend
		data[3] = (df[18] - self.prev_data[18])*SYMBOL_FACTOR #ma15_trend
		data[4] = (df[0] - df[18]) #ma15_dist
		data[5] = (df[0] - df[46]) #ma60_dist
		data[6] = (df[0] - df[19]) #ma50_15_dist
		data[7] = (df[0] - df[47]) #ma50_60_dist
		
		pivot_df = np.expand_dims(self.pivot(df), axis=0)
		data = np.expand_dims(data, axis=0)
		data2 = np.hstack((np.expand_dims(prep_data, axis=0), data, pivot_df, trend))
		chl = np.expand_dims(df[:3], axis=0)
		save_data = np.concatenate([chl, data2], axis=1)

		with open(f"data/GJ_XGB_smart_diff_stack_0506_V1_trail_{TIME}.csv", "a") as f:
			np.savetxt(f, save_data, fmt="%s", newline=' ', delimiter=',')
			f.write("\n")
		self.stack(data2)

	def trend(self, array):
		if array[18] > array[19]:
			return 1
		else:
			return 0

	def stack(self, array):
		self.data_deque.append(array)
		if len(self.data_deque) == SEQ_LEN:
			self.row_stack = np.stack(self.data_deque, axis=1).reshape(1,-1)
			self.full_stack = True
			print('STACK READY!')
        		

	def pivot(self, df):

		high = df[-2]
		low = df[-1]

		p = np.sum(df[-3:])/3
		r1 = np.subtract((2 * p), low)
		r2 = np.subtract(np.add(p, high), low)
		r3 = np.add(high, 2*(np.subtract(p, low)))
		s1 = np.subtract((2 * p), high)
		s2 = np.add(np.subtract(p, high), low)
		s3 = np.subtract(low, 2*np.subtract(high, p))

		p_dist = (df[0] - p)*SYMBOL_FACTOR
		r1_dist = (df[0] - r1)*SYMBOL_FACTOR
		r2_dist = (df[0] - r2)*SYMBOL_FACTOR
		r3_dist = (df[0] - r3)*SYMBOL_FACTOR
		s1_dist = (df[0] - s1)*SYMBOL_FACTOR
		s2_dist = (df[0] - s2)*SYMBOL_FACTOR
		s3_dist = (df[0] - s3)*SYMBOL_FACTOR

		return np.hstack((p_dist,r1_dist,r2_dist,r3_dist,s1_dist,s2_dist,s3_dist))
   

	def to_datetime(self):
		datetime_object = dt(
							year=2020,
							month=int(self.row_stack[0,0]),
							day=int(self.row_stack[0,1]),
							hour=int(self.row_stack[0,2]),
							minute=int(self.row_stack[0,3]))
		return datetime_object

	def get_tick(self):
		bid, ask, spread = self.bid_ask_spread()

		if self.trailing:
			print(f'BID: {bid:.2f} ASK: {ask:.2f} SPREAD: {spread:.2f}  TRAILING STOP AT: {self.trailstop:.4f}')
		else:
			print(f'BID: {bid:.2f} ASK: {ask:.2f} SPREAD: {spread:.2f}')

		if self.orders == 'long' and self.take_profit < bid:
			self.to_market(3)
		elif self.orders == 'short' and self.take_profit > ask:
			self.to_market(3)

		elif self.orders == 'long' and self.order_price - (STOPLOSS/SYMBOL_FACTOR) > bid:
			print(f'LONG STOPLOSS AT {bid}')
			self.to_market(3)
		elif self.orders == 'short' and self.order_price + (STOPLOSS/SYMBOL_FACTOR) < ask:
			print(f'SHORT STOPLOSS AT {ask}')
			self.to_market(3)
		elif spread > MAX_SPREAD:
			self.spread_loop()

		if self.trailing:
			if self.orders == 'long' and self.trailstop > bid:
				print(f'LONG stop by trail {bid:.4f}')
				#self.trailorder = True
				self.to_market(3)
			elif self.orders == 'short' and self.trailstop < ask:
				#self.trailorder = True
				print(f'SHORT stop by trail {ask:.4f}')
				self.to_market(3)


			elif self.orders == 'long' and self.trailstop < bid - (TRAIL/SYMBOL_FACTOR):
				self.trailstop = bid - (TRAIL/SYMBOL_FACTOR)
				print(f'Long trail moved to {self.trailstop:.4f}')
			elif self.orders == 'short' and self.trailstop > ask + (TRAIL/SYMBOL_FACTOR):
				self.trailstop = ask + (TRAIL/SYMBOL_FACTOR)
				print(f'Short trail moved to {self.trailstop:.4f}')

		if self.orders == 'long' and bid > self.order_price + ((TRAIL)/SYMBOL_FACTOR):
			self.trailing = True
			self.trail_stop = bid - (TRAIL/SYMBOL_FACTOR)
		elif self.orders== 'short' and ask < self.order_price - ((TRAIL)/SYMBOL_FACTOR):
			self.trailing = True
			self.trail_stop = ask + (TRAIL/SYMBOL_FACTOR)
			
		else:
			pass

	def spread_loop(self):
		wide_spread = True
		spread_mean = deque(maxlen=SPREAD_MEAN_SIZE)
		counter = 0
		for i in range(SPREAD_MEAN_SIZE):
			bid, ask, spread = self.bid_ask_spread()
			spread_mean.append(spread)
			time.sleep(2)
		while wide_spread:
			if sum(spread_mean)/SPREAD_MEAN_SIZE > MAX_SPREAD:
				print(f'SPREAD TOO WIDE: {spread:.2f}. CLOSING OPEN ORDER IF ANY')
				self.to_market(3)
				counter += 1
				bid, ask, spread = self.bid_ask_spread()
				spread_mean.append(spread)
				if counter%60 == 0:
					print(f'Spread too wide: {spread:.2f}')
				time.sleep(5)
			else:
				wide_spread = False

	def bid_ask_spread(self):
		bid = trader.symbols[SYMBOL_DICT[SYMBOL]].bid
		ask = trader.symbols[SYMBOL_DICT[SYMBOL]].ask
		spread = (ask - bid)*SYMBOL_FACTOR
		return bid, ask, spread


	def action_check(self, action, market_closed):
		# Check action if its legal and then send it to market or return a legal action if chosen action is illegal.
		# 0 = do nothing   #1 = buy   #2 = sell   #3 = close order
		if market_closed:
			if self.orders != "":  # Is it end of day
				action = 3
			else:
				action = 0
		else:
			if action == 0:
				pass

			elif action == 1:
				if self.orders == "long":
					action = 0
				elif self.orders == "short":
					action = 3

			elif action == 2:
				if self.orders == "long":
					action = 3
				elif self.orders == "short":
					action = 0

			elif action == 3:
				if self.orders == "long":
					action = 3
				elif self.orders == "short":
					action = 3
				else:
					action = 0

		return action


	def place_order(self):
		open_pos = []
		order_id = 0
  
		confirm = trader.open_trade(account_id=self.accountID,
								symbol=SYMBOL_DICT[SYMBOL],
								is_buy=self.ordertype,
								amount=LOTS,
								rate=0,
								at_market=0,
								time_in_force="GTC",
								order_type="AtMarket",
								stop=None,
								trailing_step=None,
								limit=None,
								is_in_pips=None)


		print(f'CONFIRM: {confirm} - SELF.ORDERS: {self.orders}')

		order_id = confirm['data']['orderId']
		time.sleep(2)
		self.tradeID = trader.get_tradeId(order_id)
		
		if self.tradeID == None:
			print('Trade ID not returned from system...trying to hack the bastard')
			open_pos = trader.get_model('OpenPosition')['open_positions']
			for index in range(len(open_pos)):
				if open_pos[index]['currency'] == SYMBOL_DICT[SYMBOL]:
					self.tradeID = open_pos[index]['tradeId']
					break

		open_pos = trader.get_model('OpenPosition')['open_positions']
		print(f'ORDER_id: {order_id} trade_ID: {self.tradeID}')
		for index in range(len(open_pos)):
			if open_pos[index]['tradeId'] == self.tradeID:
				self.order_price = open_pos[index]['open']
				print(f'Orderprice: {self.order_price}')
    
	def close_order(self):
		closed_pos = []

		info = trader.close_trade(self.tradeID, LOTS)

		print(f'INFO ON CLOSE: {info}')
		time.sleep(5)
		closed_pos = trader.get_model('ClosedPosition')['closed_positions']
		for index in range(len(closed_pos)):
			if closed_pos[index]['tradeId'] == self.tradeID:
				self.close = closed_pos[index]['close']

	def set_tp(self, tp_pred):
		bid, ask, spread = self.bid_ask_spread()
		if abs(tp_pred) > (MIN_TP/SYMBOL_FACTOR):
			if tp_pred > 0:
				action = 1
				self.take_profit = bid + tp_pred
			else:
				action = 2
				self.take_profit = ask + tp_pred

			self.to_market(action)
		else:
			print('TP to low')




	def to_market(self, action, market_closed=False):
		
		#action = self.action_check(action, market_closed)    
		if action == 1:
            #Send buy order to market
			self.orders = 'long'
			self.ordertype = True
			print(f'LONG - TAKE PROFIT AT {self.take_profit}')
			self.place_order()

		elif action == 2:
            #Send sell order to market
			self.orders = 'short'
			self.ordertype = False
			print(f'SHORT - TAKE PROFIT AT {self.take_profit}')
			self.place_order()
        
		elif action == 3:
            #Send close order to market
			self.close_order()
			if self.orders =='long':
				self.win_loss = (self.close - self.order_price)*SYMBOL_FACTOR #close long order
			
			elif self.orders =='short':
				self.win_loss = (self.order_price - self.close)*SYMBOL_FACTOR #close short order
			
			self.trailing = False
			self.trailstop = 0
			self.book_keeping() #Send trade info to CSV file
			self.orders =''
			self.order_price = 0

	def book_keeping(self):
		timestamp = self.to_datetime()
		time_string = timestamp.strftime("%Y.%m.%d, %H:%M:%S")
		self.total_win_loss += self.win_loss
        
        # Create a numpy array with all the trade/state info for the current state
		print(f'{time_string}, {self.close:6.2f}, {self.orders:5}, {self.win_loss:7.2f}, {self.total_win_loss:7.2f}')
		trade_info = (np.array([time_string, self.orders, self.order_price, self.take_profit, self.close, self.win_loss, self.total_win_loss, self.trailorder]))  #.reshape([1, -1]).flatten())

		#book_time = dt.now().strftime("%Y%m%d_%H%M%S")
		with open(f"output/{BOOKKEEPING}.csv", "a") as f:
			np.savetxt(f, trade_info, fmt="%s", newline=" ", delimiter=",")
			f.write("\n")

	def position(self, tp_pred, current_state):
		bid, ask, spread = self.bid_ask_spread()
		if tp_pred > 0 and (current_state[0,32]*SYMBOL_FACTOR) < 5: #approve trade if long trade and max 5 pip away from 15 min MA14
			return True
		elif tp_pred < 0 and (current_state[0,32]*SYMBOL_FACTOR) > -5:
			return True
		else:
			print(f'Trade rejected - Find better position! - MA: {current_state[0,32]:.2f}')
			return False



class XGB_agent():
	def __init__(self):
		self.xgb_model = joblib.load(xgb_filename)

def trader_init():
	trader.debug_level = "ERROR" # verbose logging... don't set to receive errors only. Levels are (from most to least logging) DEBUG, INFO, WARNING, ERROR, CRITICAL
	trader.login()
	time.sleep(5)
	env.accountID = None
	while env.accountID == None:
		env.accountID = trader.account_id
		print(f'ACCOUNT ID: {env.accountID}')
	trader.subscribe_symbol(SYMBOL_DICT[SYMBOL])
	


timestep = 0
xgb = XGB_agent()
env = forex_env()
print('---------0----------')
trader = fxcm_rest_api.Trader('457e2b9adg532fdd77722e78e70f8eb4a1c', 'demo')
trader_init()


env.init_data(filename)

while True:
	get_time = dt.now()
	if get_time.isoweekday() > 5 or (get_time.isoweekday() == 5 and get_time.hour > 21):
		market_closed = True
		print('MARKET CLOSED')
		if env.orders != "":
			env.to_market(3, True)
		time.sleep(60)

	elif 1 < get_time.hour >= 22:
		market_closed = True
		print('MARKET CLOSED')
		if env.orders != "":
			env.to_market(3, True)
		time.sleep(60)
		
	else:
		market_closed = False
		print(f'---------{BOOKKEEPING}---{env.orders}--ORDERPRICE: {env.order_price}-- TAKE PROFIT: {env.take_profit}--{env.total_win_loss:7.2f}----------')
		trader.subscribe_symbol(SYMBOL_DICT[SYMBOL])
		      
		current_state = env.get_data(filename)

		tp_pred = xgb.xgb_model.predict(current_state)
		bid, ask, spread = env.bid_ask_spread()
		print(f'TP: {tp_pred}')

		if env.orders == "" and not market_closed:
			env.set_tp(tp_pred)
		