import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from .strategy import *

__all__ = [
    'MatchAlgorithm', 'BacktestConfig', 'BacktestEngine'
]


@unique
class MatchAlgorithm(IntEnum):
	"""撮合类型"""
	NoTrade = 0  # No trade will be generated for order
	AlwaysFilled = 1  # Order is always filled


@dataclass
class BacktestConfig:
	# instrument file path
	instrument_path: str
	# market data file path
	bar_data_path: str
	# matching algorithm
	match_algo: MatchAlgorithm = field(default = MatchAlgorithm.AlwaysFilled)


@dataclass
class _Event:
	# event type
	event_type: int
	# order or trade
	order: Order


class BacktestEngine(StrategyCommands):
	"""
	Simple backtest engine to run strategy
	"""

	def __init__(self, config: BacktestConfig, strategy: Strategy):
		self._positions: Dict[InstrumentID, Position] = {}
		self._orders: List[Order] = []
		self._trades: List[Trade] = []
		self._events: List[_Event] = []
		self._cur_tm: dt.datetime = None
		self._cur_px: Dict[InstrumentID, float] = {}
		# load instruments
		logging.info(f'load instrument info from: {config.instrument_path}')
		if not os.path.exists(config.instrument_path):
			raise FileNotFoundError(f"missing instrument file: {config.instrument_path}")
		ins_df = pd.read_csv(config.instrument_path, encoding = "utf8", dtype = str)
		for r in ins_df.itertuples():
			logging.info(f'add instrument [{r.ins_id}]')
			ins_id = InstrumentID(r.ins_id)
			self._positions[ins_id] = \
				Position(
					instrument = InstrumentInfo(
						ins_id = InstrumentID(r.ins_id),
						px_tick = float(r.px_tick),
						qty_tick = float(r.qty_tick),
						multiplier = float(r.multiplier),
						fee_per_qty = float(r.fee_per_qty),
						fee_per_mv = float(r.fee_per_mv),
					),
					max_hold_qty = float(r.max_hold_qty),
				)
		# load time bar data from csv
		logging.info(f'load bar data from: {config.bar_data_path}')
		if not os.path.exists(config.bar_data_path):
			raise FileNotFoundError(f"missing bar data file: {config.bar_data_path}")
		bar_df = pd.read_csv(config.bar_data_path, encoding = "utf8", dtype = {
			'ins_id': str,
			'time': str,
			'last_time': str,
			'open_px': float,
			'high_px': float,
			'low_px': float,
			'last_px': float,
			'trade_qty': float,
			'trade_mv': float,
			'hold_qty': float,
		})
		bar_df['time'] = pd.to_datetime(bar_df['time'])
		bar_df['last_time'] = pd.to_datetime(bar_df['last_time'])
		# backtesting
		logging.info(f'sending bar data')
		strategy.init(cmd = self)
		strategy.on_start()
		for r in bar_df.itertuples(index = False):
			ins_id = InstrumentID(r.ins_id)
			bar_args = r._asdict()
			del bar_args['ins_id']
			bar = BarData(**bar_args)
			bar.time = bar.time.to_pydatetime()
			bar.last_time = bar.last_time.to_pydatetime()
			self._cur_tm = bar.last_time
			self._cur_px[ins_id] = bar.last_px
			# process events
			for _ev in self._events:
				if _ev.order.pend_qty <= 0:
					continue
				_ev.order.last_time = self._cur_tm
				ev_pos = _ev.order.position
				ev_qty = _ev.order.pend_qty
				if _ev.event_type == 0:
					if config.match_algo == MatchAlgorithm.AlwaysFilled:
						# fill order
						ev_px = self._cur_px[ins_id] if ins_id in self._cur_px else np.NaN
						_ev.order.exec_qty = ev_qty
						_ev.order.pend_qty = 0
						ev_pos.pending_qty[_ev.order.direction] -= ev_qty
						if _ev.order.direction == Direction.Buy:
							if ev_pos.hold_qty < 0:
								ev_close_qty = min(-ev_pos.hold_qty, ev_qty)
								ev_profit = (ev_pos.hold_mv / ev_pos.hold_qty - ev_px) * ev_close_qty
								ev_pos.hold_mv *= 1 - (ev_close_qty / -ev_pos.hold_qty)
							else:
								ev_close_qty = 0
								ev_profit = 0
							ev_pos.hold_qty += ev_qty
							ev_pos.hold_mv += (ev_qty - ev_close_qty) * ev_px
						else:
							if ev_pos.hold_qty > 0:
								ev_close_qty = min(ev_pos.hold_qty, ev_qty)
								ev_profit = (ev_px - ev_pos.hold_mv / ev_pos.hold_qty) * ev_close_qty
								ev_pos.hold_mv *= 1 - (ev_close_qty / ev_pos.hold_qty)
							else:
								ev_close_qty = 0
								ev_profit = 0
							ev_pos.hold_qty -= ev_qty
							ev_pos.hold_mv -= (ev_qty - ev_close_qty) * ev_px
						ev_fee = ev_qty * ev_pos.instrument.fee_per_qty + \
						         ev_qty * ev_px * ev_pos.instrument.fee_per_mv
						trade = Trade(
							trade_id = len(self._trades),
							order = _ev.order,
							time = self._cur_tm,
							px = ev_px,
							qty = ev_qty,
							fee = ev_fee,
							profit = ev_profit
						)
						self._trades.append(trade)
						ev_pos.fee += ev_fee
						ev_pos.real_profit += ev_profit
						strategy.on_order_executed(trade)
					elif _ev.order.tif == TimeInForce.IOC:
						_ev.order.pend_qty = 0
						ev_pos.pending_qty[_ev.order.direction] -= ev_qty
						strategy.on_order_cancelled(_ev.order)
				elif _ev.event_type == 1:
					_ev.order.pend_qty = 0
					ev_pos.pending_qty[_ev.order.direction] -= ev_qty
					strategy.on_order_cancelled(_ev.order)
				else:
					raise RuntimeError(f'unknown event type {_ev.event_type}')
			self._events.clear()
			# notify bar data
			strategy.on_bar_data(ins_id = ins_id, bar = bar)
		strategy.on_stop()
		# aggregate data
		self.orders = pd.DataFrame({
			'order_id': [x.order_id for x in self._orders],
			'ins_id': [x.position.instrument.ins_id for x in self._orders],
			'direction': [x.direction for x in self._orders],
			'tif': [x.tif for x in self._orders],
			'limit_px': [x.limit_px for x in self._orders],
			'orig_qty': [x.orig_qty for x in self._orders],
			'pend_qty': [x.pend_qty for x in self._orders],
			'exec_qty': [x.exec_qty for x in self._orders],
			'insert_time': [x.insert_time for x in self._orders],
			'last_time': [x.last_time for x in self._orders],
		})
		self.trades = pd.DataFrame({
			'trade_id': [x.trade_id for x in self._trades],
			'order_id': [x.order.order_id for x in self._trades],
			'ins_id': [x.order.position.instrument.ins_id for x in self._trades],
			'direction': [x.order.direction for x in self._trades],
			'px': [x.px for x in self._trades],
			'qty': [x.qty for x in self._trades],
			'fee': [x.fee for x in self._trades],
			'profit': [x.profit for x in self._trades],
		})

	#######################################################################################
	# Strategy commands impl
	#######################################################################################

	def get_positions(self) -> List[Position]:
		return list(self._positions.values())

	def find_position(self, ins_id: InstrumentID) -> Position:
		return self._positions[ins_id] if ins_id in self._positions else None

	def submit_order(self, ins_id: InstrumentID, dir: Direction, qty: float,
	                 tif: TimeInForce, lmt_px: Optional[float] = None) -> Optional[Order]:
		ins_pos = self.find_position(ins_id)
		if ins_pos is None:
			return None
		# check if new order is viable under current position risk settings
		if dir == Direction.Buy:
			max_qty = ins_pos.max_hold_qty - ins_pos.hold_qty - ins_pos.pending_qty[dir]
		else:
			max_qty = ins_pos.max_hold_qty + ins_pos.hold_qty - ins_pos.pending_qty[dir]
		qty = min(qty, max_qty)
		if qty <= 0:
			return None
		order = Order(
			order_id = len(self._orders),
			position = ins_pos,
			direction = dir,
			tif = tif,
			limit_px = lmt_px,
			orig_qty = qty,
			insert_time = self._cur_tm,
		)
		self._orders.append(order)
		ins_pos.pending_qty[dir] += qty
		self._events.append(_Event(event_type = 0, order = order))
		return order

	def cancel_order(self, order: Order) -> bool:
		if order.pend_qty == 0 or order.tif == TimeInForce.IOC:
			return False
		self._events.append(_Event(event_type = 1, order = order))
		return True
