from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import List, Dict, Optional

__all__ = [
	'Direction', 'TimeInForce', 'InstrumentID', 'InstrumentInfo',
	'BarData', 'Position', 'Order', 'Trade', 'StrategyCommands', 'Strategy'
]


@unique
class Direction(IntEnum):
	Buy = 0
	Sell = 1


@unique
class TimeInForce(IntEnum):
	IOC = 0  # immediate or cancel
	GTC = 1  # good till cancel


@dataclass
class InstrumentID:
	exchange: str
	symbol: str

	def __init__(self, ins_id_str: str):

		split = ins_id_str.split('.')
		if len(split) == 2:
			self.exchange = split[0]
			self.symbol = split[1]
		else:
			raise ValueError(f"Malformed {ins_id_str=}. Should be like SHFE.CU2105")

	def __str__(self) -> str:
		return f"{str(self.exchange)}.{str(self.symbol)}"

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return hash(str(self))


@dataclass
class InstrumentInfo:
	# instrument id
	ins_id: InstrumentID
	# minimum price change unit
	px_tick: float
	# minimum quantity change unit
	qty_tick: float
	# contract size
	multiplier: float
	# fee per quantity
	fee_per_qty: float
	# fee per market value
	fee_per_mv: float


@dataclass
class BarData:
	# create time
	time: dt.datetime
	# last updated time
	last_time: dt.datetime
	# open price
	open_px: float
	# highest price
	high_px: float
	# lowest price
	low_px: float
	# last price
	last_px: float
	# traded quantity during interval
	trade_qty: float
	# traded market value during interval
	trade_mv: float
	# last market holding quantity
	hold_qty: float


@dataclass
class Position:
	# instrument info
	instrument: InstrumentInfo
	# max allowed quantities of holding position
	max_hold_qty: float
	# holding quantity
	hold_qty: float = field(init = False, default = 0)
	# market value of holding position
	hold_mv: float = field(init = False, default = 0)
	# quantity of pending orders
	pending_qty: Dict[Direction, float] = field(
		init = False, default_factory = lambda: {x: 0 for x in [Direction.Buy, Direction.Sell]})
	# profit of closed position
	real_profit: float = field(init = False, default = 0)
	# fee of session's orders and trades
	fee: float = field(init = False, default = 0)


@dataclass
class Order:
	# order unique id
	order_id: int
	# trading position
	position: Position
	# trading direction
	direction: Direction
	# time in force
	tif: TimeInForce
	# limit price
	limit_px: Optional[float] = field(default = None)
	# original quantity
	orig_qty: float = field(default = 0)
	# pending quantity
	pend_qty: float = field(init = False, default = 0)
	# executed quantity
	exec_qty: float = field(init = False, default = 0)
	# order insert time
	insert_time: dt.datetime = field(default = None)
	# order last time
	last_time: dt.datetime = field(init = False, default = None)

	def __post_init__(self):
		self.pend_qty = self.orig_qty

	def _on_cancelled(self, time: dt.datetime):
		self.last_time = time
		self.pend_qty = 0

	def _on_executed(self, time: dt.datetime, qty: float):
		self.last_time = time
		self.pend_qty -= qty
		self.exec_qty += qty


@dataclass
class Trade:
	# trade unique id
	trade_id: int
	# belonging order
	order: Order
	# executed time
	time: dt.datetime
	# executed price
	px: float
	# executed qty
	qty: float
	# fee
	fee: float
	# profit
	profit: float


class StrategyCommands(ABC):
	"""
	Abstract strategy commands class
	"""

	@abstractmethod
	def get_positions(self) -> List[Position]:
		"""Get all instrument positions"""
		pass

	@abstractmethod
	def find_position(self, ins_id: InstrumentID) -> Optional[Position]:
		"""Find instrument position"""
		pass

	@abstractmethod
	def submit_order(self, ins_id: InstrumentID, dir: Direction, qty: float,
	                 tif: TimeInForce, lmt_px: Optional[float] = None) -> Optional[Order]:
		"""Submit order"""
		pass

	@abstractmethod
	def cancel_order(self, order: Order) -> bool:
		"""Cancel order"""
		pass


class Strategy(ABC):
	"""
	Abstract MLFT strategy class
	"""

	def __init__(self, name: str):
		self._name: str = name
		self._cmd: StrategyCommands = None

	def commands(self) -> StrategyCommands:
		"""Get commands"""
		return self._cmd

	def name(self) -> str:
		"""Get strategy name"""
		return self._name

	def init(self, cmd: StrategyCommands):
		"""Initialize local variables"""
		self._cmd = cmd

	@abstractmethod
	def on_start(self):
		"""Triggered when run strategy"""
		pass

	@abstractmethod
	def on_stop(self):
		"""Triggered when stop strategy"""
		pass

	@abstractmethod
	def on_order_cancelled(self, order: Order):
		"""Triggered when order cancelled"""
		pass

	@abstractmethod
	def on_order_executed(self, trade: Trade):
		"""Triggered when order executed"""
		pass

	@abstractmethod
	def on_bar_data(self, ins_id: InstrumentID, bar: BarData):
		"""Triggered when receive bar data"""
		pass
