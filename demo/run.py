import logging
import os
import random
import sys

from mlft.strategy import *
from mlft.backtest import *

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class DemoStrategy(Strategy):
    """
    Demo strategy showcasing all APIs
    """

    def __init__(self):
        super().__init__(name = "Demo")

    def on_start(self):
        logging.info(f"on_start strategy_id={self.name()}")

    def on_stop(self):
        logging.info(f"on_stop strategy_id={self.name()}")

    def on_order_cancelled(self, order: Order):
        logging.info(f"on_order_cancelled {order=}")

    def on_order_executed(self, trade: Trade):
        logging.info(f"on_order_executed {trade=}")

    def on_bar_data(self, ins_id: InstrumentID, bar: BarData):
        logging.info(f"on_bar_data {ins_id=} {bar=}")

        # generate random order
        random_dir = random.choice([Direction.Buy, Direction.Sell])
        random_qty = random.randint(1, 5)
        self.commands().submit_order(
            ins_id=ins_id,
            dir=random_dir,
            qty=random_qty,
            tif=TimeInForce.IOC,
        )


if __name__ == '__main__':

    cur_path = os.path.curdir

    config = BacktestConfig(
        instrument_path=os.path.join(cur_path, 'data', 'instrument.csv'),
        bar_data_path=os.path.join(cur_path, 'data', 'bar_data.csv'),
        match_algo=MatchAlgorithm.AlwaysFilled,
    )

    strategy = DemoStrategy()
    result = BacktestEngine(config=config, strategy=strategy)

    # show results
    print(result.orders)
    print(result.trades)
