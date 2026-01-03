# data_adapter.py

import pandas as pd


class MarketDataAdapter:
    """
    Adapts arbitrary market datasets into a standardized schema.

    Standard schema:
    - timestamp
    - target (optional, for training)
    - benchmark_return
    - risk_free_rate
    - feature columns (float)
    """

    def __init__(self, column_map: dict):
        """
        column_map example:
        {
            "timestamp": "date_id",
            "target": "market_forward_excess_returns",
            "benchmark_return": "forward_returns",
            "risk_free_rate": "risk_free_rate"
        }
        """
        self.column_map = column_map

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for standard_name, raw_name in self.column_map.items():
            if raw_name not in df.columns:
                raise KeyError(f"Required column '{raw_name}' not found in dataset")
            df[standard_name] = df[raw_name]

        return df
