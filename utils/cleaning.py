import pandas as pd
from pandas.core.frame import DataFrame


class Cleaning:
    @staticmethod
    def get_data(url : str) -> DataFrame:
        df = pd.read_csv(url)
        return df
