import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.frame import DataFrame


class Preprocessing:
    def __init__(self, dataframe: DataFrame):
        dataframe.rename(columns={"default.payment.next.month": "target"}, inplace=True)
        self.__df = dataframe

    @property
    def dataframe(self) -> DataFrame:
        return self.__df

    def display_correlation_matrix(self) -> None:
        plt.figure(figsize=(20, 10))
        plt.title("Correlation between dataset variables")
        sns.heatmap(self.dataframe.corr(), annot=True, cmap="coolwarm")
        plt.tight_layout()
        plt.show()
