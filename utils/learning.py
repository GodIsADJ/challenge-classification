import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.core.frame import DataFrame
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from utils.preprocessing import Preprocessing


class Learning(Preprocessing):
    def __init__(self, dataframe: DataFrame):
        super().__init__(dataframe)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model: object = None
        self.y_pred: list = None
        self.score: float = None

    def logistic_reg(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataframe.drop(["target"], axis=1),
            self.dataframe.target,
            test_size=0.2,
            random_state=42,
        )
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        self.y_pred = self.model.predict(self.X_test)

    def random_forest(self, random_state: int = 42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataframe.drop(["target"], axis=1),
            self.dataframe.target,
            test_size=0.2,
            random_state=42,
        )
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        self.y_pred = self.model.predict(self.X_test)

    def knn(self):
        ss = StandardScaler()
        x = ss.fit_transform(self.dataframe.drop(["target"], axis=1))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, self.dataframe.target, test_size=0.2, random_state=42
        )
        self.model = KNeighborsClassifier(10)
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        self.y_pred = self.model.predict(self.X_test)

    def decision_tree(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataframe.drop(["target"], axis=1),
            self.dataframe.target,
            test_size=0.2,
            random_state=42,
        )
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        self.y_pred = self.model.predict(self.X_test)

    def svc(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataframe.drop(["target"], axis=1),
            self.dataframe.target,
            test_size=0.2,
            random_state=42,
        )
        self.model = SVC(probability=True)
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        self.y_pred = self.model.predict(self.X_test)

    def linear_svc(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataframe.drop(["target"], axis=1),
            self.dataframe.target,
            test_size=0.2,
            random_state=42,
        )
        self.model = LinearSVC()
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        self.y_pred = self.model.predict(self.X_test)

    def display_confusion_matrix(self) -> None:
        if self.y_pred is not None:
            cnf_matrix = confusion_matrix(self.y_test, self.y_pred)
            group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
            group_counts = ["{0}".format(value)
                            for value in cnf_matrix.flatten()]
            group_percentages = [
                "{0:.2%}".format(value)
                for value in cnf_matrix.flatten() / np.sum(cnf_matrix)
            ]
            labels = [
                f"{v1}\n{v2}\n{v3}"
                for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
            ]
            labels = np.asarray(labels).reshape(2, 2)
            ax = sns.heatmap(cnf_matrix, annot=labels, fmt="", cmap="coolwarm")
            ax.set(
                title="Confusion Matrix", xlabel="Predicted label", ylabel="True label"
            )

    def display_ROC_curve(self) -> None:
        if self.X_test is not None:
            y_scores = self.model.predict_proba(self.X_test)
            y_scores = y_scores[:, 1]
            # compute true positive rate and false positive rate
            fpr, tpr, thresholds = roc_curve(self.y_test, y_scores)
            # plotting them against each other
            plt.figure(figsize=(14, 7))
            plt.plot(fpr, tpr, linewidth=4)
            plt.plot([0, 1], [0, 1], "k--", linewidth=2)
            plt.xlabel("False Positive Rate (FPR)", fontsize=16)
            plt.ylabel("True Positive Rate (TPR)", fontsize=16)
            plt.tight_layout()
            plt.show()

    def print_roc_auc_score(self) -> None:
        if self.X_test is not None:
            y_scores = self.model.predict_proba(self.X_test)
            y_scores = y_scores[:, 1]
            auc_score = roc_auc_score(self.y_test, y_scores)
            print(f"auc score : {auc_score} ({self.model})")

    def print_score(self) -> None:
        if self.score is not None:
            print(f"{self.score} ({self.model})")
