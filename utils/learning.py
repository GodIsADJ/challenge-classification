import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.core.frame import DataFrame
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from utils.preprocessing import Preprocessing


class Learning(Preprocessing):
    def __init__(self, dataframe: DataFrame):
        """Initialize the Learning object. The parameters are initialize at None.
        Call logistic_reg, random_forest, knn or decision_tree function to initialize param with values.

        Args:
            dataframe (DataFrame): The dataframe created before
        """
        super().__init__(dataframe)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model: object = None
        self.y_pred: list = None
        self.train_score: float = None
        self.test_score: float = None
        self.auc_score: float = None

    def __set_score(self) -> None:
        """Private function to initalize the 3 parameter "score"
        This function is called in logistic_reg, random_forest, knn and decision_tree
        """
        self.train_score = self.model.score(self.X_train, self.y_train)
        self.test_score = self.model.score(self.X_test, self.y_test)
        self.auc_score = self.__get_auc_score()

    def print_best_parameters(self, paramList: dict) -> None:
        """Print the best parameters found with the GridSearchCV function for the current model

        Args:
            paramList (dict): The list of parameters to test
        """
        gridsearch = GridSearchCV(estimator=self.model,
                                  param_grid=paramList,
                                  scoring='accuracy',
                                  cv=5,  # Use 5 folds
                                  verbose=1,
                                  n_jobs=-1  # Use all but one CPU core
                                  )
        result = gridsearch.fit(self.X_train, self.y_train)
        print(f"Best params score : {result.best_score_}")
        print(f"Best params list : {result.best_params_}")

    def logistic_reg(self) -> None:
        """Make a LogisticRegression ans initialize parameters
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataframe.drop(["target"], axis=1),
            self.dataframe.target,
            test_size=0.2,
            random_state=42,
        )
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        self.__set_score()
        self.y_pred = self.model.predict(self.X_test)

    def random_forest(self, random_state: int = 42) -> None:
        """Make a RandomForestClassifier and initialize parameters

        Args:
            random_state (int, optional):  Defaults to 42.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataframe.drop(["target"], axis=1),
            self.dataframe.target,
            test_size=0.2,
            random_state=42,
        )
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)
        self.__set_score()
        self.y_pred = self.model.predict(self.X_test)

    def __find_best_n_knn(self, max_neighbors: int) -> int:
        """Find the best n_neighbors parameter between 1 and max_neighbors for the KNeighborsClassifier

        Args:
            max_neighbors (int): the maximum number

        Returns:
            int: the best parameter
        """
        test_score = []
        k_vals = []
        for k in range(1, max_neighbors):
            k_vals.append(k)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.y_train)

            te_score = knn.score(self.X_test, self.y_test)
            test_score.append(te_score)
        # Determine the value of k.
        max_test_score = max(test_score)
        test_scores_ind = [i for i, v in enumerate(
            test_score) if v == max_test_score]
        return test_scores_ind[len(test_scores_ind)//2]

    def knn(self, n_neighbors: int = 18, weights: str = "uniform", algorithm: str = "ball_tree", metric: str = "manhattan") -> None:
        """Make a KNeighborsClassifier and initialize parameters

        Args:
            n_neighbors (int, optional):  Defaults to 18.
            weights (str, optional):  Defaults to "uniform".
            algorithm (str, optional):  Defaults to "ball_tree".
            metric (str, optional):  Defaults to "manhattan".
        """

        # "": [10, 16, 17, 18, 20],
        #               "": ["uniform", "distance"],
        #               "": ["ball_tree", "auto"],
        #               "leaf_size": [30],
        #               "": ["manhattan", "euclidean"]

        ss = StandardScaler()
        x = ss.fit_transform(self.dataframe.drop(["target"], axis=1))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, self.dataframe.target, test_size=0.2, random_state=42
        )
        # n_neighbors = self.__find_best_n_knn(max_neighbors)
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, metric=metric)
        self.model.fit(self.X_train, self.y_train)
        self.__set_score()
        self.y_pred = self.model.predict(self.X_test)

    def decision_tree(self) -> None:
        """Make a decision tree and initialize parameters
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataframe.drop(["target"], axis=1),
            self.dataframe.target,
            test_size=0.2,
            random_state=42,
        )
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.__set_score()
        self.y_pred = self.model.predict(self.X_test)

    def svc(self) -> None:
        """Make a SVC and initialize the parameters
        """
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

    def linear_svc(self) -> None:
        """Make a linearSVC and initialize the parameters
        """
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
        """Diplay the confusion matrix for the current model in a popup window
        """
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
        """Display the ROC curve in a popup window
        """
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

    def __get_auc_score(self) -> float:
        """Calculate the auc score for the current model

        Returns:
            float: the AUC score
        """
        if self.X_test is not None:
            y_scores = self.model.predict_proba(self.X_test)
            y_scores = y_scores[:, 1]
            auc_score = roc_auc_score(self.y_test, y_scores)
            return auc_score
        else:
            return 0

    def print_roc_auc_score(self) -> None:
        """Print the AUC score
        """
        auc_score = self.__get_auc_score()
        print(f"Auc score : {auc_score} ({self.model})")

    def print_train_score(self) -> None:
        """Print the train score
        """
        if self.train_score is not None:
            print(f"Train score : {self.train_score} ({self.model})")

    def print_test_score(self) -> None:
        """Print the test score
        """
        if self.test_score is not None:
            print(f"Test score : {self.test_score} ({self.model})")
