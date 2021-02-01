from sys import float_repr_style

from utils.cleaning import Cleaning
from utils.learning import Learning

if __name__ == "__main__":

    url = "data/UCI_Credit_Card.csv"
    df = Cleaning.get_data(url)

    logistic = Learning(df)
    # ml.display_correlation_matrix()
    logistic.logistic_reg()
    # logistic.display_confusion_matrix()
    # logistic.display_ROC_curve()
    logistic.print_train_score()
    logistic.print_test_score()
    logistic.print_roc_auc_score()

    forest = Learning(df)
    forest.random_forest(10)
    # forest.display_confusion_matrix()
    # forest.display_ROC_curve()
    forest.print_train_score()
    forest.print_test_score()
    forest.print_roc_auc_score()

    knn = Learning(df)
    knn.knn()
    params = {"n_neighbors": [10, 16, 17, 18, 20],
              "weights": ["uniform", "distance"],
              "algorithm": ["ball_tree", "auto"],
              "metric": ["manhattan", "euclidean"]
              }
    knn.print_best_parameters(params)
    knn.knn(n_neighbors=16)
    # knn.display_confusion_matrix()
    # knn.display_ROC_curve()
    knn.print_train_score()
    knn.print_test_score()
    knn.print_roc_auc_score()

    DT = Learning(df)
    DT.decision_tree()
    # DT.display_confusion_matrix()
    # DT.display_ROC_curve()
    DT.print_train_score()
    DT.print_test_score()
    DT.print_roc_auc_score()

    # SLOW
    # svc = Learning(df)
    # svc.svc()
    # # svc.display_confusion_matrix()
    # # svc.display_ROC_curve()
    # svc.print_test_score()

    # linear_svc = Learning(df)
    # linear_svc.linear_svc()
    # linear_svc.print_test_score()
