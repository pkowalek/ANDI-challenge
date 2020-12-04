import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from _07b_stats import plot_confusion_matrix, pandas_classification_report,get_features_importances

np_load_old = np.load
# save np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def generate_stats(simulation_folder, featured_model="RF",dim="1D"):
    """
    Function for generating statistics about model
    :param scenario: "Old" or other if defined
    :param featured_model: "RF", "GB" or other id defined
    :return: statistics
    """

    Start = datetime.now()
    project_directory = os.getcwd()
    path_to_data = os.path.join(project_directory, "Data", "Synthetic data", simulation_folder)
    path_to_model = os.path.join(project_directory, "Models", featured_model, simulation_folder)
    path_to_characteristics_data = os.path.join(project_directory,"Data", "characteristics")
    X_train = np.load(os.path.join(path_to_data, "X_train.npy"))
    X_test = np.load(os.path.join(path_to_data, "X_test.npy"))
    y_train = np.load(os.path.join(path_to_data, "y_train.npy"))
    y_test = np.load(os.path.join(path_to_data, "y_test.npy"))

    path_to_stats = os.path.join(path_to_model, "Stats")
    if not os.path.exists(path_to_stats):
        os.makedirs(path_to_stats)

    path_to_model_file = os.path.join(path_to_model, "model.sav")

    classes = list(range(5))

    model = joblib.load(path_to_model_file)
    data_type = ["Train", "Test"]
    for dt in data_type:
        X = X_train if dt == "Train" else X_test
        y = y_train if dt == "Train" else y_test
        # Making the Confusion Matrix
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        # Plot non-normalized confusion matrix
        fig = plt.figure()
        plot_confusion_matrix(cm, classes=classes, title='Confusion matrix, without normalization')
        fig.savefig(os.path.join(path_to_stats,
                                 "Confusion_Matrix_NotNormalized_" + dt + "_" + featured_model + ".pdf"),
                    dpi=fig.dpi)
        plt.close()

        # Plot normalized confusion matrix
        fig = plt.figure()
        plot_confusion_matrix(cm, classes=classes, normalize=True, title='Normalized confusion matrix')
        fig.savefig(os.path.join(path_to_stats,
                                 "Confusion_Matrix_Normalized_" + dt + "_" + featured_model + ".pdf"),
                    dpi=fig.dpi)
        plt.close()

        # class report
        print("class report")
        report = classification_report(y, y_pred)#, target_names=classes)
        print(report)
        #report = pandas_classification_report(report)
        #report.to_csv(
        #    os.path.join(path_to_stats, "Classification_Report_" + dt + "_" + featured_model + ".csv"))

        # accuracy
        print("acc")
        acu = accuracy_score(y, y_pred)
        df = pd.DataFrame({'acc': [acu]})
        df.to_csv(os.path.join(path_to_stats, "Accuracy_" + dt + "_" + featured_model + ".csv"))
        
    characteristics_data = pd.read_csv(os.path.join(path_to_characteristics_data, "characteristics_"+dim+".csv"))
    get_features_importances(model, characteristics_data)
    '''# feature importances
    print("features importances")
    fig = plt.figure()
    df, pl = get_features_importances(model, characteristics_data)
    df.to_csv(os.path.join(path_to_stats, "Feature_importances.csv"), index=False)
    pl.get_figure().savefig(
        os.path.join(path_to_stats, "Features_Importances" + test_version + "_" + featured_model + ".pdf"), dpi=fig.dpi)

    # permutation importances
    print("permutation importances")
    df = get_permutation_importances(model, X_train, y_train, characteristics_data)
    df.to_csv(os.path.join(path_to_stats, "Permutation_fi" + test_version + "_" + featured_model + ".csv"), index=True)

    # drop column feature importance
    #    # FIX: after change X_train etc. to contain data names, change the snippet here
    #    X_train_df = pd.DataFrame(X_train, columns=column_names)
    #    df = drop_col_feat_imp(model, X_train_df, y_train)
    #    df.to_csv(os.path.join(path_to_stats, "Drop_column_fi.csv"), index=False)'''

    End = datetime.now()
    ExecutedTime = End - Start
    df = pd.DataFrame({'ExecutedTime': [ExecutedTime]})
    df.to_csv(os.path.join(path_to_stats, "time_for_stats_generator.csv"))
    print(ExecutedTime)


if __name__ == "__main__":
    generate_stats(simulation_folder="Base_corr_Andi_1D", featured_model='GB',dim="1D")
    #generate_stats(simulation_folder="Base_corr_Andi_2D", featured_model='GB',dim="2D")
    generate_stats(simulation_folder="Base_corr_Andi_3D", featured_model='GB',dim="3D")
    np.load = np_load_old
