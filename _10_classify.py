import os
import numpy as np
import joblib
import pandas as pd


def get_classification(simulation_folder,set_name,res_folder):
    project_directory = os.getcwd()
    featured_model = "GB"
    path_to_val = os.path.join(project_directory, "ValidationDatasets","Results")
    path_to_data = os.path.join(path_to_val,set_name)
    path_to_save = os.path.join(path_to_val,res_folder,featured_model)
    path_to_model = os.path.join(project_directory, "Models", featured_model, simulation_folder)
    X = np.load(os.path.join(path_to_data, "X_data.npy"))

    path_to_model_file = os.path.join(path_to_model, "model.sav")
    model = joblib.load(path_to_model_file)
    y_pred = model.predict_proba(X)
       
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    np.save(os.path.join(path_to_save, "y_pred.npy"), y_pred)



    df = pd.DataFrame(y_pred,columns=["x1","x2","x3","x4","x5"])
    df["i"] = 2
    df = df[["i","x1","x2","x3","x4","x5"]]
    np.savetxt(os.path.join(path_to_save, "task2.txt"), df.values, delimiter=';')
    df.to_csv(os.path.join(path_to_save, "task2.csv"))