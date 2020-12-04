from _08_prepare_validation_data import \
    save_data, prepare_data, marge_results
from _09_generate_characteristics_Validation_Andi import generate_characteristics
from _10_classify import get_classification


if __name__ == "__main__":
    file_name = "trajs.txt"
    save_data(file_name)

    generate_characteristics(characteristics_filename="characteristics.csv", set_name="Set_1D", dim=1)
    generate_characteristics(characteristics_filename="characteristics.csv", set_name="Set_2D", dim=2)
    #generate_characteristics(characteristics_filename="characteristics.csv", set_name="Set_3D", dim=3)

    prepare_data(characteristics_file_name="characteristics.csv", save_file_name="X_data.npy", set_name="Set_1D")
    prepare_data(characteristics_file_name="characteristics.csv", save_file_name="X_data.npy", set_name="Set_2D")
    #prepare_data(characteristics_file_name="characteristics.csv", save_file_name="X_data.npy", set_name="Set_3D")

    get_classification(set_name="Set_1D", res_folder="Res_Set_1D", simulation_folder="Base_corr_Andi_1D")
    get_classification(set_name="Set_2D", res_folder="Res_Set_2D", simulation_folder="Base_corr_Andi_2D")
    #get_classification(set_name="Set_3D", res_folder="Res_Set_3D", simulation_folder="Base_corr_Andi_3D")

    marge_results(res_folder1="Res_Set_1D", res_folder2="Res_Set_2D", res_folder3=None)
