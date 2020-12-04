# ANDI Challenge contribution
## Patrycja Kowalek
### 4.11.2020

This repository contains the necessary codes written in Python 3 to train the classifiers for Task 2 of the ANDI challenge (see https://competitions.codalab.org/competitions/23601). 


## 1. Required modules 

1. `andi_datasets` for data generation.
2. `math`, `scipy`,`numpy`, `multiprocessing`, `itertools` and `pandas` for data handling and procesing.
3. `sklearn` for basic ML functionalities.
4. `joblib` for storing the classifiers.


## 2. Classification algorithms

### 2.1 Basic assumption

As our previous classifications attempts [1],[2] brought us good results in the classification problem of STP trajectories, we decided to use these algorithms and methods to ANDI challenge.
For each dimension, we extracted the characteristics for each trajectory, and we trained separate classifier for 1D, 2D and 3D subtasks.
Gradient boosting model is built from multiple decision trees. The trees are not independent. The predictions of individual trees are made sequentially by learning from mistakes committed by the ensemble.

### 2.2 Characteristics used for feeding model:
- diffusion coefficient 
- anomalous exponent
- efficiency
- mean squared displacement ratio
- straightness
- max excursion normalised
- asymmetry 
- fractal dimension
- gaussianity
- kurtosis 
- trappedness 
- p variation
- T stat

### 2.3 Algorithms

* Gradient Boosting 

	
## 3. Usage

1. Download the whole repository and extract it in a directory of your choice.

2. If you want to train new classifiers: <br>
a. run _01_generate_data_*D scripts to download data <br>
b. run _04_generate_characteristicts_Andi script to generate characteristics <br>
c. run _05_split_data script to prepare data <br>
d. run _06_generate_gb_hyperparameters script to look for the best fit of hyperparameters <br>
e. run _07_model_generator script for generating a model <br>

3. If you want to get results for new data (classifiers could be found in Models folder):
a. Put your txt data into ValidationDatasets folder <br>
b. Run get_results script <br>


## References

[1] Patrycja Kowalek, Hanna Loch-Olszewska, and Janusz Szwabiński. Classification of diffusion modes in single-particle tracking data: Feature-based versus deep-learning approach. 2019. https://link.aps.org/doi/10.1103/PhysRevE.100.032410

[2] Joanna Janczura, Patrycja Kowalek, Hanna Loch-Olszewska, Janusz Szwabiński, and Aleksander Weron. Classification of particle trajectories in living cells: Machine learning versus statistical testing hypothesis for fractional anomalous diffusion. 2020. https://link.aps.org/doi/10.1103/PhysRevE.102.032402
