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

As our previous classifications attempts [1],[2] brought good results, we decided to try these algorithms to validate if the same methodology could be adapted for 5 different types of motions.
For each dimension, we extracted the characteristics fr each trajectory, and separate classifiers were trained for 1D, 2D and 3D subtasks.

### 2.2 Characteristics used for feeding model:
- diffusion coeffitient D
- anomalous exponent
- efficiency
- mean squared displacement ratio
- straightness
- max excursion normalised
- asymmetry (only 2D, 3D)
- fractald imension
- gaussianity
- kurtosis (only 2D, 3D)
- trappedness 
- velocity autocorrelation (only 2D)
- p variation (only 2D)

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
