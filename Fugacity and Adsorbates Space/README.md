![image](https://github.com/theOsaroJ/Active-Learning-of-alchemical-adsorption-simulations-towards-a-universal-adsorption-model/assets/64130121/345c5b89-2c40-4b13-9d80-51c2a9b401d9)

This folder is for the chapter where we only navigate the adsorbate space represented by their effective force field potentials (epsilon, sigma, bond-length, and charge), and fugacity.


${\textbf{GP.py}}$ - This is the python file containing the gaussian process regression model. Each MOF in the 1800 MOF database all use the same model.

${\textbf{mlpplot.py}}$ - This is the python file used to compute with the ${R^2}$ between the MLP adsorption and the GP predicted adsorption.

${\textbf{run.sh}}$ - This is the algorithm that carries out the Active Learning process ( see figure above) for each MOF in the mofs.txt file.

${\textbf{\texttt{gas\_loading\_prediction.h5}}}$ - The original Deep learning MLP model used as a surrogate for the GCMC simulations.

${\textbf{model.py}}$ - Python file to make predictions on test arrays with maximum GP MAE during an AL iteration

${\textbf{training.csv}}$ -  Training dataset used to train the original MLP model. This contain the adsorption of 1800 MOFs for different alchemical molecules.
