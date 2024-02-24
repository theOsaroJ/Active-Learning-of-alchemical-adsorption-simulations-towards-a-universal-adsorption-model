
![Final AL procedure](https://github.com/theOsaroJ/Active-Learning-of-alchemical-adsorption-simulations-towards-a-universal-adsorption-model/assets/64130121/49de8cf1-e07b-4a4e-8edd-93c8dc189780)

This folder is for the chapter where we only navigate the adsorbate space represented by their effective force field potentials (epsilon, sigma, bond-length, and charge), and fugacity.


${\textbf{GP.py}}$ - This is the python file containing the gaussian process regression model. Each MOF in the 1800 MOF database all use the same model.

${\textbf{mlpplot.py}}$ - This is the python file used to compute with the ${R^2}$ between the MLP adsorption and the GP predicted adsorption.

${\textbf{run.sh}}$ - This is the algorithm that carries out the Active Learning process ( see figure above) for each MOF in the mofs.txt file.

${\textbf{\texttt{gasloadingprediction.h5}}}$ - The original Deep learning MLP model used as a surrogate for the GCMC simulations.

${\textbf{model.py}}$ - Python file to make predictions on test arrays with maximum GP MAE during an AL iteration

${\textbf{training.csv}}$ -  Training dataset used to train the original MLP model. This contain the adsorption of 1800 MOFs for different alchemical molecules.

Steps:
1. Have all the MOFs in mof.txt
2. Run uge.sh to execute run.sh based on computational resources to be used
3. Make sure to have mlpData.csv and DL_AL.csv in each 'MOF'/RQ directory. mlpData.csv is the MOF functional group densities, structural properties and alchemical adsorption values for each MOF. This is generated using the MLP model for all the test data to be navigated via the GPR. An example of the mlpData.csv for OPT_acs_sym_7_mc_4_1B_2Br is in this directory as mlpData.csv. The DL_AL.csv file contains the mof functional group densities and structural properties, with placeholders to be replaced by the fugacity, bond_length, charge, epsilon and sigma test array of the highest GP absoluter error.


The folder called NeuralNetwork contains a zip file that contains the New MLP model and the compiled training data collected from the 1800 MOFs models.
