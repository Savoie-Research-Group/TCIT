TCIT-Hf
TCIT, the short of Taffi component increment theory, is a powerful tool to predict thermochemistry properties, like enthalpy of formation.

This script implemented TCIT which performs on a given folder of target compounds based on a fixed TCIT CAV database distributed with the paper "A Self-Consistent Component Increment Theory for Predicting Enthalpy of Formation" by Zhao and Savoie. Further ring correction is added distributed with the paper "Transferable Ring Corrections for Predicting Enthalpy of Formation of Cyclic Compounds" by Zhao, Iovanac and Savoie. The heat of vaporization calculation will use the NIST value if available, otherwise it will be computed by the SIMPOL1 model ("Vapor pressure prediction – simple group contribution method"). Similarly, the heat of sublimation comes from "Simple yet accurate prediction method for sublimation enthalpies of organic contaminants using their molecular structure"

The script operates on either a folder of xyz files or a list of smiles strings, prints out the Taffi components and corresponding CAVs that are used for each prediction, and returns the 0K and 298K enthalpy of formation as well as enthalpy of formation of liquid phase and solid phase.

Software requirement:
-openbabel 2.4.1 or higher
-anaconda

Set up an environment if needed
conda create -n TCIT -c conda-forge python=3.7 rdkit
source activate TCIT
pip install alfabet
pip install mordred

Other necessary packages may include:
tensorflow (tested with 1.15)
matplotlib (tested with 3.5.3)
keras (tested with 2.3.1)
scikit-learn (tested with 0.24.2)
rdkit (listed above, tested with 2021.03.4, others may be fine)

A yaml file of the testing environment is also included.

Usage:
The TCIT script and its config file are in TCIT_Scripts/TCIT

If your input type a xyz file:

Put xyz files of the compounds with research interest in one folder (default: input_xyz)
Type "python TCIT.py -h" for help if you want specify the database files and run this program.

If your input type is smiles string:

Make a list of smiles string (default: input_list/test_inp_public.txt)
Type "python TCIT.py" for help if you want specify the database files and run this program.


Notes
Make sure the bin folder of openbabel is added in the environment setting, or 'alias obabel=' to that bin folder. Check by running 'obabel -H'.
Currently TCIT just works for Linux and MacOS.
For assistance, reach out to Tyler Pasut (tpasut@purdue.edu)
24JUN2024
