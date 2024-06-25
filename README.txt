TCIT-Hf
TAFFI Component Increment Theory, or TCIT, is a powerful method for predicting the thermochemical properties of molecules that are component decomposable, such as enthalpy of formation.

This script implements TCIT which operates on a given folder of target compounds based on a fixed TCIT CAV database distributed with the paper "A Self-Consistent Component Increment Theory for Predicting Enthalpy of Formation" by Zhao and Savoie. Further ring correction is distributed with the paper "Transferable Ring Corrections for Predicting Enthalpy of Formation of Cyclic Compounds" by Zhao, Iovanac and Savoie. The heat of vaporization calculation will use the NIST value if available, otherwise it will be computed by the SIMPOL1 model ("Vapor pressure prediction – simple group contribution method"). Similarly, the heat of sublimation comes from "Simple yet accurate prediction method for sublimation enthalpies of organic contaminants using their molecular structure"

The script operates on either a folder of xyz files or a list of SMILES strings, prints out the Taffi components and corresponding CAVs that are used for each prediction, and returns the 0K and 298K enthalpy of formation as well as (separately) enthalpy of formation of liquid phase and solid phase.

Software requirement:
-openbabel 2.4.1 or higher
-anaconda

Set up an environment if needed
conda create -n TCIT -c conda-forge python=3.7 rdkit
source activate TCIT
pip install mordred (used for Hvap and Hsub)

Other necessary packages may include:
tensorflow (tested with 1.15)
matplotlib (tested with 3.5.3)
keras (tested with 2.3.1)
scikit-learn (tested with 0.24.2)
rdkit (listed above, tested with 2021.03.4, others may be fine)
openbabel (can be called by a binary or through Python if installed)

A yaml file of the testing environment is also included.

Usage:
The TCIT script and its config file are in TCIT_Scripts/TCIT

If your input type is a xyz file:

Put xyz files of the compounds with research interest in one folder (default: input/test_xyz)
Type "python TCIT.py -h" for help if you want specify the database files and run this program or "python TCIT.py" to run.

If your input type is smiles string:

Make a list of smiles string (default: input/test_inp_public.txt)
Type "python TCIT.py -h" for help if you want specify the database files and run this program or "python TCIT.py" to run.


Notes
The config file specifies the path to the databases used. This should go to the Public_Database folder distributed with this program.
Make sure the bin folder of openbabel is added in the environment setting, or 'alias obabel=' to that bin folder. Check by running 'obabel -H'.
Currently TCIT just works for Linux and MacOS.
For assistance, reach out to Tyler Pasut (tpasut@purdue.edu)
25JUN2024
