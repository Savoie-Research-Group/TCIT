import sys,os,argparse,subprocess,time,fnmatch
from operator import add
import numpy as np

# Load taffi modules        
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-4])+'/Lib')
import transify as transify
import adjacency as adjacency
import file_parsers as file_parsers
import id_types as id_types
import write_functions as write_functions

# load in rdkit
from rdkit import Chem

# function to return RDCHI
# coding based on 'Design of topological indices. Part 4*. Reciprocal distance matrix, related local vertex invariants and topological indices'
def return_RDCHI(E,G):

    # remove hydrogens
    non_H= [count_i for count_i,i in enumerate(E) if i != 'H']
    E    = [E[ind] for ind in non_H]
    G    = G[non_H]

    # calculate adjacency matrix and distance matrix 
    adj_mat = adjacency.Table_generator(E,G)
    gs   = adjacency.graph_seps(adj_mat)
    Rgs  = np.zeros([len(E),len(E)])

    # calculate reverse distance matrix and bonded atom pairs
    bonds= []
    for i in range(len(E)):
        for j in range(len(E))[i+1:]:

            if adj_mat[i][j] != 0:
                bonds += [[i,j]]

            if gs[i][j] != 0:
                Rgs[i][j] = 1.0/gs[i][j]
                Rgs[j][i] = 1.0/gs[i][j]  

    # calculate reciprocal distance sum
    RDS = []
    for i in range(len(E)):
        RDSi = 0
        for j in range(len(E)):
            RDSi += Rgs[i][j]

        RDS += [RDSi]

    RDSUM = 0.5 * sum(RDS)
    RDSQ  = sum(np.sqrt(RDS[b[0]]*RDS[b[1]]) for b in bonds)
    RDCHI = sum(1.0/np.sqrt(RDS[b[0]]*RDS[b[1]]) for b in bonds)

    return RDCHI
    

# function to return RDCHI
def return_TSPA(smiles):
    
    # load in mordred modules
    from mordred import Calculator, descriptors, TopoPSA

    # create objects
    mol = Chem.MolFromSmiles(smiles)
    calc = Calculator(TopoPSA.TopoPSA)
    
    # calculate TSPA
    result = calc(mol)
    TSPA = result['TopoPSA']

    return TSPA

# function to return number of hydroxyl groups
def return_nROH(smiles):
    
    # assign smarts for alcohols
    hydroxyl    = Chem.MolFromSmarts('[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])]')
    carboxylic  = Chem.MolFromSmarts('[CX3;$([R0][#6]),$([H1R0])](=[OX1])[$([OX2H]),$([OX1-])]')

    #Convert smiles to molecule
    mol  = Chem.MolFromSmiles(smiles)

    # count nROH
    hydroxyl_structs  = mol.GetSubstructMatches(hydroxyl)
    carboxylic_structs= mol.GetSubstructMatches(carboxylic)
    nROH = len(hydroxyl_structs)+len(carboxylic_structs)

    return nROH

# function to determine HSUB
def calc_HSUB(smiles,E=[],G=[]):

    TSPA  = return_TSPA(smiles)
    nROH  = return_nROH(smiles)

    if len(E) == 0:
        success,E,G,q = parse_smiles(smiles)

        if success:
            RDCHI = return_RDCHI(E,G)

        else:
            print("Have trouble parsing smiles...")
            RDCHI = 0.0

    else:
        RDCHI = return_RDCHI(E,G)
    
    # calculate HSUB
    # equ obtained from 'Simple yet accurate prediction method for sublimation enthalpies of organic contaminants using their molecular structure'
    HSUB = 23 + 9.0 * RDCHI ** 2 + 13.0 * nROH + 0.5 * TPSA

    return HSUB

# Function that take smile string and return element and geometry
def parse_smiles(smiles,ff='mmff94',steps=100):
    
    # load in rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        # construct rdkir object
        m = Chem.MolFromSmiles(smiles)
        m2= Chem.AddHs(m)
        AllChem.EmbedMolecule(m2)
        q = 0

        # parse mol file and obtain E & G
        lines = Chem.MolToMolBlock(m2)

        # create a temporary molfile
        tmp_filename = '.tmp.mol'
        with open(tmp_filename,'w') as g:
            g.write(lines)

        # apply force-field optimization
        command = 'obabel {} -O result.xyz --sd --minimize --steps {} --ff {}'.format(tmp_filename,steps,ff)
        os.system(command)
        # parse E and G
        E,G = fp.xyz_parse("result.xyz")

        # Remove the tmp file that was read by obminimize
        try:
            os.remove(tmp_filename)
            os.remove("result.xyz")

        except:
            pass

        return True,E,G,q

    except: 

        return False,[],[],0
