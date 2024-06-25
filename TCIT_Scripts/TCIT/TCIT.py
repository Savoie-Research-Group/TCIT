# This is file is write to 
# 1. Indentify whether this is a minimal structure or not, if so, append it into database
# 2. Use Taffi component increment theory to get the prediction of following properties:
#    A. Enthalpy of formation of gas at 0k and 298k, 1 atom pressure
#    B. Entropy of gas at standard conditions
#    C. Gibbs free energy of formation of gas at standard conditions 
#    D. Constant volumn heat capacity of gas (and constant pressure)
# Author: Qiyuan Zhao, Nicolae C. Iovanac (Based on taffi)
# Public version created by Tyler Pasut

def warn(*args,**kwargs): #Keras spits out a bunch of junk
    pass
import warnings
warnings.warn = warn

import sys,os,argparse,subprocess,json
import numpy as np
from fnmatch import fnmatch
import tensorflow as tf
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Tensorflow also spits out a bunch of junk
np.random.seed(0)
tf.compat.v2.random.set_seed(0)
random.seed(0)

# Add TAFFY Lib to path and import taffi modules 
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-3])+'/Lib')
import adjacency as adj
import file_parsers as fp
from id_types import id_types

# Import frag_gen_inter and ring related functions
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-3])+'/TCIT_Scripts')
from frag_gen_Taffi import minimal_structure
from deal_ring import gen_RCMCs,return_smi,return_inchikey

# Import machine leaning module
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/ML-package')
import preprocess

def main(argv):

    parser = argparse.ArgumentParser(description='Calculate Taffi structure CAVs and make prediction')

    #optional arguments                                                                                                                   
    parser.add_argument('-c', dest='config', default='config.txt',
                        help = 'The program expects a configuration file from which to assign various run conditions. (default: config.txt) in the current working directory)')

    parser.add_argument('-o', dest='outputname', default='result',
                        help = 'Controls the output folder name for the result')


    # parse configuration dictionary (c)                                                                                                   
    print("parsing calculation configuration...")
    args=parser.parse_args()    
    c = parse_configuration(parser.parse_args())
    
    ##################################################                                                                                    

    # Initializing dictionary

    ##################################################
    kj2kcal = 0.239006

    # load database
    append_flag = False
    CAV_dict = parse_CAV_database(c["database"])    
    G4_dict  = parse_G4_database(c["g4_db"])
    ring_dict= parse_ringcorr(c["ring_db"])
    if os.path.isfile(c["tcit_db"]) is True:
        with open(c["tcit_db"],"r") as f:
            TCIT_dict= json.load(f) 

    # create similarity match dictionary
    similar_match = {}
    for inchi in G4_dict.keys():
        similar_match[inchi[:14]]=inchi

    sys.stdout = Logger(args.outputname)

    # load in ML model
    base_model = getModel()

    # identify the input type
    if c["input_type"] == 'xyz':
        items = sorted(c["target_xyz"])
    else:
        items = c["target_smiles"]

    # loop for target xyz files
    for i in items:

        # get the output name
        if c["input_type"] == 'xyz':
            name = i.split('/')[-1]
        else:
            name = i

        print("Working on {}...".format(name))

        # set default values
        flag = True
        ring_flag = False

        # parse E and G
        if c["input_type"] == 'xyz':
            E,G = fp.xyz_parse(i)
            q   = fp.parse_q(i) 
            adj_mat = adj.Table_generator(E,G)
            readin = True
            smiles = return_smi(E,G,adj_mat)

        else:
            readin,E,G,q = parse_smiles(i)
            adj_mat = adj.Table_generator(E,G)
            smiles = i

        # check if the item already in the TCIT result
        if smiles in TCIT_dict.keys():
            print("Hf_0 for {} is {} kJ/mol".format(name, TCIT_dict[smiles]["Hf_0"]))
            print("Hf_298 for {} is {} kJ/mol".format(name, TCIT_dict[smiles]["Hf_298"]))
            print("Gf_298 for {} is {} kJ/mol".format(name, TCIT_dict[smiles]["Gf_298"]))
            print("S0_gas for {} is {} J/(mol*K)".format(name, TCIT_dict[smiles]["S0"]))
            print("Cv_gas for {} is {} J/(mol*K)".format(name, TCIT_dict[smiles]["Cv"]))
            continue

        # obtain number of rotatable bonds for Cv correction (in kJ/mol)
        append_flag = True
        Nrot = return_Nrot(E,G)
        Cv_corr = 1.95 * Nrot + 0.84
        #Cv_corr = 0

        if readin is False:
            print("Invalid smiles string, skip this compounds...")
            continue
            
        if True in [element.lower() not in ['h','b','c','n','o','f','s','cl','br','p','i'] for element in E]:
            print("can't deal with some element in this compounds")
            continue

        try:
            lone_electrons,bonding_electrons,core_electrons,bond_mat,fc = adj.find_lewis(E,adj_mat,q_tot=q,keep_lone=[],return_pref=False,return_FC=True)
            continue_flag = True

        except:
            print("Have error determine lewis structure for {}, skip...".format(name))
            continue_flag = False

        if continue_flag is False:
            continue

        keep_lone = [ [ count_j for count_j,j in enumerate(lone_electron) if j%2 != 0] for lone_electron in lone_electrons]
        atom_types = id_types(E,adj_mat,gens=2,fc=fc,keep_lone=keep_lone)
        # replace "R" group by normal group
        atom_types = [atom_type.replace('R','') for atom_type in atom_types]

        # identify ring structure
        ring_inds = [adj.ring_atom(adj_mat,j) for j,Ej in enumerate(E)] 

        # initialize ring corrections
        ring_corr_Hf0=0
        ring_corr_S0 =0
        ring_corr_Cv =0
        ring_corr_Hf298=0
        ring_flag = True

        # don't consider rings
        #ring_inds = [] 

        # don't consider ions and radicals
        #if len(keep_lone[0]) != 0 or max(np.abs(fc[0])) > 0:
            #print("Contains radicals/charges")
            #continue

        if True in ring_inds: # add ring correction to final prediction
            
            try:
                RC0,RC2=gen_RCMCs(E,G,gens=2,return_R0=True) 
                exit()
            except:
                print("ring generation fails for {}, check the geometry of given xyz file".format(name))

            if len(RC0.keys()) > 0:
                print("Identify rings! Add ring correction to final prediction")
                ring_flag = True
                for key in RC0.keys():
                    depth0_ring=RC0[key]
                    depth2_ring=RC2[key] 

                    NonH_E = [ele for ele in E if ele is not 'H']
                    ring_NonH_E = [ele for ele in depth2_ring["elements"] if ele is not 'H']

                    if float(depth0_ring["hash_index"]) in ring_dict.keys() and len(depth2_ring["ringsides"]) == 0:

                        print("\n{} can not use TCIT to calculate, the result comes from G4 result".format(name))
                        RC_Hf0K   = ring_dict[float(depth0_ring["hash_index"])]["HF_0"]  
                        RC_Hf298  = ring_dict[float(depth0_ring["hash_index"])]["HF_298"]
                        RC_S0     = ring_dict[float(depth0_ring["hash_index"])]["S_0"]   
                        RC_Cv     = ring_dict[float(depth0_ring["hash_index"])]["Cv"]    

                        ring_corr_Hf0  +=RC_Hf0K 
                        ring_corr_Hf298+=RC_Hf298
                        ring_corr_S0   +=RC_S0 
                        ring_corr_Cv   +=RC_Cv 
                        
                        print("Add ring correction {}: {:< 6.3f} kcal/mole into final prediction".format(depth0_ring["hash_index"],RC_Hf298))

                    elif float(depth0_ring["hash_index"]) in ring_dict.keys() and len(depth2_ring["ringsides"]) > 0:

                        # predict difference of Hf_0k
                        weights_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])+'/ML-package'
                        base_model.load_weights(weights_path+'/Hf_0k.h5')
                        diff_Hf0K = getPrediction([depth2_ring["smiles"]],[depth0_ring["smiles"]],base_model)

                        # predict difference of Hf_298k
                        base_model.load_weights(weights_path+'/Hf_298k.h5')
                        diff_Hf298 = getPrediction([depth2_ring["smiles"]],[depth0_ring["smiles"]],base_model)

                        # predict difference of S0
                        base_model.load_weights(weights_path+'/S0_298k.h5')
                        diff_S0 = getPrediction([depth2_ring["smiles"]],[depth0_ring["smiles"]],base_model)

                        # predict difference of Cv
                        base_model.load_weights(weights_path+'/Cv.h5')
                        diff_Cv = getPrediction([depth2_ring["smiles"]],[depth0_ring["smiles"]],base_model)

                        # calculate RC based on the differences
                        RC_Hf0K   = ring_dict[float(depth0_ring["hash_index"])]["HF_0"]   + diff_Hf0K
                        RC_Hf298  = ring_dict[float(depth0_ring["hash_index"])]["HF_298"] + diff_Hf298
                        RC_S0     = ring_dict[float(depth0_ring["hash_index"])]["S_0"]    + diff_S0
                        RC_Cv     = ring_dict[float(depth0_ring["hash_index"])]["Cv"]     + diff_Cv

                        ring_corr_Hf0  +=RC_Hf0K 
                        ring_corr_Hf298+=RC_Hf298
                        ring_corr_S0   +=RC_S0 
                        ring_corr_Cv   +=RC_Cv 
                        
                        print("Add ring correction {}: {:< 6.3f} kcal/mole into final prediction (based on depth=0 ring {}:{: < 6.3f})".format(depth2_ring["hash_index"],RC_Hf298,depth0_ring["hash_index"],\
                                                                                                                                               ring_dict[float(depth0_ring["hash_index"])]["HF_298"]))
                    else:
                        print("Information of ring {} is missing, the final prediction might be not accurate, please update ring_correction database first".format(depth0_ring["hash_index"]))
                        ring_flag = False
                        write_xyz(os.getcwd()+"/missing_rings/ring_{}".format(float(depth0_ring["hash_index"])),depth0_ring["elements"],depth0_ring["geometry"])


            else: 
                print("Identify rings, but the heavy atom number in the ring is greater than 12, ignore the ring correction")
        
        # collect ring correction as a dictionary
        ring_corr = {}
        ring_corr["HF_0"]  = ring_corr_Hf0
        ring_corr["HF298"] = ring_corr_Hf298
        ring_corr["S0"]    = ring_corr_S0
        ring_corr["Cv"]    = ring_corr_Cv

        # remove terminal atoms                                                                                                           
        B_inds = [count_j for count_j,j in enumerate(adj_mat) if sum(j) > 1 ]
        
        # Apply pedley's constrain 1                                                                                                      
        H_inds = [count_j for count_j in range(len(E)) if E[count_j] == "H" ]
        P1_inds = [ count_j for count_j,j in enumerate(adj_mat) if E[count_j] == "C" and len([ count_k for count_k,k in enumerate(adj_mat[count_j,:]) if k == 1 and count_k not in H_inds ]) == 1 ]

        # determine compounent types and find unknown ones
        group_types = [atom_types[Bind] for Bind in B_inds if Bind not in P1_inds]
        Unknown = [j for j in group_types if j not in CAV_dict.keys()]

        # indentify whether this is a minimal structure or not
        min_types = [ j for j in group_types if minimal_structure(j,G,E,q_tot=q,gens=2) is True ]
        
        if ring_flag is False: # deal with linear structure

            if len(min_types) > 0:
                print("\n{} can not use TCIT to calculate, the result comes from G4 result".format(name))

            if len(group_types) < 2: 
                inchikey = return_inchikey(E,G,adj.Table_generator(E,G))
                if inchikey not in G4_dict.keys() and inchikey[:14] in similar_match.keys():
                    inchikey = similar_match[inchikey[:14]]
            
                # Look up G4 database 
                if inchikey in G4_dict.keys():
                    S0     = G4_dict[inchikey]["S0"]
                    Cv     = G4_dict[inchikey]["Cv"]
                    Hf_0   = G4_dict[inchikey]["HF_0"]
                    Hf_298 = G4_dict[inchikey]["HF_298"]
                    Gf_298 = G4_dict[inchikey]["GF_298"]
            
                    print("Hf_0 for {} is {:.3f} kJ/mol".format(name, Hf_0/kj2kcal))
                    print("Hf_298 for {} is {:.3f} kJ/mol".format(name, Hf_298/kj2kcal))
                    print("Gf_298 for {} is {:.3f} kJ/mol".format(name, Gf_298/kj2kcal))
                    print("S0_gas for {} is {:.3f} J/(mol*K)".format(name, S0/kj2kcal))
                    print("Cv_gas for {} is {:.3f} J/(mol*K)".format(name, Cv/kj2kcal + Cv_corr))
                    pred = {"Hf_0":float("{:.3f}".format(Hf_0/kj2kcal)),"Hf_298":float("{:.3f}".format(Hf_298/kj2kcal)),"Gf_298":float("{:.3f}".format(Gf_298/kj2kcal)),\
                            "S0":float("{:.3f}".format(S0/kj2kcal)),"Cv":float("{:.3f}".format(Cv/kj2kcal + Cv_corr))}
                    TCIT_dict[smiles] = pred 

                    '''
                    # also append such data into TCIT CAVs db
                    for j in P1_inds:
                        NH = len([ind_j for ind_j,adj_j in enumerate(adj_mat[j,:]) if adj_j == 1 and ind_j in H_inds])

                        if NH == 3:
                            Hf_0   -= CAV_dict["[6[6[1][1][1]][1][1][1]]"]["HF_0"]
                            S0     -= CAV_dict["[6[6[1][1][1]][1][1][1]]"]["S0"]
                            Cv     -= CAV_dict["[6[6[1][1][1]][1][1][1]]"]["Cv"]
                            Hf_298 -= CAV_dict["[6[6[1][1][1]][1][1][1]]"]["HF_298"]
                            Gf_298 -= CAV_dict["[6[6[1][1][1]][1][1][1]]"]["GF_298"]

                        elif NH == 2:
                            Hf_0   -= CAV_dict["[6[6[1][1]][1][1]]"]["HF_0"]
                            S0     -= CAV_dict["[6[6[1][1]][1][1]]"]["S0"]
                            Cv     -= CAV_dict["[6[6[1][1]][1][1]]"]["Cv"]
                            Hf_298 -= CAV_dict["[6[6[1][1]][1][1]]"]["HF_298"]
                            Gf_298 -= CAV_dict["[6[6[1][1]][1][1]]"]["GF_298"]

                        elif NH == 1:
                            Hf_0   -= CAV_dict["[6[6[1]][1]]"]["HF_0"]
                            S0     -= CAV_dict["[6[6[1]][1]]"]["S0"]
                            Cv     -= CAV_dict["[6[6[1]][1]]"]["Cv"]
                            Hf_298 -= CAV_dict["[6[6[1]][1]]"]["HF_298"]
                            Gf_298 -= CAV_dict["[6[6[1]][1]]"]["GF_298"]

                        else:
                            print("Error, so such group in Constrain 1, when dealing with Compound {}".format(m))
                    
                    if len(group_types) == 0:
                        continue

                    if group_types[0] in CAV_dict.keys():
                        continue

                    print("Write information of {} into database".format(group_types[0]))
                    CAV_dict[group_types[0]] = {}
                    CAV_dict[group_types[0]]["HF_0"]  = Hf_0
                    CAV_dict[group_types[0]]["S0"]    = S0
                    CAV_dict[group_types[0]]["Cv"]    = Cv
                    CAV_dict[group_types[0]]["HF_298"]= Hf_298
                    CAV_dict[group_types[0]]["GF_298"]= Gf_298
                    
                    if len(P1_inds) == 0:
                        with open(c["database"],"a") as f:
                            f.write("{:<60s} {:< 20.8f} {:< 20.8f} {:< 20.8f}  {:< 10.5f}  {:< 15.5f}  {:<10s}  {:<20s}\n".format\
                                    (group_types[0],Hf_0,Hf_298,Gf_298,Cv,S0,"None",inchikey))

                    
                    else:
                        with open(c["database"],"a") as f:
                            f.write("{:<60s} {:< 20.8f} {:< 20.8f} {:< 20.8f}  {:< 10.5f}  {:< 15.5f}  {:<10s}  {:<20s}\n".format\
                                    (group_types[0],Hf_0,Hf_298,Gf_298,Cv,S0,"C1",inchikey))
                    '''
                else:                    
                    write_xyz("{}/{}".format(c["xyz_task"],inchikey),E,G,q)
                    print("\n Put {} in xyz_task folder, wait for G4 calculation...".format(inchikey))

            elif len(Unknown) == 0: 
            
                print("\n"+"="*120)
                print("="*120)
                print("\nNo more information is needed, begin to calculate enthalpy of fomation of {}".format(name))
                pred = calculate_CAV(E,G,q,name,CAV_dict,ring_corr,Cv_corr)
                TCIT_dict[smiles] = pred
                
            elif len(Unknown) > 0:
                print("\n"+"="*120)
                print("="*120)
                print("\nInformation of {} is missing, contact Tyler Pasut with the Savoie Research Group".format(Unknown))

        # deal with ring structure        
        else: 

            if len(Unknown) == 0:
                print("\n"+"="*120)
                print("="*120)
                print("\nNo more information is needed, begin to calculate enthalpy of fomation of {}".format(name))
                pred = calculate_CAV(E,G,q,name,CAV_dict,ring_corr,Cv_corr)
                TCIT_dict[smiles] = pred
                
            else:
                print("\n"+"="*120)
                print("="*120)
                print("\nInformation of {} is missing, contact Tyler Pasut with the Savoie Research Group".format(Unknown))

        # if ring flag is False, remove this item from dict
        if not ring_flag and smiles in TCIT_dict.keys(): del(TCIT_dict[smiles])

    # append data into TCIT dict
    if append_flag:
        with open(c["tcit_db"],"w") as f:
            json.dump(TCIT_dict, f)

    # run G4 jobs is there are missing CAVs
    if len(os.listdir(c["xyz_task"])) > 0:
        print("Please first perform G4 calculations for missing model compounds and then run this file again")

    return

# Fucntion to calculate properties based on known CAVs
def calculate_CAV(E,G,q,name,CAV_dict,ring_corr,Cv_corr=0):

    # Tabulated  absolute entropy of element in its standard reference state (in J/mol/K)
    # (http://www.codata.info/resources/databases/key1.html)
    S_atom_298k = {"H":65.34, "Li": 29.12, "Be": 9.50, "B": 5.90, "C":5.74, "N":95.805 , "O": 102.58, "F": 101.396, "Na": 51.30, "Mg": 32.67, "Al": 28.30, \
                   "Si": 18.81, "P": 18.81, "S": 32.054, "Cl": 111.54, "Br": 76.11}

    # define j to cal
    kj2kcal = 0.239006
    
    # parse E and G
    adj_mat = adj.Table_generator(E,G)
    lone_electrons,bonding_electrons,core_electrons,bond_mat,fc = adj.find_lewis(E,adj_mat,q_tot=q,keep_lone=[],return_pref=False,return_FC=True)
    keep_lone = [ [ count_i for count_i,i in enumerate(lone_electron) if i%2 != 0] for lone_electron in lone_electrons]
    atom_types = id_types(E,adj_mat,gens=2,fc=fc,keep_lone=keep_lone)

    # replace "R" group by normal group
    atom_types = [atom_type.replace('R','') for atom_type in atom_types]

    # remove terminal atoms                                                                                                           
    B_inds = [count_j for count_j,j in enumerate(adj_mat) if sum(j)>1 ]

    # Apply pedley's constrain 1                                                                                                      
    H_inds = [count_j for count_j in range(len(E)) if E[count_j] == "H" ]
    P1_inds = [ count_j for count_j,j in enumerate(adj_mat) if E[count_j] == "C" and\
                len([ count_k for count_k,k in enumerate(adj_mat[count_j,:]) if k == 1 and count_k not in H_inds ]) == 1 ]
    group_types = [atom_types[Bind] for Bind in B_inds if Bind not in P1_inds]

    # initialize with ring correction 
    Hf_target_0  = ring_corr["HF_0"]
    Hf_target_298= ring_corr["HF298"]
    S0_target    = ring_corr["S0"]
    Cv_target    = ring_corr["Cv"]

    for j in group_types:
        S0_target  += CAV_dict[j]["S0"]
        Cv_target  += CAV_dict[j]["Cv"]
        Hf_target_0   += CAV_dict[j]["HF_0"]
        Hf_target_298 += CAV_dict[j]["HF_298"]

    for j in P1_inds:
        NH = len([ind_j for ind_j,adj_j in enumerate(adj_mat[j,:]) if adj_j == 1 and ind_j in H_inds])
        if NH == 3:
            Hf_target_0   += CAV_dict["[6[6[1][1][1]][1][1][1]]"]["HF_0"]
            S0_target     += CAV_dict["[6[6[1][1][1]][1][1][1]]"]["S0"]
            Cv_target     += CAV_dict["[6[6[1][1][1]][1][1][1]]"]["Cv"]
            Hf_target_298 += CAV_dict["[6[6[1][1][1]][1][1][1]]"]["HF_298"]

        elif NH == 2:
            Hf_target_0   += CAV_dict["[6[6[1][1]][1][1]]"]["HF_0"]
            S0_target     += CAV_dict["[6[6[1][1]][1][1]]"]["S0"]
            Cv_target     += CAV_dict["[6[6[1][1]][1][1]]"]["Cv"]
            Hf_target_298 += CAV_dict["[6[6[1][1]][1][1]]"]["HF_298"]

        elif NH == 1:
            Hf_target_0   += CAV_dict["[6[6[1]][1]]"]["HF_0"]
            S0_target     += CAV_dict["[6[6[1]][1]]"]["S0"]
            Cv_target     += CAV_dict["[6[6[1]][1]]"]["Cv"]
            Hf_target_298 += CAV_dict["[6[6[1]][1]]"]["HF_298"]

        else:
            print("Error, no such NH = {} in Constrain 1".format(NH))
            print("{} shouldn't appear here".format([atom_types[Pind] for Pind in P1_inds]))
            quit()
    
    # evaluate Gf based on Hf and S
    S_atom = kj2kcal * sum([ S_atom_298k[_] for _ in E ])
    Gf_target_298 = Hf_target_298 - 298.15 * (S0_target - S_atom) / 1000.0   
    print("Prediction are made based on such group types: (Hf_298k/S0_298k)")
    for j in group_types:
        print("{:30s}: {:<5.2f}/{:<5.2f}".format(j,CAV_dict[j]["HF_298"],CAV_dict[j]["S0"]))

    print("Prediction of Hf_0 for {} is {:.3f} kJ/mol".format(name, Hf_target_0/kj2kcal))
    print("Prediction of Hf_298 for {} is {:.3f} kJ/mol".format(name, Hf_target_298/kj2kcal))
    print("Prediction of Gf_298 for {} is {:.3f} kJ/mol".format(name, Gf_target_298/kj2kcal))
    print("Prediction of S0_gas for {} is {:.3f} J/(mol*K)".format(name, S0_target/kj2kcal))
    print("Prediction of Cv_gas for {} is {:.3f} J/(mol*K)".format(name, Cv_target/kj2kcal + Cv_corr))

    pred = {"Hf_0":float("{:.3f}".format(Hf_target_0/kj2kcal)),"Hf_298":float("{:.3f}".format(Hf_target_298/kj2kcal)),"Gf_298":float("{:.3f}".format(Gf_target_298/kj2kcal)),\
            "S0":float("{:.3f}".format(S0_target/kj2kcal)),"Cv":float("{:.3f}".format(Cv_target/kj2kcal + Cv_corr))}

    return pred

# Function that take smile string and return element and geometry
def parse_smiles(smiles,steps=500):
    
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
        try:
            command = 'obabel {} -O result.xyz --sd --minimize --steps {} --ff mmff94'.format(tmp_filename,steps)
            os.system(command)

            # parse E and G
            E,G = fp.xyz_parse("result.xyz")

        except:
            command = 'obabel {} -O result.xyz --sd --minimize --steps {} --ff uff'.format(tmp_filename,steps)
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

# write a xyz file
def write_xyz(Output,Elements,Geometry,charge=0):
    
    # Open file for writing and write header
    fid = open(Output+'.xyz','w')
    fid.write('{}\n'.format(len(Elements)))
    fid.write('q {}\n'.format(charge))
    for count_i,i in enumerate(Elements):
        fid.write('{: <4} {:< 12.6f} {:< 12.6f} {}\n'.format(i,Geometry[count_i,0],Geometry[count_i,1],Geometry[count_i,2]))

    fid.close()

# load in G4 database
def parse_G4_database(db_files,G4_dict={}):
    with open(db_files,'r') as f:
        for lines in f:
            fields = lines.split()
            if len(fields) ==0: continue
            if fields[0] == '#': continue
            if len(fields) >= 7:
                if fields[0] not in G4_dict.keys():
                    G4_dict[fields[0]] = {}
                    G4_dict[fields[0]]["smiles"] = fields[1]
                    G4_dict[fields[0]]["HF_0"]   = float(fields[2])
                    G4_dict[fields[0]]["HF_298"] = float(fields[3])
                    G4_dict[fields[0]]["S0"]     = float(fields[4])
                    G4_dict[fields[0]]["GF_298"] = float(fields[5])
                    G4_dict[fields[0]]["Cv"]     = float(fields[6])

    return G4_dict

# load in TCIT CAV database
def parse_CAV_database(db_files,CAV_dict={}):
    with open(db_files,'r') as f:
        for lines in f:
            fields = lines.split()
            if len(fields) ==0: continue
            if fields[0] == '#': continue
            if len(fields) == 8 and fields[0] not in CAV_dict.keys():
                CAV_dict[fields[0]]  = {}
                CAV_dict[fields[0]]["HF_0"]     = float(fields[1])
                CAV_dict[fields[0]]["HF_298"]   = float(fields[2])
                CAV_dict[fields[0]]["GF_298"]   = float(fields[3])
                CAV_dict[fields[0]]["Cv"]       = float(fields[4])
                CAV_dict[fields[0]]["S0"]       = float(fields[5])
                
    return CAV_dict

# load in ring correction database
def parse_ringcorr(db_file,RC_dict={}):
    with open(db_file,'r') as f:
        for lines in f:
            fields = lines.split()
            if len(fields) ==0: continue
            if fields[0] == '#': continue 
            if len(fields) >= 7:
                RC_dict[float(fields[0])] = {}
                RC_dict[float(fields[0])]["HF_0"]   = float(fields[1])
                RC_dict[float(fields[0])]["HF_298"] = float(fields[2])
                RC_dict[float(fields[0])]["GF_298"] = float(fields[3])
                RC_dict[float(fields[0])]["Cv"]     = float(fields[4])
                RC_dict[float(fields[0])]["S_0"]    = float(fields[5])

    return RC_dict

# load in config
def parse_configuration(args):

    # Convert inputs to the proper data type
    if os.path.isfile(args.config) is False:
        print("ERROR in python_driver: the configuration file {} does not exist.".format(args.config))
        quit()

    # Process configuration file for keywords
    keywords = ["input_type","target_path", "target_file", "database", "G4_db", "TCIT_db", "ring_db", "xyz_task"]
    keywords = [ _.lower() for _ in keywords ]

    list_delimiters = [ "," ]  # values containing any delimiters in this list will be split into lists based on the delimiter
    space_delimiters = [ "&" ] # values containing any delimiters in this list will be turned into strings with spaces replacing delimites
    configs = { i:None for i in keywords }    

    with open(args.config,'r') as f:
        for lines in f:
            fields = lines.split()
            
            # Delete comments
            if "#" in fields:
                del fields[fields.index("#"):]
            
            # Parse keywords
            l_fields = [ _.lower() for _ in fields ] 
 
            for i in keywords:
                if i in l_fields:

                    # Parse keyword value pair
                    ind = l_fields.index(i) + 1
                    if len(fields) >= ind + 1:
                        configs[i] = fields[ind]

                        # Handle delimiter parsing of lists
                        for j in space_delimiters:
                            if j in configs[i]:
                                configs[i] = " ".join([ _ for _ in configs[i].split(j) ])
                        for j in list_delimiters:
                            if j in configs[i]:
                                configs[i] = configs[i].split(j)
                                break
                                
                    # Break if keyword is encountered in a non-comment token without an argument
                    else:
                        print("ERROR in python_driver: enountered a keyword ({}) without an argument.".format(i))
                        quit()

    # Set defaults if None
    if configs["input_type"] is None:
        configs["input_type"] == "xyz"

    elif configs["input_type"].lower() not in ["smiles","xyz"]:
        print("Warning! input_type must be either xyz or smiles, use default xyz...")
        configs["input_type"] == "xyz"

    if str(configs["target_path"]).lower() == "none":
        configs["target_path"] = None    

    # Makesure detabase folder exits
    if configs["input_type"] == "xyz" and os.path.isdir(configs["target_path"]) is False:
        print("No input target files")
        quit()

    if configs["input_type"] == "smiles" and os.path.isfile(configs["target_file"]) is False:
        print("No input target files")
        quit()
        
    if os.path.isfile(configs["database"]) is False: 
        print("No such data base")
        quit()
            
    if os.path.isfile(configs["ring_db"]) is False:
        print("No such ring correction database, please check config file, existing...")
        quit()

    if os.path.isfile(configs["g4_db"]) is False:
        print("No such G4 result database, please check config file, existing...")
        quit()

    if len(os.listdir(configs["xyz_task"]) ) > 0:
        subprocess.Popen("rm {}/*".format(configs["xyz_task"]),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]
   
    # files in need of running 
    if configs["input_type"] == 'xyz':
        if configs["target_path"] is None:
            configs["target_xyz"]=[os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames if (fnmatch(f,"*.xyz"))]

        else:
            configs["target_xyz"]=[os.path.join(dp, f) for dp, dn, filenames in os.walk(configs["target_path"]) for f in filenames if (fnmatch(f,"*.xyz"))]

    else:
        configs["target_smiles"] = []
        with open(configs["target_file"],"r") as g:
            for line in g:
                configs["target_smiles"].append(line.strip())

    return configs

# function to determine number of rotatable bonds
def return_Nrot(E,G):
    
    write_xyz("Nrot_input",E,G)
    # use obconformer to determine the functional group
    substring = "obconformer 1 1 Nrot_input.xyz"
    output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[1] 
    output = output.decode('utf-8').split('\n')
    Nrot=0
    for line in output:
        fields = line.split()
        if len(fields) == 5 and fields[0] == 'NUMBER' and fields[2] == 'ROTATABLE':
            try:
                Nrot = int(fields[-1])
            except:
                print("Have error using obconformer to determine N_rot, Nrot = 0")
                
    # Remove the tmp file that was read by obconformer
    try:
        os.remove('Nrot_input.xyz')
    except:
        pass

    return Nrot

#getModel and getPrediction are the two main functions. Build the model and load the parameters,
def getModel():
    params = {
        'n_layers'  :3,
        'n_nodes'   :256,
        'fp_length' :256,
        'fp_depth'  :3,
        'conv_width':30,
        'L2_reg'    :0.0004, #The rest of the parameters don't really do anything outside of training
        'batch_normalization':1,
        'learning_rate':1e-4,
        'input_shape':2
    }

    from gcnn_model import build_fp_model
    
    predictor_MLP_layers = []
    for l in range(params['n_layers']):
        predictor_MLP_layers.append(params['n_nodes'])

    model = build_fp_model(
        fp_length = params['fp_length'],
        fp_depth = params['fp_depth'],
        conv_width=params['conv_width'],
        predictor_MLP_layers=predictor_MLP_layers,
        L2_reg=params['L2_reg'],
        batch_normalization=params['batch_normalization'],
        lr = params['learning_rate'],
        input_size = params['input_shape']
    )

    #model.load_weights(weights)
    return model
    
def getPrediction(smiles,R0_smiles,model):
    X_eval = (smiles,R0_smiles)
    processed_eval = preprocess.neuralProcess(X_eval)
    predictions = np.squeeze(model.predict_on_batch(x=processed_eval))
    return predictions

# Logger object redirects standard output to a file.
class Logger(object):
    def __init__(self,filename):
        self.terminal = sys.stdout
        self.log = open("{}.log".format(filename), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass

if __name__ == "__main__":
    main(sys.argv[1:])
