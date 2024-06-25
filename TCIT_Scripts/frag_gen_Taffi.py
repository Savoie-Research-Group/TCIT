#!/bin/env python
# In this version, don't consider ring structure and other FF parameters
import sys,argparse,os,subprocess
import numpy as np
from math import sqrt,sin,cos,tan,factorial,acos
from fnmatch import fnmatch
from copy import deepcopy
from itertools import permutations
from numpy import cross
from builtins import any  

# Add TAFFY Lib to path
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/Lib')
import adjacency as adj
from id_types import id_types

# Description:   Checks if the supplied geometry corresponds to the minimal structure of the molecule
# 
# Inputs:        atomtype:      The taffi atomtype being checked
#                geo:           Geometry of the molecule
#                elements:      elements, indexed to the geometry 
#                adj_mat:       adj_mat, indexed to the geometry (optional)
#                atomtypes:     atomtypes, indexed to the geometry (optional)
#                gens:          number of generations for determining atomtypes (optional, only used if atomtypes are not supplied)
# 
# Outputs:       Boolean:       True if geo is the minimal structure for the atomtype, False if not.
def minimal_structure(atomtype,geo,elements,q_tot=0,adj_mat=None,atomtypes=None,gens=2):

    # If required find the atomtypes for the geometry
    if atomtypes is None or adj_mat is None:
        if len(elements) != len(geo):
            print("ERROR in minimal_structure: While trying to automatically assign atomtypes, the elements argument must have dimensions equal to geo. Exiting...")
            quit()

        # Generate the adjacency matrix
        # NOTE: the units are converted back angstroms
        adj_mat = adj.Table_generator(elements,geo)

        # Generate the atomtypes
        lone_electrons,bonding_electrons,core_electrons,bond_mat,fc = adj.find_lewis(elements,adj_mat,q_tot=q_tot,return_pref=False,return_FC=True)    
        keep_lone = [ [ count_i for count_i,i in enumerate(lone_electron) if i%2 != 0] for lone_electron in lone_electrons] 
        atom_types = id_types(elements,adj_mat,gens,fc=fc,keep_lone=keep_lone)
        atomtypes=[atom_type.replace('R','') for atom_type in atom_types] 

    # Check if this is a ring type, if not and if there are rings
    # in this geometry then it is not a minimal structure. 
    if "R" not in atomtype:
        if True in [ "R" in i for i in atomtypes ]:
            return False
        
    # Check minimal conditions
    count = 0
    for count_i,i in enumerate(atomtypes):
    
        # If the current atomtype matches the atomtype being searched for then proceed with minimal geo check
        if i == atomtype:
            count += 1
            
            # Initialize lists for holding indices in the structure within "gens" bonds of the seed atom (count_i)
            keep_list = [count_i]
            new_list  = [count_i]
            
            # Carry out a "gens" bond deep search
            for j in range(gens):

                # Id atoms in the next generation
                tmp_new_list = []                
                for k in new_list:
                    tmp_new_list += [ count_m for count_m,m in enumerate(adj_mat[k]) if m == 1 and count_m not in keep_list ]

                # Update lists
                tmp_new_list = list(set(tmp_new_list))
                if len(tmp_new_list) > 0:
                    keep_list += tmp_new_list
                new_list = tmp_new_list
            
            # Check for the minimal condition
            keep_list = set(keep_list)
            if False in [ elements[j] == "H" for j in range(len(elements)) if j not in keep_list ]:
                minimal_flag = False
            else:
                minimal_flag = True
    try:    
        return minimal_flag

    except:
        print("check on {}".format(atomtype))
        return False
