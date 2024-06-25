
import sys,os,argparse,subprocess,shutil,time,matplotlib,glob

# For plotting (Agg called needed for cluster image generation)
matplotlib.use('Agg') 
from pylab import *
from scipy.stats import linregress
from scipy.optimize import curve_fit,minimize,lsq_linear
from copy import deepcopy

def parse_configuration(args): #parse config.txt
    

    # Convert inputs to the proper data type
    if os.path.isfile(args) is False:
        print("ERROR in python_driver: the configuration file {} does not exist.".format(args))
        quit()

    # Process configuration file for keywords
    keywords = [ "TAFFI_PATH", "LAMMPS_EXE", "ORCA_EXE", "FF", "MODULE_STRING", "CHARGE", "GENS", "BASIS", "FUNCTIONAL",\
                 "PARAM_GEOOPT_PROCS", "PARAM_GEOOPT_WT", "PARAM_GEOOPT_Q", "PARAM_GEOOPT_SCHED", "PARAM_GEOOPT_PPN", "PARAM_GEOOPT_SIZE",\
                 "PARAM_BA_PROCS", "PARAM_BA_WT", "PARAM_BA_Q", "PARAM_BA_SCHED", "PARAM_BA_PPN", "PARAM_BA_SIZE",\
                 "PARAM_D_PROCS", "PARAM_D_WT", "PARAM_D_Q", "PARAM_D_SCHED", "PARAM_D_PPN", "PARAM_D_SIZE",\
                 "PARAM_FIT_WT", "PARAM_FIT_Q","PARAM_FIT_PPN",\
                 "CHARGES_MD_PROCS", "CHARGES_MD_WT", "CHARGES_MD_Q", "CHARGES_MD_NPP", "CHARGES_MD_SCHED", "CHARGES_MD_PPN", "CHARGES_MD_SIZE",\
                 "CHARGES_QC_PROCS", "CHARGES_QC_WT", "CHARGES_QC_Q", "CHARGES_QC_SCHED", "CHARGES_QC_PPN", "CHARGES_QC_SIZE",\
                 "VDW_MD_PROCS", "VDW_MD_WT", "VDW_MD_Q", "VDW_MD_NPP", "VDW_MD_SCHED", "VDW_MD_PPN", "VDW_MD_SIZE",\
                 "VDW_QC_PROCS", "VDW_QC_WT", "VDW_QC_Q", "VDW_QC_SCHED", "VDW_QC_PPN", "VDW_QC_SIZE","ACCOUNT"]
    keywords = [ _.lower() for _ in keywords ]

    list_delimiters = [ "," ]  # values containing any delimiters in this list will be split into lists based on the delimiter
    space_delimiters = [ "&" ] # values containing any delimiters in this list will be turned into strings with spaces replacing delimiters
    configs = { i:None for i in keywords }    
    with open(args,'r') as f:
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
                        print( "ERROR in python_driver: enountered a keyword ({}) without an argument.".format(i))
                        quit()

    # Find module string (special delimit)
    with open(args,'r') as f:
        for lines in f:
            fields = lines.split('*')
            if "MODULE_STRING" in fields:
               l_fields = [_.lower() for _ in fields]
               for i in keywords:
                   if i in l_fields:
                     ind = l_fields.index(i) + 1
                     if len(fields) >= ind + 1:
                        configs[i] = fields[ind]
               break
            
    # Set defaults if None
    if configs["taffi_path"] is None:
        configs["taffi_path"] = '/'.join(os.path.abspath(__file__).split('/')[:-2])

    if configs["functional"] is None:
        configs["functional"] = 'B3LYP'

    if configs["basis"] is None:
        configs["basis"] = 'def2-TZVP'    

    if configs["module_string"] is None:
        configs["module_string"] = ''

    # Handle special treatment of wB97X-D3 in orca
    if configs["functional"] == 'wB97X-D3':
        configs["nod3"] = '--no_D3'
    else:
        configs["nod3"] = ''

    # Collect the xyz files
    configs["xyz"] = [ f for f in os.listdir('.') if os.path.isfile(f) and ".xyz" in f ]

    # Give absolute path to all ff files
    configs["ff"] = os.path.abspath(configs["ff"])

    return configs

def list2String(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1  

if __name__ == "__main__":
    args = list2String(sys.argv[1:])
    parse_configuration(args)
