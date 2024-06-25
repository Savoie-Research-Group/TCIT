#!/bin/env python
# Author: Brett Savoie (brettsavoie@gmail.com)

import numpy as np 
import fnmatch
import os


# Description: Parses taffi.db files and returns a dictionary with the parameters and modes
def parse_FF_params(FF_files,FF_dict={"masses":{},"charges":{},"bonds":{},"angles":{},"dihedrals":{},"dihedrals_harmonic":{},"vdw":{}}):
                   
    modes_from_FF = []
    for i in FF_files:
        with open(i,'r') as f:
            for lines in f:
                fields = lines.split()
                if len(fields) == 0: continue
                if fields[0].lower() == "atom":   FF_dict["masses"][fields[1]] = float(fields[3])
                if fields[0].lower() == "charge": FF_dict["charges"][fields[1]] = float(fields[2])
                if fields[0].lower() == "bond":   
                    modes_from_FF += [(fields[1],fields[2])]
                    modes_from_FF += [(fields[2],fields[1])]
                    FF_dict["bonds"][(fields[1],fields[2])] = [fields[3],float(fields[4]),float(fields[5])]
                if fields[0].lower() == "angle":
                    modes_from_FF += [(fields[1],fields[2],fields[3])]
                    modes_from_FF += [(fields[3],fields[2],fields[1])]
                    FF_dict["angles"][(fields[1],fields[2],fields[3])] = [fields[4],float(fields[5]),float(fields[6])]
                if fields[0].lower() in ["dihedral","torsion"]: 
                    modes_from_FF += [(fields[1],fields[2],fields[3],fields[4])]
                    modes_from_FF += [(fields[4],fields[3],fields[2],fields[1])]
                    if fields[5] == "opls":       
                        FF_dict["dihedrals"][(fields[1],fields[2],fields[3],fields[4])] = [fields[5]] + [ float(i) for i in fields[6:10] ]
                    elif fields[5] == "harmonic":
                        FF_dict["dihedrals_harmonic"][(fields[1],fields[2],fields[3],fields[4])] = [fields[5]] + [ float(fields[6]),int(float(fields[7])),int(float(fields[8])) ] 
                    elif fields[5] == "quadratic":
                        FF_dict["dihedrals_harmonic"][(fields[1],fields[2],fields[3],fields[4])] = [fields[5]] + [ float(fields[6]),float(fields[7]) ]
                if fields[0].lower() == "vdw":    
                    FF_dict["vdw"][(fields[1],fields[2])] = [fields[3],float(fields[4]),float(fields[5])]
                    FF_dict["vdw"][(fields[2],fields[1])] = [fields[3],float(fields[4]),float(fields[5])]

    return FF_dict,modes_from_FF


# Description: wrapper for grabbing file locations, with wildcard support.
def find_files(files,recursive=False):
    wc_files  = [ i for i in files if "*" in i ]
    files = [ i for i in files if "*" not in i ]
    if recursive:
        files += [ dp+"/"+f for i in wc_files for dp,dn,fn in os.walk('.') for f in fn if fnmatch.fnmatch(f,i) ] 
    else:
        for i in wc_files:
            path = '/'.join(i.split('/')[:-1])
            if len(path) == 0:
                path = "."
            files += [ path+"/"+f for f in os.listdir(path) if fnmatch.fnmatch(f,i) ] 

    # Handle "./" condition
    for count_i,i in enumerate(files):
        if i[:2] == "./": 
            files[count_i] = files[count_i][2:]    

    return list(set(files))

# Description: Simple wrapper function for grabbing the coordinates and
#              elements from an xyz file
#
# Inputs      input: string holding the filename of the xyz
# Returns     Elements: list of element types (list of strings)
#             Geometry: Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#
# To Do: should return a dictionary rather than a variable number of objects
def xyz_parse(input,read_types=False,q_opt=False):

    # Commands for reading only the coordinates and the elements
    if read_types is False:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print( "ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        count = 0
                
                # Get charge
                if lc == 1 and q_opt:
                    if "q" in fields:
                        try:
                            q = int(fields[fields.index("q")+1])
                        except:
                            print("Charge specification misformatted in {}. Defaulting to q=0.".format(input))
                            q = 0
                    else:
                        q = 0
                            
                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print( "ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print( "ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        if q_opt:
            return Elements,Geometry,q
        else:
            return Elements,Geometry

    # Commands for reading the atomtypes from the fourth column
    if read_types is True:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print( "ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        Atom_types = [None]*N_atoms
                        count = 0

                # Get charge
                if lc == 1 and q_opt:
                    if "q" in fields:
                        try:
                            q = int(fields[fields.index("q")+1])
                        except:
                            print("Charge specification misformatted in {}. Defaulting to q=0.".format(input))
                            q = 0
                    else:
                        q = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print( "ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            if len(fields) > 4:
                                Atom_types[count] = fields[4]
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print( "ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))
        
        if q_opt:
            return Elements,Geometry,Atom_types,q
        else:
            return Elements,Geometry,Atom_types

# Description: Parses the molecular charge from the comment line of the xyz file if present
#
# Inputs       input: string holding the filename of the xyz file. 
# Returns      q:     int or None
def parse_q(xyz):

    with open(xyz,'r') as f:
        for lc,lines in enumerate(f):
            if lc == 1:
                fields = lines.split()
                if "q" in fields:
                    q = int(float(fields[fields.index("q")+1]))
                else:
                    q = 0
                break
    return q

# Description: Parses keywords and geometry block from an orca input file
#
# Inputs        input: string holding the filename of the orca input file
# Returns       orca_dict: dictionary holding the run information for each job in the input file
#                          the first key in the dictionary corresponds to the job number (i.e.,
#                          orca_dict["0"] references the information in the first job. The job info
#                          can be accessed with content specific keys ("header_commands", "geo", 
#                          "elements", "constraints", "N_proc", "job_name", "charge", "multiplicity",
#                          "content", "geom_block" )
def orca_in_parse(input):

    # Iterate over the contents and return a dictionary of input components indexed to each job in the input file
    job_num = 0
    orca_dict = {str(job_num):{"header_commands": "","content": "","elements": None, "geo": None, "constraints": None, "geo_opts_block": None, "job_name": None}}
    geo_opts_flag = 0
    geo_block_flag = 0
    con_flag  = 0
    
    # Open the file and begin the parse
    with open(input,'r') as f:
        for lc,lines in enumerate(f):

            # Grab fields 
            fields = lines.split()            
            
            # Update the "content" block, which contains everything
            orca_dict[str(job_num)]["content"] += lines

            # If a new job is encountered reset all flags and update the job_num counter            
            if len(fields) > 0 and fields[0] == "$new_job":
                job_num += 1
                con_flag = 0
                geo_opts_flag = 0
                geo_block_flag = 0
                orca_dict[str(job_num)] = {"header_commands": "","content": "","elements": None, "geo": None, "constraints": None, "geo_opts_block": None, "N_proc": orca_dict[str(job_num-1)]["N_proc"]}

            # Component based parse commands
            if len(fields) > 0 and fields[0] == "!":
                orca_dict[str(job_num)]["header_commands"] += " ".join(fields[1:]) + " "
                if "PAL" in lines:
                    orca_dict[str(job_num)]["N_proc"] = int([ i.split("PAL")[1] for i in fields if "PAL" in i ][0])
                elif job_num != 0:
                    orca_dict[str(job_num)]["N_proc"] = orca_dict[str(job_num-1)]["N_proc"]
                else:
                    orca_dict[str(job_num)]["N_proc"] = 1                    
            if len(fields) > 0 and fields[0] == "%base":
                orca_dict[str(job_num)]["job_name"] = fields[1]
                
            # Check for turning on flags
            if len(fields) > 0 and fields[0] == "%geom":
                geo_opts_flag = 1
                orca_dict[str(job_num)]["geo_opts_block"] = ""                
                continue
            if len(fields) > 0 and fields[0] == "Constraints":
                if geo_opts_flag == 1:
                    orca_dict[str(job_num)]["geo_opts_block"] += lines                
                con_flag = 1
                orca_dict[str(job_num)]["constraints"] = ""
                continue
            if len(fields) >= 2 and fields[0] == "*" and fields[1] == "xyz":
                geo_block_flag = 1
                orca_dict[str(job_num)]["charge"] = float(fields[2])
                orca_dict[str(job_num)]["multiplicity"] = int(fields[3])
                orca_dict[str(job_num)]["geo"] = []
                orca_dict[str(job_num)]["elements"] = []
                continue
            if len(fields) >= 2 and fields[0] == "*" and fields[1] == "xyzfile":
                orca_dict[str(job_num)]["charge"] = float(fields[2])
                orca_dict[str(job_num)]["multiplicity"] = int(fields[3])                
                orca_dict[str(job_num)]["geo"] = None
                orca_dict[str(job_num)]["elements"] = None
                continue

            # Checks for turning off flags
            if con_flag == 1 and len(fields) > 0 and fields[0] == "end":
                con_flag = 0
                continue
            if geo_opts_flag == 1 and len(fields) > 0 and fields[0] == "end":
                geo_opts_flag = 0            
                continue
            if geo_block_flag == 1 and len(fields) > 0 and fields[0] == "*":
                geo_block_flag = 0            
                continue
            
            # Flag based parse commands
            if geo_opts_flag == 1:
                orca_dict[str(job_num)]["geo_opts_block"] += lines
            if con_flag == 1:
                orca_dict[str(job_num)]["constraints"] += lines
            if geo_block_flag == 1:
                orca_dict[str(job_num)]["geo"] += [ [ float(i) for i in fields[1:] ] ]
                orca_dict[str(job_num)]["elements"] += [ str(fields[0]) ]
            
    return orca_dict

