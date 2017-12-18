#!/usr/bin/env python3

"""
author:  Huan Wang
email:   huan.wang@mail.huji.ac.il (or wanghuan@iccas.ac.cn)
version: v 1.5
"""

import numpy as np
import pandas as pd
import os, re, sys, time
from glob import glob
from functools import reduce

path_str1 = "\nPlease Enter the Directory Contains the PDBSlicer Result Files:\n"
path_str2 = "\nPlease Enter the Directory Contains the dssp files:\n"
path_str3 = "\nPlease Enter the Output Directory:\n"
methodstr = "\nPlease Enter the Cutting Methods: (e.g. Ramachandran, backbone, RamaFullRes, RamaSideChain)\n"
codes_str = "\nPlease Enter The Residue Name (3-letter code, e.g. GLN) or prePRO_ResName (e.g. prePRO_ARG):\n"
angle_str = "\nPlease Enter The Angle Name (e.g. chi1, NCCO(backbone), etc.):\n"

path_csm_results = input(path_str1)
path_dssp = input(path_str2)
method = input(methodstr)
residue = input(codes_str).upper()
parameters = list(filter(None, re.split(r'\s+|,|;', input(angle_str))))
path_output = input(path_str3)
structure_file = "initial_normalized_coordinates.pdb"
CSMoutput_file = "output.txt"


length_dict = {
"ALA_l": r"CA|CB",
"SER_l": r"CA|OG",
"CYS_l": r"CA|SG",
"PRO_l": r"CA|CG",  # ????
"THR_l": r"CA|OG1", # "CG2"},
"VAL_l": r"CA|CG1", # "CG2"},
"MET_l": r"CA|CE",
"ILE_l": r"CA|CD1",
"LEU_l": r"CA|CD1", # "CD2"},
"ASN_l": r"CA|ND2",
"ASP_l": r"CA|OD2",
"LYS_l": r"CA|NZ",
"GLN_l": r"CA|NE2",
"GLU_l": r"CA|OE2",
"HIS_l": r"CA|NE2", # ????
"ARG_l": r"CA|NH1", # "NH2"},
"PHE_l": r"CA|CZ",
"TYR_l": r"CA|OH",
"TRP_l": r"CA|CH2" # ??
}


angle_dict = {"tau": r"N$|CA|C$"}


diAng_dict = {
"chi1": r"N$|CA|CB|CG$|CG1$|OG$|OG1$|SG",
"backbone": r"N$|CA|C$|O$",#["C", "N", "CA", "O"],
"Rama": r"C$|N$|CA",#["C", "N", "CA", "C", "N"],
#'omega': #['CA(i-1)', 'C(i-1)', 'N(i)', 'CA(i)']
# the following keys and values are the dihedral angle of \nu
"SER_nu": r"N$|CA|CB|OG",
"CYS_nu": r"N$|CA|CB|SG",
"PRO_nu": r"N$|CA|CB|CG",  # ????
"THR_nu": r"N$|CA|CB|OG1", # "CG2"},
"VAL_nu": r"N$|CA|CB|CG1", # "CG2"},
"MET_nu": r"N$|CA|CB|CE",
"ILE_nu": r"N$|CA|CB|CD1",
"LEU_nu": r"N$|CA|CB|CD1", # "CD2"},
"ASN_nu": r"N$|CA|CB|ND2",
"ASP_nu": r"N$|CA|CB|OD2",
"LYS_nu": r"N$|CA|CB|NZ",
"GLN_nu": r"N$|CA|CB|NE2",
"GLU_nu": r"N$|CA|CB|OE2",
"HIS_nu": r"N$|CA|CB|NE2", # ????
"ARG_nu": r"N$|CA|CB|NH1", # "NH2"},
"PHE_nu": r"N$|CA|CB|CZ",
"TYR_nu": r"N$|CA|CB|OH",
"TRP_nu": r"N$|CA|CB|CH2", # ??
# the following keys and values are the dihedral angle of \eta
"ALA_eta": r"N$|C$|CA|CB",
"SER_eta": r"N$|C$|CA|OG",
"CYS_eta": r"N$|C$|CA|SG",
"PRO_eta": r"N$|C$|CA|CG",  # ????
"THR_eta": r"N$|C$|CA|OG1", # "CG2"},
"VAL_eta": r"N$|C$|CA|CG1", # "CG2"},
"MET_eta": r"N$|C$|CA|CE",
"ILE_eta": r"N$|C$|CA|CD1",
"LEU_eta": r"N$|C$|CA|CD1", # "CD2"},
"ASN_eta": r"N$|C$|CA|ND2",
"ASP_eta": r"N$|C$|CA|OD2",
"LYS_eta": r"N$|C$|CA|NZ",
"GLN_eta": r"N$|C$|CA|NE2",
"GLU_eta": r"N$|C$|CA|OE2",
"HIS_eta": r"N$|C$|CA|NE2", # ????
"ARG_eta": r"N$|C$|CA|NH1", # "NH2"},
"PHE_eta": r"N$|C$|CA|CZ",
"TYR_eta": r"N$|C$|CA|OH",
"TRP_eta": r"N$|C$|CA|CH2" # ??
}


res_dict = {
"GLY": {"N", "CA", "C", "O"},
"ALA": {"N", "CA", "C", "O", "CB"},
"SER": {"N", "CA", "C", "O", "CB", "OG"},
"CYS": {"N", "CA", "C", "O", "CB", "SG"},
"PRO": {"N", "CA", "C", "O", "CB", "CG",  "CD"},
"THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
"VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
"MET": {"N", "CA", "C", "O", "CB", "CG",  "SD",  "CE"},
"ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
"LEU": {"N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2"},
"ASN": {"N", "CA", "C", "O", "CB", "CG",  "OD1", "ND2"},
"ASP": {"N", "CA", "C", "O", "CB", "CG",  "OD1", "OD2"},
"LYS": {"N", "CA", "C", "O", "CB", "CG",  "CD",  "CE",  "NZ"},
"GLN": {"N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "NE2"},
"GLU": {"N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "OE2"},
"HIS": {"N", "CA", "C", "O", "CB", "CG",  "ND1", "CD2", "CE1", "NE2"},
"ARG": {"N", "CA", "C", "O", "CB", "CG",  "CD",  "NE",  "CZ",  "NH1", "NH2"},
"PHE": {"N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ"},
"TYR": {"N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ",  "OH"},
"TRP": {"N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}
}


def find_coordinates_dir(path):
    return glob(os.path.join(path, "*/"))


def pdb_structure(string):
    pdb = [string[0:6].strip(),      # 0.  record name
           string[6:12].strip(),     # 1.  atom serial number
           string[12:16].strip(),    # 2.  atom name with type
           string[16],               # 3.  alternate locatin indicator
           string[17:20].strip(),    # 4.  residue name
           string[21],               # 5.  chain identifier
           string[22:26].strip(),    # 6.  residue sequence number
           string[26],               # 7.  insertion code
           string[30:38].strip(),    # 8.  coordinates X
           string[38:46].strip(),    # 9.  coordinates Y
           string[46:54].strip(),    # 10. coordinates Z
           string[54:60].strip(),    # 11. standard deviation of occupancy
           string[60:66].strip(),    # 12. standard deviation of temperature
           string[76:78].strip(),    # 13. element symbol
           string[98:80].strip()]    # 14. charge on the atom
    return pdb


def read_initial_geometries(filename, method, res):
    ''' This function reads the "initial_normalized_coordinates.pdb" file and
    checks whether the number of coordinates lines match the number of atoms
    of the subunit. If match, collect the coordinates; if not, omit them.

        Additionaly, given the Ramachandran subunit consitituted of 3-adjacent
    residues, some of the adjacent residues could be the same, which should be
    careful.
    For instance in 1BRT.pdb file,

    ATOM      1  C   GLN A  63      -0.032   0.591   2.356  1.00  0.00     C
    ATOM      2  N   SER A  64       0.588   0.069   1.268  1.00  0.00     N
    ATOM      3  CA  SER A  64      -0.094   0.114  -0.022  1.00  0.00     C
    ATOM      4  C   SER A  64      -1.119  -1.025  -0.126  1.00  0.00     C ++
    ATOM      5  CB  SER A  64       0.950  -0.062  -1.149  1.00  0.00     C
    ATOM      6  OG  SER A  64       1.689   1.170  -1.186  1.00  0.00     O
    ATOM      7  N   SER A  65      -1.984  -0.857  -1.141  1.00  0.00     N **

    ATOM      1  C   SER A  64       1.667   0.872   1.466  1.00  0.00     C ++
    ATOM      2  N   SER A  65       0.802   1.040   0.451  1.00  0.00     N **
    ATOM      3  CA  SER A  65      -0.156  -0.028   0.171  1.00  0.00     C
    ATOM      4  C   SER A  65       0.558  -1.322  -0.119  1.00  0.00     C
    ATOM      5  CB  SER A  65      -0.959   0.348  -1.098  1.00  0.00     C
    ATOM      6  OG  SER A  65      -1.723   1.514  -0.879  1.00  0.00     O
    ATOM      7  N   GLN A  66      -0.187  -2.425   0.010  1.00  0.00     N

        In the above case, the two N atoms labeled by '**' and the two C atoms
    labeled'++' are duplicated. Only the central residues of each block should
    be kept.

    The following script implements two round filter processing, enabling to
    obtain correct data.

    Arguments
    ------------------------------------------
    df_geom: the DataFrame contains the atom symbols and the corresponding
             coordinates obtained from the initial_normalized_coordinates.pdb
    res: the name of residue (e.g. ALA or SER, etc.)
    atoms:   the list of the atoms in the dihedral angle to be calculated
    '''
    if method == "Rama":
        number = 5
    elif method == "backbone":
        number = 4
    elif method == "RamaFullRes" and res in res_dict.keys():
        number = len(res_dict[res]) + 2
    elif method == 'RamaSideChain' and res in res_dict.keys():
        number = len(res_dict[res]) + 1

    geom = []

    items = ["Records", # 0
             "AtomSeq", # 1
             "AtomTyp", # 2
             "Alt_Loc", # 3
             "ResName", # 4
             "ChainID", # 5
             "Seq_Num", # 6
             "InsCode", # 7
             "Coord_X", # 8
             "Coord_Y", # 9
             "Coord_Z", # 10
             "SD_Occp", # 11
             "SD_Temp", # 12
             "Element", # 13
             "Charges"] # 14

    col1 = ["AtomTyp", "ResName", "ChainID", "Seq_Num",
            "Coord_X", "Coord_Y", "Coord_Z"]

    col2 = ["ResName", "Seq_Num", "ChainID"]

    with open(filename, "r") as fo:
        temp = []
        for line in fo:
            if line.startswith("ENDMDL"):
                temp = []

            elif line.startswith("ATOM"):
                info = pdb_structure(line)
                temp.append(info)
                #### collecting the residues without "missing atoms"
                if len(temp) == number:
                    #### keeping the atoms in the centeral residues
                    geom.append(temp[1:-1])
    if geom:
        geom = pd.DataFrame(np.array(geom).reshape(-1, len(items)),
                            columns=items)[col1]

        df = geom[col2].drop_duplicates(subset=["Seq_Num"],
                                        keep="first").reset_index(drop=True)
        return df, geom
    else:
        return pd.DataFrame(), pd.DataFrame()


def pick_coordinates(df_geom, res, atoms):
    ''' This function collects the coordinates of atoms in the dihedral angle
    in the subunit from the "initial_normalized_coordinates.pdb" files, which
    obtained from the PDBSlicer ouptut folders.

    It returns a 3-D data array of atom symbols and their coordinates (str type)

    Arguments
    ------------------------------------------
    df_geom: the DataFrame contains the atom symbols and the corresponding
             coordinates obtained from the initial_normalized_coordinates.pdb
    res: the name of residue (e.g. ALA or SER, etc.)
    atoms:   the list of the atoms in the dihedral angle to be calculated
    '''
    cols = ["AtomTyp", "Coord_X", "Coord_Y", "Coord_Z"]
    atom_list = list(filter(None, re.split(r"\$|\|", atoms)))

    if not df_geom.empty:
        #### filtering the atoms within the dihedral angle ####
        mask = (df_geom.AtomTyp.str.match(atoms)) & (df_geom.ResName == res)
        df_geom = df_geom[mask]
        #print("df_geom is:\n{:}\n".format(df_geom))
        try:
            xyz = df_geom[cols].values.reshape(-1,len(atom_list),4)

        except ValueError:
            sys.exit("Parse Data Error! Might have special case.")
            xyz = np.array([])
        return xyz, atom_list

    else:
        return np.array([]), []


def calc_distance(coordinates, atom_pair):
    ''' Calculate the distance between two potins (e.g. p1 and p2).
    This function returns an 1-D array of distance values of the given
    atom pair.

    Arguments
    ------------------------------------------
    coordinates: a 3-D array contains the atom symbols and their coordinates
    atoms_list:  an ordered list of atom symbols

    Then, the coordinates of p1 and p2 are 2-D arrays.
    '''
    atom1 = atom_pair[0]
    atom2 = atom_pair[1]
    p1 = coordinates[coordinates[:,:,0] == atom1][:,1:].astype(np.float64)
    p2 = coordinates[coordinates[:,:,0] == atom2][:,1:].astype(np.float64)
    return np.linalg.norm(p1 - p2, axis=-1)


def calc_angles(coordinates, atoms_list):
    ''' Calculate the angle of atom1-atom2-atom3 (e.g. N--C_alpha--C) in each
    residue of the protine (PDB file).

    This function returns a 1D array of the angles (in the unit of degree) of
    the given ordered atom list.

    Arguments
    ------------------------------------------
    coordinates: a 3-D array contains the atom symbols and their coordinates
    atoms_list:  an ordered list of atom symbols

    Then, the coordinates of p1, p2 and p3 are 2-D arrays.
    '''
    atom1 = atoms_list[0]
    atom2 = atoms_list[1]
    atom3 = atoms_list[2]
    p1 = coordinates[coordinates[:,:,0] == atom1][:,1:].astype(np.float64)
    p2 = coordinates[coordinates[:,:,0] == atom2][:,1:].astype(np.float64)
    p3 = coordinates[coordinates[:,:,0] == atom3][:,1:].astype(np.float64)

    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = (v1 * v2).sum(axis=-1)
    sin_angle = np.linalg.norm(np.cross(v1, v2), axis=-1)
    return np.rad2deg(np.arctan2(sin_angle, cos_angle))


def calc_dihedral_angles(coordinates, atoms_list):
    ''' Using the Gram–Schmidt process to calculate the dihedral_angle
    of atom1-atom2-atom3-atom4 (e.g. N--C_alpha--C--O) in each residue
    of the protine (PDB file).
        This function returns a 1D data array of the dihedral angles (in
    the unit of degree) of the given ordered atom list.

    Arguments
    ------------------------------------------
    coordinates: a 3-D array contains the atom symbols and their coordinates
    atoms_list:  an ordered list of atom symbols

    Then, the coordinates of p1, p2, p3, and p4 are 2-D arrays.
    '''
    atom1 = atoms_list[0]
    atom2 = atoms_list[1]
    atom3 = atoms_list[2]
    atom4 = atoms_list[3]
    p1 = coordinates[coordinates[:,:,0] == atom1][:,1:].astype(np.float64)
    p2 = coordinates[coordinates[:,:,0] == atom2][:,1:].astype(np.float64)
    p3 = coordinates[coordinates[:,:,0] == atom3][:,1:].astype(np.float64)
    p4 = coordinates[coordinates[:,:,0] == atom4][:,1:].astype(np.float64)

    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3

    # normalize b1 so that it does not influence magnitude of vector
    # projections that come next
    b1 /= np.linalg.norm(b1, axis=-1).reshape(-1,1)

    # vector projection using Gram–Schmidt process
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - (b0 * b1).sum(axis=-1).reshape(-1,1) * b1
    w = b2 - (b2 * b1).sum(axis=-1).reshape(-1,1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = (v * w).sum(axis=-1)
    y = (np.cross(b1, v) * w).sum(axis=-1)
    return np.rad2deg(np.arctan2(y, x))


def collect_CsmCcm(path, CSMoutput_file, PDB_ID):
    ''' This function returns a 2-D array contained the amino_acid_names,
    sequence_numbers and CSM/CCM_values.
    '''
    data = []
    df_CsmCcm = pd.DataFrame()
    cols = ["ResName", "Seq_Num", "csm_value", "ChainID"]

    chirality_file = os.path.join(path, CSMoutput_file)

    with open(chirality_file, "r") as fo2:
        #### skip the headlines
        for line in fo2:
            if line.strip().startswith("mdl_indx"):
                break
        #### read the data we want
        for line in fo2:
            info = line.split()
            resname, seq_num, csm_value, chain = info[3:]
            data.append([resname, seq_num, csm_value, chain])
        df_CsmCcm = pd.DataFrame(data, columns=cols)
        df_CsmCcm["PDB_ID"] = PDB_ID.upper()
    return df_CsmCcm


def dssp_format(string):
    ''' The data format of the DSSP file.
    '''
    data = [string[0:5].strip(),        # 0. id
            string[5:10].strip(),       # 1. residue number
            string[10],                 # 2. insertion code
            string[11],                 # 3. chainID
            string[13],                 # 4. resname 1-letter code
            string[14],                 # 5. chain breaker noation, *
            string[16],                 # 6. secStru
            string[18],                 # 7. 3-turns/helix
            string[19],                 # 8. 4-turns/helix
            string[20],                 # 9. 5-turns/helix
            string[21],                 # 10.geometrical bend
            string[22],                 # 11.chirality
            string[23],                 # 12.beta bridge label
            string[24],                 # 13.beta bridge label
            string[25:29].strip(),      # 14.beta bridge partner resnum, bp1
            string[29:33].strip(),      # 15.beta bridge partner resnum, bp2
            string[33],                 # 16.beta sheet label
            string[35:38].strip(),      # 17.solvent accessibility, ACC
            string[41:50].strip(),      # 18.N-H --> O
            string[52:61].strip(),      # 19.O --> H-N
            string[63:72].strip(),      # 20.N-H --> O
            string[74:83].strip(),      # 21.O --> H-N
            string[85:91].strip(),      # 22.cosine of angle between C=O or residue i and C=O of residue i-1, TCO
            string[91:97].strip(),      # 23.virtual bond angle (bend) defined by the 3-CA atoms of residues i-2, i, i+2, kappa
            string[97:103].strip(),     # 24.virtual bond angle (dihedral angle) defined by the 4-CA atoms of residues i-1, i, i+1, i+2, alpha
            string[103:109].strip(),    # 25.dihedral angle, Phi
            string[109:115].strip(),    # 26.dihedral angle, Psi
            string[115:122].strip(),    # 27.coordinate X of CA
            string[122:129].strip(),    # 28.coordinate Y of CA
            string[129:136].strip()]    # 29.coordinate Z of CA
    return data


def dssp_reader(path_dssp, dssp_file, PDB_ID):
    ''' Read the DSSP data into pandas DataFrame.
    '''
    dssp = []
    items = ["line_ID", "Seq_Num", "InsCode", "ChainID", "ResName", "Breaker",
             "secStru", "helix3t", "helix4t", "helix5t", "GeoBend", "chiral",
             "betaBG1", "betaBG2", "bp1", "bp2", "betaL", "ACC",
             "NH_O1", "O_HN1", "NH_O2", "O_HN2", "TCO", "Kappa", "Alpha",
             "Phi", "Psi", "XCA", "YCA", "ZCA"]
    cols = ["Phi", "Psi", "secStru", "ResName", "Seq_Num", "ChainID", "PDB_ID"]

    one_to_3code = {"A": "ALA",
                    "R": "ARG",
                    "N": "ASN",
                    "D": "ASP",
                    "C": "CYS",
                    "Q": "GLN",
                    "E": "GLU",
                    "G": "GLY",
                    "H": "HIS",
                    "I": "ILE",
                    "L": "LEU",
                    "K": "LYS",
                    "M": "MET",
                    "F": "PHE",
                    "P": "PRO",
                    "S": "SER",
                    "T": "THR",
                    "W": "TRP",
                    "Y": "TYR",
                    "V": "VAL"}

    with open(os.path.join(path_dssp, dssp_file), "r") as fo:
        ########################################
        #### skip the head lines
        for line in fo:
            if line.strip().startswith("#"):
                break
        ########################################
        #### read the data we want
        for line in fo:
            info = dssp_format(line)
            dssp.append(info)
        dssp = pd.DataFrame(dssp, columns=items)
        dssp.replace({"ResName": one_to_3code}, inplace=True)
        dssp.secStru.replace(" ", "C", inplace=True)
        dssp["PDB_ID"] = PDB_ID.upper()

        if np.any(dssp.ResName.str.islower()):
            dssp.ResName.replace(r"[a-z]+", "CYS", regex=True, inplace=True)
        return dssp[cols]


def merge_tables(df, df_CsmCcms, df_dssp, title):
    ''' This function returns merged DataFrames when all of them not empty.
    Once any one of the is empty, returns an empty DataFrame.
    '''
    if df.empty or df_CsmCcms.empty or df_dssp.empty:
        return pd.DataFrame()

    else:
        dfs = [df, df_CsmCcms, df_dssp]
        criteria = ["ResName", "Seq_Num", "ChainID", "PDB_ID"]

        merged = reduce(lambda left, right: pd.merge(left, right,
                                                     how="inner",
                                                     on=criteria), dfs)
        return merged[title]


def main(path_csm_results, structure_file, CSMoutput_file,
         path_dssp, method, residue, path_output, parameters):
    ''' Workflow:
    * Find all subdirectories in the given path;
    if "initial_coordinates.pdb" file exist and not empty, do the following
    processings in a loop:

      - read the coordinates of the subunits in "initial_coordinates.pdb" file;

      - calculate the dihedral angles of the subunits (here is chi1);

      - read the CCM/CSM values in the corresponding output file;

      - read the Phi, Psi, and secondary structure codes in the corresponding
       DSSP file;

    * Merge the above information and save as a ".csv" file.
    '''
    if residue[-3:] in res_dict.keys():
        res = residue[-3:]
    else:
        sys.exit("Caution! Wrong Residue Name.")

    initial_time = time.time()
    final_df = pd.DataFrame()
    empty_file = []
    drawline = ''.join(("\n", "-" * 79, "\n"))
    fmt_step = ''.join(("\nData Processing Done.",
                        "\tUsed Time at This Step: {:.3f} seconds"))
    title = ["Phi", "Psi"] + parameters + ["csm_value", "secStru",
             "ResName", "Seq_Num", "ChainID", "PDB_ID"]

    para = ['_'.join((res, x)) for x in parameters if x != "tau"]
    print("dihedral angles are: {}\n".format(para))

    subdirs = find_coordinates_dir(path_csm_results)

    for i, subdir in enumerate(subdirs):
        start_time = time.time()

        PDB_ID = subdir.split(os.sep)[-2][4:8]

        print_fmt = ''.join((drawline, "No. {:}, file {:}"))
        print(print_fmt.format(i + 1, PDB_ID.upper()))

        coords_file = os.path.join(subdir, structure_file)

        #### processing only if file exist and not an empty file
        if os.path.isfile(coords_file) and os.stat(coords_file).st_size > 10:

            #### deal with ccm values (output.txt from PDBSlicer)
            df_CsmCcms = collect_CsmCcm(subdir, CSMoutput_file, PDB_ID)
            #print("df_CsmCcms is:\n{:}\n".format(df_CsmCcms))

            ####
            df, df_geom = read_initial_geometries(coords_file, method, res)

            #### deal with dihedral angles (e.g. N-CA-CB-last-atom-of-sidechain)
            nu_str = diAng_dict[para[0]]
            eta_str = diAng_dict[para[1]]
            l_str = length_dict[para[2]]
            tau_str = angle_dict[parameters[3]]

            nu_array, nu_atoms = pick_coordinates(df_geom, res, nu_str)
            eta_array, eta_atoms = pick_coordinates(df_geom, res, eta_str)
            l_array, l_atoms = pick_coordinates(df_geom, res, l_str)
            tau_array, tau_atoms = pick_coordinates(df_geom, res, tau_str)
            #print("l array is:\n{:}\n".format(l_array))

            if nu_array.size and eta_array.size and l_array.size and tau_array.size:
                nu_dihedral = calc_dihedral_angles(nu_array, nu_atoms)
                df_nu = pd.Series(nu_dihedral)

                eta_dihedral = calc_dihedral_angles(eta_array, eta_atoms)
                df_eta = pd.Series(eta_dihedral)

                l_leng = calc_distance(l_array, l_atoms)
                df_l = pd.Series(l_leng)

                tau_angle = calc_angles(tau_array, tau_atoms)
                df_tau = pd.Series(tau_angle)

            else:
                df_nu = pd.Series()
                df_eta = pd.Series()
                df_l = pd.Series()
                df_tau = pd.Series()

            df[parameters[0]] = df_nu
            df[parameters[1]] = df_eta
            df[parameters[2]] = df_l
            df[parameters[3]] = df_tau
            df["PDB_ID"] = PDB_ID.upper()
            #print("df is:\n{:}\n".format(df))

            #### deal with dssp file (Phi, Psi, secondary structures)
            dssp_f = ''.join((PDB_ID, ".dssp"))
            df_dssp = dssp_reader(path_dssp, dssp_f, PDB_ID)
            #print("df_dssp is:\n{:}\n".format(df_dssp))

            merged = merge_tables(df, df_CsmCcms, df_dssp, title)

            step_time = time.time() - start_time
            print(fmt_step.format(step_time))

            final_df = final_df.append(merged, ignore_index=True)

        else:
            print("The {:} Does NOT Exist!".format(coords_file))
            empty_file.append(PDB_ID.upper())

    print("\nThe Final Table is:\n", final_df)

    outputf = ''.join(("Phi_Psi_CCM_dihedrals_", residue, ".csv"))
    out_path_file = os.path.join(path_output, outputf)
    final_df.to_csv(out_path_file, sep=",", columns=title, index=False)
    fmt_save = ''.join((drawline,
                       "Merged file saved as:\n{:}",
                       drawline))
    print(fmt_save.format(out_path_file))

    if empty_file:
        string = "The non-existent/empty PDBSlicer output.txt file(s):\n"
        fmt_none = ''.join((drawline,
                            string, "{:<6}" * len(empty_file),
                            drawline))
        print(fmt_none.format(*empty_file))
        nonfile = ''.join(("non-existent", "_", residue, ".txt"))
        none_path_file = os.path.join(path_output, nonfile)
        with open(none_path_file, "w") as fw:
            fw.write(fmt_none.format(*empty_file))

    fmt_end = ''.join((drawline,
                       "Work Complete. Used Time: {:.3f} seconds.",
                       drawline))
    used_time = time.time() - initial_time
    print(fmt_end.format(used_time))


if __name__ == "__main__":
    main(path_csm_results, structure_file, CSMoutput_file,
         path_dssp, method, residue, path_output, parameters)
