#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os, sys, time
from glob import glob
from functools import reduce

pathstr1 = "\nPlease Enter the Directory Contains the PDBSlicer Result Files:\n"
pathstr2 = "\nPlease Enter the Directory Contains the dssp files:\n"
pathstr3 = "\nPlease Enter the Output Directory:\n"
type_str = "\nPlease Enter The Residue Name (3-letter code) or subunit:\n"
anglestr = "\nPlease Enter The Angle Name (e.g. chi1, NCCO(backbone), etc.):\n"

path_csm_results = input(pathstr1)
path_dssp = input(pathstr2)
subunits = input(type_str)
angle = input(anglestr)
path_output = input(pathstr3)
structure_file = "initial_normalized_coordinates.pdb"
CSMoutput_file = "output.txt"


subunit_dict = {
"chi1":r'N$|CA|CB|CG$|CG1$|OG$|OG1$|SG',
"backbone":r'N$|CA|C$|O$',#["C", "N", "CA", "O"],
"Rama":r'C$|N$|CA',#["C", "N", "CA", "C", "N"],
'GLY':["N", "CA", "C", "O"],
'ALA':["N", "CA", "C", "O", "CB"],
'SER':["N", "CA", "C", "O", "CB", "OG"],
'CYS':["N", "CA", "C", "O", "CB", "SG"],
'PRO':["N", "CA", "C", "O", "CB", "CG",  "CD"],
'THR':["N", "CA", "C", "O", "CB", "OG1", "CG2"],
'VAL':["N", "CA", "C", "O", "CB", "CG1", "CG2"],
'MET':["N", "CA", "C", "O", "CB", "CG",  "SD",  "CE"],
'ILE':["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
'LEU':["N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2"],
'ASN':["N", "CA", "C", "O", "CB", "CG",  "OD1", "ND2"],
'ASP':["N", "CA", "C", "O", "CB", "CG",  "OD1", "OD2"],
'LYS':["N", "CA", "C", "O", "CB", "CG",  "CD",  "CE",  "NZ"],
'GLN':["N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "NE2"],
'GLU':["N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "OE2"],
'HIS':["N", "CA", "C", "O", "CB", "CG",  "ND1", "CD2", "CE1", "NE2"],
'ARG':["N", "CA", "C", "O", "CB", "CG",  "CD",  "NE",  "CZ",  "NH1", "NH2"],
'PHE':["N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ"],
'TYR':["N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ",  "OH"],
'TRP':["N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
}

items = ['Records',
         'AtomSeq',
         'AtomTyp',
         'Alt_Loc',
         'ResName',
         'ChainID',
         'Seq_Num',
         'InsCode',
         'Coord_X',
         'Coord_Y',
         'Coord_Z',
         'SD_Occp',
         'SD_Temp',
         'Element',
         'Charges']

cols = ['ResName', 'Seq_Num', 'csm_value', 'ChainID']

one_to_3code = {'A': 'ALA',
                'R': 'ARG',
                'N': 'ASN',
                'D': 'ASP',
                'C': 'CYS',
                'Q': 'GLN',
                'E': 'GLU',
                'G': 'GLY',
                'H': 'HIS',
                'I': 'ILE',
                'L': 'LEU',
                'K': 'LYS',
                'M': 'MET',
                'F': 'PHE',
                'P': 'PRO',
                'S': 'SER',
                'T': 'THR',
                'W': 'TRP',
                'Y': 'TYR',
                'V': 'VAL'}


def find_coordinates_dir(path):
    return glob(os.path.join(path, '*/'))


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


def read_PDB(filename, items):
    data = []
    with open(filename, 'r') as fo:
        for line in fo:
            if line.startswith("ATOM"):
                info = pdb_structure(line)
                data.append(info)
    return pd.DataFrame(np.array(data), columns=items)


def calc_dihedral_angles(coordinates):
    ''' Using the Gram–Schmidt process to calculate the dihedral_angle
    of atom1-atom2-atom3-atom4 (e.g. N--C_alpha--C--O) in each residue
    of the protine (PDB file).
        This function returns a 1D data array of the dihedral angles
    in the unit of degree.
    '''
    p1 = np.asfarray(coordinates[:,0,:], dtype=np.float64)
    p2 = np.asfarray(coordinates[:,1,:], dtype=np.float64)
    p3 = np.asfarray(coordinates[:,2,:], dtype=np.float64)
    p4 = np.asfarray(coordinates[:,3,:], dtype=np.float64)
    
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


def collect_dihedral_angles(coords_file, pdb_id, items, subunits, diAng):
    ''' This function collects the coordinates of the subunits in 
    the initial_normalized_coordinates.pdb files from the PDBSlicer 
    output folders.
    
        Since Ramachandran subunits cross 3-adjacent residues, 
    additionally, in the special case, some of the adjacent 
    residues could be the same. For instance in 1BRT.pdb file,
    
    ATOM      1  C   GLN A  63      -0.032   0.591   2.356  1.00  0.00     C  
    ATOM      2  N   SER A  64       0.588   0.069   1.268  1.00  0.00     N  
    ATOM      3  CA  SER A  64      -0.094   0.114  -0.022  1.00  0.00     C  
    ATOM      4  C   SER A  64      -1.119  -1.025  -0.126  1.00  0.00     C  
    ATOM      5  CB  SER A  64       0.950  -0.062  -1.149  1.00  0.00     C  
    ATOM      6  OG  SER A  64       1.689   1.170  -1.186  1.00  0.00     O  
    ATOM      7  N   SER A  65      -1.984  -0.857  -1.141  1.00  0.00     N **
    
    ATOM      1  C   SER A  64       1.667   0.872   1.466  1.00  0.00     C  
    ATOM      2  N   SER A  65       0.802   1.040   0.451  1.00  0.00     N ** 
    ATOM      3  CA  SER A  65      -0.156  -0.028   0.171  1.00  0.00     C  
    ATOM      4  C   SER A  65       0.558  -1.322  -0.119  1.00  0.00     C  
    ATOM      5  CB  SER A  65      -0.959   0.348  -1.098  1.00  0.00     C  
    ATOM      6  OG  SER A  65      -1.723   1.514  -0.879  1.00  0.00     O  
    ATOM      7  N   GLN A  66      -0.187  -2.425   0.010  1.00  0.00     N
    
        In the above case, the two N atoms labeled by '**' are duplicated. 
    Only the 2nd one should be kept. The following script implements two 
    round filter processing, enabling to obtain correct data.
    '''
    title = ['AtomTyp', 'ResName', 'ChainID', 'Seq_Num', 
             'Coord_X', 'Coord_Y', 'Coord_Z']
    atoms = subunit_dict[diAng]
    dihedral = df_dihedral = pd.DataFrame()
    coords_list = ["Coord_X", "Coord_Y", "Coord_Z"]
    cols = ['ResName', 'Seq_Num', 'ChainID']
    
    geom = read_PDB(coords_file, items)[title]
    
    #### filter 1st round ####
    factor = (geom.AtomTyp.str.match(atoms)) & (geom.ResName == subunits)
    geom = geom[factor]
    
    #### filter 2nd round ####
    if diAng == 'chi1':
        geom = geom[geom.AtomTyp.shift(-1) != geom.AtomTyp]
        if geom['Seq_Num'].iloc[-1] != geom['Seq_Num'].iloc[-2]:
            geom = geom.iloc[:-1]
        #print("\n----- 2nd filter ----\n{}\n".format(geom))
    
    try:
        xyz = geom[coords_list].values.reshape(-1,4,3)
    except ValueError:
        fmt = ''.join((''.join(("\n", "#" * 79, "\n")),
                       "Missing atom(s)!\n",
                       "\nPlease check the following file:\n{:}" ,
                       ''.join(("\n", "#" * 79, "\n"))))
        print(fmt.format(coords_file))
        sys.exit(fmt.format(coords_file))
    
    #### chi1 is defined by the dihedral angle N-CA-CB-\wG|\wG1
    dihedral = calc_dihedral_angles(xyz)
    
    dic = {}
    key, value = 'Chi1', dihedral
    dic.setdefault(key, value)
    
    dihedral = pd.Series(dic)
    df_dihedral = geom.drop_duplicates(subset=['Seq_Num'], 
                                       keep='first')[cols].reset_index(drop=True)
    df_dihedral["PDB_ID"] = pdb_id.upper()
    df_dihedral['Chi1'] = dic['Chi1']
    return df_dihedral


def collect_CsmCcm(path, CSMoutput_file, cols, PDB_ID):
    ''' This function returns a 2-D array contained the amino_acid_names,
    sequence_numbers and CSM/CCM_values.
    '''
    data = []
    df_CsmCcm = pd.DataFrame()
    
    chirality_file = os.path.join(path, CSMoutput_file)
    
    with open(chirality_file, 'r') as fo2:
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
    items = ['lineID', 'Seq_Num', 'InsCode', 'ChainID', 'ResName', 'Breaker',
             'secStru', 'helix3t', 'helix4t', 'helix5t','GeoBend','chiral', 
             'betaBG1', 'betaBG2','bp1', 'bp2', 'betaL', 'ACC',
             'NH_O1', 'O_HN1', 'NH_O2', 'O_HN2', 'TCO', 'Kappa', 'Alpha',
             'Phi', 'Psi', 'XCA', 'YCA', 'ZCA']
    cols = ['Phi', 'Psi', 'secStru', 'ResName', 'Seq_Num', 'ChainID', 'PDB_ID']

    with open(os.path.join(path_dssp, dssp_file), 'r') as fo:
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
        dssp.replace({'ResName': one_to_3code}, inplace=True)
        dssp.secStru.replace(' ', 'C', inplace=True)
        dssp["PDB_ID"] = PDB_ID.upper()
        
        if np.any(dssp.ResName.str.islower()):
            dssp.ResName.replace(r'[a-z]+', 'CYS', regex=True, inplace=True)
        return dssp[cols]


def merge_tables(df_dihedral, df_CsmCcms, df_dssp, title):
    ''' This function returns merged DataFrames when all of them not empty.
    Once any one of the is empty, returns an empty DataFrame.
    '''
    if not (df_dihedral.empty and df_CsmCcms.empty and df_dssp.empty):
        dfs = [df_dihedral, df_CsmCcms, df_dssp]
        criteria = ['ResName', 'Seq_Num', 'ChainID', 'PDB_ID']
        
        merged = reduce(lambda left, right: pd.merge(left, right,
                                                     how='inner',
                                                     on=criteria), dfs)
        return merged[title]
    else: # "Empty data! Pass."
        return pd.DataFrame()


def main(path_csm_results, structure_file, CSMoutput_file, 
         path_dssp, subunits, angle, path_output):
    ''' Workflow:
    Find all subdirectories in the given path;
    if "initial_coordinates.pdb" file exist and not empty, do the following
    processings in a loop:
    
      - read the coordinates of the subunits in "initial_coordinates.pdb" file;

      - calculate the dihedral angles of the subunits (here is chi1);

      - read the CCM/CSM values in the corresponding output file;

      - read the Phi, Psi, and secondary structure codes in the corresponding
       DSSP file;

    Merge the above information and save as a ".csv" file.
    '''
    initial_time = time.time()
    final_df = pd.DataFrame()
    empty_file = []
    drawline = ''.join(("\n", "-" * 79, "\n"))
    fmt_step = ''.join(('\nData Processing Done.',
                        '\tUsed Time at This Step: {:.3f} seconds'))
    title = ['Phi', 'Psi', 'Chi1', 'csm_value', 'secStru',
             'ResName', 'Seq_Num', 'ChainID', 'PDB_ID']
    
    subdirs = find_coordinates_dir(path_csm_results)
    
    for i, subdir in enumerate(subdirs):
        start_time = time.time()
        
        PDB_ID = subdir.split(os.sep)[-2][4:8]
        
        print_fmt = ''.join((drawline, "No. {:}, file {:}"))
        print(print_fmt.format(i + 1, PDB_ID.upper()))
        
        coords_file = os.path.join(subdir, structure_file)
        
        #### processing only if file exist and not an empty file
        if os.path.isfile(coords_file) and os.stat(coords_file).st_size > 10:
            
            #### deal with chi1 angles (dihedral angle N-CA-CB-\wG|\wG1)
            df_dihedral = collect_dihedral_angles(coords_file, PDB_ID, items,
                                                  subunits, angle)
            #print("df_dihedral is:\n{:}\n".format(df_dihedral))
        
            #### deal with ccm values (output.txt from PDBSlicer)
            df_CsmCcms = collect_CsmCcm(subdir, CSMoutput_file, cols, PDB_ID)
            #print("df_CsmCcms is:\n{:}\n".format(df_CsmCcms))
        
            #### deal with dssp file (Phi, Psi, secondary structures)
            dssp_f = ''.join((PDB_ID, '.dssp'))
            df_dssp = dssp_reader(path_dssp, dssp_f, PDB_ID)
            #print("df_dssp is:\n{:}\n".format(df_dssp))
        
            merged = merge_tables(df_dihedral, df_CsmCcms, df_dssp, title)
            
            step_time = time.time() - start_time
            print(fmt_step.format(step_time))
            
            final_df = final_df.append(merged, ignore_index=True)
            
        else:
            print("The {:} Does NOT Exist!".format(coords_file))
            empty_file.append(PDB_ID.upper())
        
    print("\nThe Final Table is:\n", final_df)
    
    outputf = ''.join(('Phi_Psi_CCM_', angle, '_', subunits, '.csv'))
    out_path_file = os.path.join(path_output, outputf)
    final_df.to_csv(out_path_file, sep=',', columns=title, index=False)
    fmt_save = ''.join((drawline, 
                       "Merged file saved as:\n{:}",
                       drawline))
    print(fmt_save.format(out_path_file))
    
    fmt_none = ''.join(("\nThe non-existent PDBSlicer output.txt file(s):\n", 
                        "{:<6}" * len(empty_file)))
    print(fmt_none.format(*empty_file))
    
    fmt_end = ''.join((drawline, 
                       "Work Complete. Used Time: {:.3f} seconds.",
                       drawline))
    used_time = time.time() - initial_time
    print(fmt_end.format(used_time))

  
if __name__ == "__main__":
    main(path_csm_results, structure_file, CSMoutput_file, 
         path_dssp, subunits, angle, path_output)
