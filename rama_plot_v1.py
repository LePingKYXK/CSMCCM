# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:04:46 2017

@author: wangh
"""

import numpy as np
import pandas as pd
import os, time
import matplotlib.pyplot as plt
#import seaborn as sns
#from matplotlib.colors import ListedColormap

pstr = "\nPlease enter the directory contains PDB files: \n"
path = "D:\HUJI_data\Continuous_Symmetry\scripts"#input(pstr)
#"D:\HUJI_data\Continuous_Symmetry\my_proteins\cleaned_my_PDB"

#subunit = "THR"

subunits_dict = {
"chi1":r'N$|CA|CB|CG$|CG1$|OG$|OG1$|SG',
"backbone":r'N$|CA|C$|O$',#["N", "CA", "C", "O"],
"Rama":r'C$|N$|CA',#["C", "N", "CA", "C", "N"]}
# The following items are the amino acids and the their atoms.
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


def find_PDB_files(path):
    suffix = ".pdb"
    return (f for f in os.listdir(path) if f.endswith(suffix))


def pdb_reader(filename):
    pdb_info = [] 

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

    with open(filename, 'r') as fo:
        for line in fo:
            if line.startswith("REMARK 465"):
                pass

            elif line.startswith("SEQRES"):
                pass

            elif line.startswith("MODEL"):
                pass
            
            elif line.startswith("ENDMDL"):
                pass
            
            elif line.startswith("ATOM") or \
                 line.startswith("HETATM") or \
                 line.startswith("TER"):

                info = pdb_structure(line)
                pdb_info.append(info)
                
            elif line.startswith("SIGATM"):
                pass
            
            elif line.startswith("ANISOU"):
                pass
            
            elif line.startswith("SIGUIJ"):
                pass
        
    pdb_info = pd.DataFrame(pdb_info, columns=items)
    pdb_info['Seq_Num'] = pdb_info['Seq_Num'].astype(int)

    #####################################################################
    #### locates the last 'TER' in the sequence.
    terminal_id = pdb_info[pdb_info["Records"] == "TER"].index.tolist()

    #####################################################################    
    #### return the pdb information without solvent.
    if pdb_info.shape[0] > terminal_id[-1]:
        return pdb_info.drop(pdb_info.index[terminal_id[-1]+1:])
    
    elif pdb_info.shape[0] == terminal_id[-1]:
        return pdb_info
    
    else: # pdb_info.shape[0] < terminal_id[-1], 
        print("Error!")


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


def build_Ramachandran_array(dataframe):
    ''' 
    '''
    frame = (dataframe.AtomTyp == 'CA').sum() - 2
    Ramachandran = np.empty((frame, 5, 3))
    #print('shape shape shape\n{}\n===='.format(np.shape(Ramachandran)))    
    """
    #### using numpy 
    i = 0
    
    array = dataframe.values
    
    for i in range(frame):
        Ramachandran[i] = array[(2 + i*3):(2 + i*3)+5,-3:]
    """
    #### using pandas shift() 
    df = dataframe.shift(-2)
    i = 0    
    while (not df.iloc[:5].isnull().values.any()) and i < frame - 1:
        data = df.iloc[:5,-3:].values
        Ramachandran[i] = data
        df = df.shift(-3)
        i += 1
    
    return np.array(Ramachandran, dtype=np.float64).reshape(-1,5,3)

    ''' to do: for other subunit cuts, need to specify subunit names
    
    elif subunit == 'backbone':
        subunit_array = np.array(subunit_array).reshape(-1,4,3)
        theta = subunit_array[:,]
        return subunit_array, theta
    
    elif subunit in 'residue':
        size = len(subunits_dict[subunit])
        subunit_array = np.array(subunit_array).reshape(-1,size,3)
    return subunit_array
    '''

def calc_dihedral_angles(coordinates):
    ''' Using the Gram–Schmidt process to calculate the dihedral_angle
    of atom1-atom2-atom3-atom4 (e.g. N--C_alpha--C--O) in each residue
    of the protine (PDB file).
        This function returns a 1D data array of the dihedral angles
    in the unit of degree.
    '''
    p1 = coordinates[:,0,:]
    p2 = coordinates[:,1,:]
    p3 = coordinates[:,2,:]
    p4 = coordinates[:,3,:]
    
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


def plot_rama(phi, psi):
    """Function to plot two one-dimensional arrays"""
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    #fig.subplots_adjust(top=0.9, bottom=0.1 ,left=0.1, right=0.9)
    ax.set_title('The Ramachandran Plot', fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel("$\phi$ $(\degree)$", fontsize=20)    #xlabel
    ax.set_ylabel("$\psi$ $(\degree)$", fontsize=20)    #ylabel
    ax.set_xlim(-180, 180) # Sets axis limits
    ax.set_ylim(-180, 180) # Sets axis limits
    ax.set_xticks(np.arange(-180.1, 180.1, 60))
    ax.set_yticks(np.arange(-180.1, 180.1, 60))
    ax.annotate('', xy=(-180,0), xytext=(180, 0),
                arrowprops={'arrowstyle': '-', 'ls': 'dashed'})#, va='center')
    ax.annotate('', xy=(0,-180), xytext=(0, 180),
                arrowprops={'arrowstyle': '-', 'ls': 'dashed'})
    ax.scatter(phi, psi)
    ax.plot(phi, psi, '.')
    plt.show()
    #fig.set_size_inches(7.0, 8.0) # Changes figure size
    #fig.savefig('rama.png', dpi=600) # Saves figures
    
    
def main(path):
    '''
    '''
    initial_time = time.time()
    drawline = ''.join(("\n", "-" * 79, "\n"))
    fmt_step = ''.join(('\nData Processing Done.',
                        '\tUsed Time at This Step: {:.3f} seconds'))
    cols = ['AtomTyp',
            'ResName',
            'ChainID',
            'Seq_Num',
            'Coord_X',
            'Coord_Y',
            'Coord_Z',]
    
    pdbfiles = find_PDB_files(path)
    error_file = []
    for i, f in enumerate(pdbfiles):
        start_time = time.time()
        
        fmt_head = ''.join((drawline, "No.{:>5d},\tPDB ID:{:>5s}"))
        print(fmt_head.format(i + 1, f[:4].upper()))
        
        # Call function to read PDB file
        filename = os.path.join(path, f)
        pdb_info = pdb_reader(filename)[cols]
        
        # pick the Ramachandran subunit (no seq_gap case)
        factor = pdb_info.AtomTyp.str.match(subunits_dict['Rama'])
        pdb_info = pdb_info[factor].reset_index()
        #print("==== filter 1st round ====:\n", pdb_info)
        count = pdb_info.Seq_Num.value_counts()
        x = pdb_info[pdb_info.Seq_Num.isin(count[count < 3].index)]
        print("Count the Seq_Num:\n", x)
        #try:
        Ramachandran = build_Ramachandran_array(pdb_info)
        #except ValueError:
        #    print(''.join((drawline,
        #                   "The {} might have problem!",
        #                   drawline)).format(f))
        #    error_file.append(f)
            #pass
        
        steptime = time.time() - start_time
        print(fmt_step.format(steptime))
        
        # Call function to plot torsion angles
        phi_coord = Ramachandran[:,:4,:]
        psi_coord = Ramachandran[:,1:,:]
        
        # Call function to calculate torsion angles
        phi = calc_dihedral_angles(phi_coord)
        psi = calc_dihedral_angles(psi_coord)
        plot_rama(phi, psi)
        
    print(''.join((drawline, "The following files have problem!\n",
                   "{}\n" * len(error_file),
                   drawline)).format(*error_file))
    error_file = pd.DataFrame(error_file)
    error_file.to_csv(os.path.join(path,"error_file.csv"), sep=',', index=False)
    
    total_time = time.time() - initial_time
    fmt_end = "{:}Works Completed! Total Time: {:.4f} Seconds.\n"
    print(fmt_end.format(drawline, total_time))
    return pdb_info


if __name__ == "__main__":
    pdb_info = main(path)
