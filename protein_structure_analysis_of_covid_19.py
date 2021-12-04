# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from Bio.PDB import *
import os
import collections
location = '/home/saul/protein/544698_993839_bundle_archive/structures_4_3_2020'
os.chdir(location)
import xpdb   # this is the module described below
import pandas as pd
from collections import Counter
from itertools import groupby
import glob
import seaborn as sb
from matplotlib import pyplot as plt

#fig, axs = plt.subplots(ncols=3)

#This codes get the frequency distribution of each atoms in all protein types and outputs them to CSV files.
# read
sloppyparser = PDBParser(PERMISSIVE=True,
                         structure_builder=xpdb.SloppyStructureBuilder())

class Atoms:

    def __init__(self):

        #self.atomlist= []
        self.PL_PRO_C_terminal = pd.DataFrame(columns=['Atom', 'freq'])
        self.nsp2 = pd.DataFrame(columns=['Atom', 'freq'])
        self.nsp4= pd.DataFrame(columns=['Atom', 'freq'])
        self.M_protein = pd.DataFrame(columns=['Atom', 'freq'])
        self.Protein_3a = pd.DataFrame(columns=['Atom', 'freq'])
        self.nsp6 = pd.DataFrame(columns=['Atom', 'freq'])
        self.allData = pd.DataFrame(columns=['fileName','Atom', 'freq'])
        self.pdbfile = ''

    def getPDB(self, filename):
        self.pdbfile = filename.split(".")[0]
        #print("File Name {} !!!".format(self.pdbfile))
        structure = sloppyparser.get_structure('MD_system', filename)
        self.atoms = structure.get_atoms()
        print("File name {} has atoms {}".format(filename, self.atoms))
        self.__atomdict()

    def getResidue(self):
        print("Residue")

    def __atomdict(self):
        print("atomdict called!!!")
        atomlist = []
        for atom in self.atoms:
            # print(type(atom))
            atomlist.append(atom)
        atomdict = Counter(atomlist)
        #print(atomdict)
        self.__atomfreq(atomdict)

    def __atomfreq(self, atomdict):
        atoms = []
        for item, val in enumerate(atomdict):
            # print(" Item {} has the value of {}".format(item, val))
            value = str(val).strip('<>Atom ')
            atoms.append(value)
            #print(value)
            #atoms.append(value)

        atomfreq = collections.Counter(atoms) # count number of occurrences of atoms
        sortedatomfreq = {k: v for k, v in sorted(atomfreq.items(), key=lambda item: item[1], reverse=True)}
        #print(sortedatomfreq)
        self.__printAtomFreq(sortedatomfreq, atomfreq)

    def __printAtomFreq(self, sortedatomfreq, atomfreq):

        # Print Atom frequency by descented order
        print("File Name {} !!!".format(self.pdbfile))
        for item, val in enumerate(sortedatomfreq):
            # print(" Item {} has the value of {}".format(item, val))
            #print(" Atom {} has the value of {}".format(val, atomfreq[val]))
            # print(" Item {} has the value of {}".format(item, val))

            if self.pdbfile == 'PL_PRO_C_terminal':
                #print("File {} called".format(self.pdbfile ))
                self.__getPDBData(val,atomfreq[val], self.pdbfile)

            elif self.pdbfile == 'nsp2':
                #print("File {} called".format(self.pdbfile ))
                self.__getPDBData(val, atomfreq[val], self.pdbfile)

            elif self.pdbfile == 'nsp4':
                #print("File {} called".format(self.pdbfile ))
                self.__getPDBData(val, atomfreq[val], self.pdbfile)

            elif self.pdbfile == 'nsp6':
                #print("File {} called".format(self.pdbfile))
                self.__getPDBData(val, atomfreq[val], self.pdbfile)

            elif self.pdbfile == 'M_protein':
                #print("File {} called".format(self.pdbfile))
                self.__getPDBData(val, atomfreq[val], self.pdbfile)

            elif self.pdbfile == 'Protein_3a':
                #print("File {} called".format(self.pdbfile ))
                self.__getPDBData(val, atomfreq[val], self.pdbfile)
            else:
                print("File does not exist")

        print("PDB file ", self.pdbfile)
        self.__outputPDB()

    def __getPDBData(self, val, freq, pdbname):
        #insert data to the dataframes
        #print("PDB File::: ", pdbname)
        new_row = {'Atom': val, 'freq': freq}
        if pdbname == 'PL_PRO_C_terminal':
            self.PL_PRO_C_terminal = self.PL_PRO_C_terminal.append(new_row, ignore_index=True )
        elif pdbname == 'nsp2':
            self.nsp2 = self.nsp2.append(new_row, ignore_index=True)
        elif pdbname == 'nsp4':
            self.nsp4 = self.nsp4.append(new_row, ignore_index=True)
        elif pdbname == 'nsp6':
            self.nsp6 = self.nsp6.append(new_row, ignore_index=True)
        elif pdbname == 'M_protein':
            self.M_protein = self.M_protein.append(new_row, ignore_index=True)
        elif pdbname == 'Protein_3a':
            self.Protein_3a = self.Protein_3a.append(new_row, ignore_index=True)

    def __outputPDB(self):
        # import data to csv files
        self.PL_PRO_C_terminal.to_csv('PL_PRO_C_terminal_freq.csv', sep=',', index=False)
        self.nsp2.to_csv('nsp2_freq.csv', sep=',', index=False)
        self.nsp4.to_csv('nsp4_freq.csv', sep=',', index=False)
        self.nsp6.to_csv('nsp6_freq.csv', sep=',', index=False)
        self.M_protein.to_csv('M_protein.csv', sep=',', index=False)
        self.Protein_3a.to_csv('Protein_3a.csv', sep=',', index=False)

    def combineAllData(self):
        print("Append all data")
        self.PL_PRO_C_terminal['protein_name'] = 'PL_PRO_C_terminal'
        print('PL_PRO_C_terminal Columns ',self.PL_PRO_C_terminal.columns )
        self.nsp2['protein_name'] = 'nsp2'
        print('nsp2 ',self.nsp2.columns )
        self.nsp4['protein_name'] = 'nsp4'
        print('nsp4 Columns ',self.nsp4.columns )
        self.nsp6['protein_name'] = 'nsp6'
        print('nsp6 Columns ',self.nsp6.columns )
        self.M_protein['protein_name'] = 'M_protein'
        print('M_protein Columns ',self.M_protein.columns )
        self.Protein_3a['protein_name'] = 'Protein_3a'
        print('Protein_3a Columns ',self.Protein_3a.columns )

        # protein structure size
        print("PL_PRO_C_terminal Size", len(self.PL_PRO_C_terminal))
        print("nsp2 Size ", len(self.nsp2))
        print("nsp4 Size ", len(self.nsp4))
        print("nsp6 Size", len(self.nsp6))
        print("M_protein Size", len(self.M_protein))
        print("Protein_3a Size", len(self.Protein_3a))


        alldata = self.PL_PRO_C_terminal.append(self.nsp2, ignore_index=True)
        alldata = alldata.append(self.nsp4, ignore_index=True)
        alldata = alldata.append(self.nsp6, ignore_index=True)
        alldata = alldata.append(self.M_protein, ignore_index=True)
        alldata = alldata.append(self.Protein_3a, ignore_index=True)

        print("All data columns ", alldata.columns)
        print("All data length ", len(alldata))
        self.__visualiseAll(alldata)

    def visualiseData(self, filename):
        print("Visualise Data")
        print("File Name !!!!!!", filename )

        if filename.split(".")[0] == 'PL_PRO_C_terminal':
            print("PL_PRO_C_terminal called")
            sb.barplot(data=self.PL_PRO_C_terminal, x='Atom', y='freq')
            plt.show()
        elif filename.split(".")[0] == 'nsp2':
            sb.barplot(data=self.nsp2, x='Atom', y='freq')
            plt.show()
        elif filename.split(".")[0] == 'nsp4':
            sb.barplot(data=self.nsp4, x='Atom', y='freq')
            plt.show()
        elif filename.split(".")[0] == 'nsp6':
            sb.barplot(data=self.nsp6, x='Atom', y='freq')
            plt.show()
        elif filename.split(".")[0] == 'M_protein':
            sb.barplot(data=self.M_protein, x='Atom', y='freq')
            plt.show()
        elif filename.split(".")[0] == 'Protein_3a':
            sb.barplot(data=self.Protein_3a, x='Atom', y='freq')
            plt.show()

    def __visualiseAll(self, alldata):
        print("Visualise All")
        # sb.barplot(data=self.PL_PRO_C_terminal, x='Atom', y='freq').set_title('PL_PRO_C_terminal')
        # sb.barplot(data=self.nsp2, x='Atom', y='freq').set_title('nsp2')
        # sb.barplot(data=self.nsp4, x='Atom', y='freq').set_title('nsp4')
        # sb.barplot(data=self.nsp6, x='Atom', y='freq').set_title('nsp6')
        # sb.barplot(data=self.M_protein, x='Atom', y='freq').set_title('M_protein')
        # sb.barplot(data=self.Protein_3a, x='Atom', y='freq').set_title('Protein_3a')
        #plt.hist([self.nsp4['freq'], self.nsp2['freq']])
        #plt.show()

        #Show frequency Distribution of each protein types of COVID-19
        sb.barplot(data=alldata, x= 'protein_name', y='freq', estimator=sum).set_title('Protein Frequency')
        plt.show()

def getFileNames(location):
    files = []
    print('getFileNames called!!!')
    #print(location)
    for file_name in glob.iglob(location + '/*.pdb', recursive=True):
        #print(file_name)
        #print(file_name.split('/')[-1])
        files.append(file_name.split('/')[-1])
    return files

if __name__ == '__main__':
    proteins = Atoms()
    #proteins.atomdict()
    fileNames = getFileNames(location)
    #print(fileNames)
    for name in range(len(fileNames)):
        print(fileNames[name])
        proteins.getPDB(fileNames[name])
        #proteins.visualiseData(fileNames[name])
    #proteins.visualiseAll()
    proteins.combineAllData()


