#!/usr/bin/python
# -*- coding: utf-8 -*-

##########################################################################
#
#   AutoTST - Automated Transition State Theory
#
#   Copyright (c) 2015-2018 Prof. Richard H. West (r.west@northeastern.edu)
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the 'Software'),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#
##########################################################################

import os
from autotst.species import Species, Conformer
from autotst.reaction import Reaction, TS
import logging

class Orca():
    """
    A class for writing and reading Orca jobs.
    """
    def __init__(self,conformer=None,directory='.',partition='general',time='1-00:00:00',nprocs=20,mem=110):
        
        self.command = 'orca'
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(self.directory)
        if conformer:
            assert isinstance(conformer, Conformer), 'conformer must be an autotst conformer object'
            self.conformer = conformer
            self.load_conformer_attributes()
        else:
            self.label = None
            self.conformer = None

        self.nprocs = int(nprocs)
        self.mem = str(mem).upper()
        self.time = str(time)
        self.partition = partition

        self.mem_per_proc = self.get_mem_per_proc()
        
    def get_mem_per_proc(self,mem=None,nprocs=None):
        """
        A method to calculate the memory per processor (in MB) for an Orca calculation.
        Returns mem_per_proc (int)

        :param mem (str or int): Total memory available in GB or MB (Assumes GB if unitless).  
        If not specified, self.mem will be used.
        :param nprocs (str or int): Number of processors.
        If not specified, self.nprocs will be used.
        """

        # Use mem is specified, else use self.mem
        if mem is None:
            assert self.mem is not None
            mem = str(self.mem).upper()
        else:
            mem = str(mem).upper()

        # Use nprocs is specified, else use self.nprocs
        if nprocs is None:
            assert self.nprocs is not None
            nprocs = int(self.nprocs)
        else:
            nprocs = int(nprocs)

        # Convert mem to MB
        if 'GB' in mem:
            mem_mb = float(mem.strip('GB')) * 1000
        elif 'MB' in mem:
            mem_mb = float(mem.strip('MB'))
        else:  # assume GB
            mem_mb = float(mem) * 1000
        
        # Calculate mem_per_proc
        mem_per_proc = int(mem_mb/nprocs)

        # Return mem_per_proc
        return mem_per_proc

    def __repr__(self):
        return '<Orca Calculator>'

    def load_conformer_attributes(self):
        """
        A method that loads attributes from a conformer attatched to an Orca calculator instance. 
        Orca calculator instance must be initialized with or provided an AutoTST conformer before calling this method.
        The method tries to get smiles, charge, multiplicity, and coordinates from the conformer.
        If found, it creates an attribute for that property.
        """

        # Assert AutoTST conformer is attached to Orca Calculator
        assert self.conformer is not None,'Must provide an AutoTST conformer object'
        assert isinstance(self.conformer,Conformer),'conformer must be an autotst conformer object'
        
        # Assign smiles or reaction label as label attribute
        if isinstance(self.conformer, TS):
            self.label = self.conformer.reaction_label
        else:
            self.label = self.conformer.smiles

        # Replace problematic characters in temporary files and assign to base attribute
        if '(' in self.label or '#' in self.label:
            self.base = self.label.replace('(', '{').replace(')', '}').replace('#', '=-')
        else:
            self.base = self.label
  
        self.charge = self.conformer.rmg_molecule.get_net_charge()
        self.mult = self.conformer.rmg_molecule.multiplicity

        try:
            self.coords = self.conformer.get_xyz_block()
        except:
            logging.warning('could not get coordinates of conformer...setting coords to None')
            self.coords = None


    def write_fod_input(self,directory=None):
        """
        Generates input files to run finite temperaure DFT to determine the Fractional Occupation number weighted Density (FOD number).
        Uses the default functional, basis set, and SmearTemp (TPSS, def2-TZVP, 5000 K) in Orca.
        See resource for more information:
        Bauer, C. A., Hansen, A., & Grimme, S. (2017). The Fractional Occupation Number Weighted Density as a Versatile Analysis Tool for Molecules with a Complicated Electronic Structure. 
        Chemistry - A European Journal, 23(25), 6150–6164. https://doi.org/10.1002/chem.201604682
        """

        # Make sure we have required properties of conformer to run the job
        assert None not in [self.mult,self.charge,self.coords]
        
        # If directory is not specified, use the instance directory
        if directory is None:
            directory = self.directory
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Path for FOD input file
        outfile = os.path.join(directory,self.label+'_fod.inp')

        # Write FOD input
        with open(outfile, 'w+') as f:
            f.write('# FOD anaylsis for {} \n'.format(self.label))
            f.write('! FOD \n')
            f.write('\n')
            f.write('%pal nprocs {} end \n'.format(str(self.nprocs)))
            f.write('%scf\n  MaxIter  600\nend\n')
            f.write('%base "{}_fod" \n'.format(self.base))
            f.write('*xyz {} {}\n'.format(self.charge, self.mult))
            f.write(self.coords)
            f.write('*\n')

    def write_sp_input(self, directory=None, nprocs=None, mem=None, method = 'ccsd(t)-f12', basis= 'cc-pvdz-f12', atom_basis = {'Cl':'cc-pvt(+d)z','Br':'aug-cc-pvtz'}, use_atom_basis = False,
                        scf_convergence = 'verytightscf', max_iter = '600'):
        """
        A method to write single point energy calculation input files for ORCA.
        
        :param directory (str): directory to write input file (instance directory will be used if not specified)
        :param nprocs (str or int): number of processors to run the calculation on.
        (instance nprocs will be used if not specified)
        :param mem (str or int): amount of memory requested (in GB or MB) to run the job
        (instance mem will be used if not specified)
        :param method (str): calculation method to run.  Supported methods are (hf,ccsd(t),ccsd(t)-f12). Default is ccsd(t)-f12
        :param basis (str): basis set for calculation. Default is cc-pvdz-f12.
        :param atom_basis (dict): A dictionary of atom, basis pairs for atom-specific basis sets.
        :param use_atom_basis (T/F): If True, the atom-specific basis set in atom_basis dict will override the general basis set for that atom if the atom is in the molecule (default is False).
        :param scf_convergence : convergence option for scf. supported options are (normalscf,loosescf,sloppyscf,strongscf,tightscf,verytightscf,extremescf). Default is 'verytightscf'
        :param max_iter : maximum number of scf iterations to reach convergence criteria. (default is 600)

        Returns label for file_name
        """
        
        # Make sure we have required properties of conformer to run the job
        assert None not in [self.mult, self.charge, self.coords]

        # Get mem_per_proc needed to write input
        if (mem or nprocs) is not None:
            if (mem and nprocs) is not None:
                mem_per_proc = self.get_mem_per_proc(mem=mem,nprocs=nprocs)
            elif nprocs is not None:
                mem = self.mem
                mem_per_proc = self.get_mem_per_proc(nprocs=nprocs)
            else:
                nprocs = self.nprocs
                mem_per_proc = self.get_mem_per_proc(mem=mem)
        else:
            mem_per_proc = self.mem_per_proc
            mem = self.mem
            nprocs = self.nprocs

        # if directory is not specified,
        # Use Orca instance directory
        if directory is None:
            directory = self.directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Convert parameters to lowercase strings
        method = method.lower()
        basis = basis.lower()
        scf_convergence = scf_convergence.lower()
        max_iter = max_iter.lower()

        # list of currently supported methods
        suppported_methods = ['hf','ccsd(t)','ccsd(t)-f12']

        # Assert method is supported
        assert method in suppported_methods,'is appears {0} is not a supported method. Please select a method from {1}'.format(method,suppported_methods)
        
        # Assert scf_convergence is supported
        scf_convergence_options = 'normalscf loosescf sloppyscf strongscf tightscf verytightscf extremescf'
        assert scf_convergence in scf_convergence_options


        # Select UHF for open-shell,
        # RHF for closed shell molecules
        if 'hf' not in method:
            if int(self.mult) == 1:
                hf = 'rhf'
            else:
                hf = 'uhf'
        else:
            if int(self.mult) == 1:
                method = 'rhf'
                hf = ''
            else:
                method = 'uhf'
                hf = ''

        # Auxiliary basis sets for F12 methods
        auxiliary_basis_sets_dict = {
            'cc-pvdz-f12': 'CC-PVDZ-F12-CABS CC-PVTZ/C',
            'cc-pvtz-f12': 'CC-PVTZ-F12-CABS CC-PVQZ/C',
            'cc-pvqz-f12': 'CC-PVQZ-F12-CABS CC-PVQZ/C'}

        # Get auxiliary basis sets 
        if basis in auxiliary_basis_sets_dict.keys():
            aux_basis = auxiliary_basis_sets_dict.get(basis)
            if int(self.mult) == 1 or 'hf' in method:
                aux_basis = aux_basis.split(' ')[0]
        else:
            aux_basis = ''

        # Get atom-specific basis sets if use_atom_basis is True
        if use_atom_basis is True:
            assert atom_basis is not None
            assert isinstance(atom_basis,dict)
            new_basis = ''
            new_basis_strs = []
            atom_symbols = [atom.lower() for atom in self.conformer.ase_molecule.get_chemical_symbols()]
            for atom, b in atom_basis.iteritems():
                if atom.lower() in atom_symbols:
                    new_basis = new_basis + '{}={},'.format(atom.lower(), b)
                    new_basis_strs.append('  newGTO {} "{}" end\n'.format(atom, b))
        
        # Use RI for open-shell (Orca only supprts RI-F12 methods for open-shell)
        if method == 'ccsd(t)-f12' and int(self.mult) != 1:
            method = 'ccsd(t)-f12/ri'
        
        # Create file name
        if use_atom_basis is True and len(new_basis_strs) > 0:
            file_name = self.label + '_' + method + '[' + str(basis) + ',' + str(new_basis)+ ']' + '.inp'
        else:
            file_name = self.label + '_' + method + '[' + str(basis) + ']' + '.inp'
        if '/' in file_name:
            file_name = file_name.replace('/','-')

        # Create Label
        label = file_name.strip('.inp')

        # Create file path
        file_path = os.path.join(directory,file_name)

        # Create base for scratch files
        base = self.base + '_' + method + '_' + basis
        if use_atom_basis is True and len(new_basis_strs) > 0:
            base = base + str(new_basis)
        if '(' in base or '#' in base or '/' in base:
            base = base.replace('(', '{').replace(')', '}').replace('#', '=-').replace('/','-')

        # Write sp input
        with open(file_path, 'w+') as f:
            f.write('# {0}/{1} calculation for {2} \n'.format(method,basis,self.label))
            f.write('! {0} {1} {2} {3} {4} PRINTBASIS\n'.format(hf.upper(),method.upper(),basis.upper(),aux_basis.upper(),scf_convergence.upper()))
            f.write('\n')
            f.write('%pal nprocs {0} end \n'.format(nprocs))
            f.write('%maxcore {0}\n'.format(mem_per_proc))
            f.write('%scf\n  MaxIter  {0}\nend\n'.format(max_iter))
            f.write('%base "{0}" \n'.format(base))
            if use_atom_basis is True and len(new_basis_strs) > 0:
                f.write('%basis\n')
                f.writelines(new_basis_strs)
                f.write('end\n')
            f.write('*xyz {0} {1}\n'.format(self.charge, self.mult))
            f.write(self.coords)
            f.write('*\n')
        
        return label
    
    def write_extrapolation_input(self, directory=None, nprocs=None, mem=None, option='EP3', basis_family='aug-cc', 
                                scf_convergence='verytightscf', method='DLPNO-CCSD(T)', method_details='tightpno', 
                                n=3, m=4):
        """
        A method to write single point automated CBS extrapolation calculation input files for ORCA.
        
        :param directory (str): directory to write input file (instance directory will be used if not specified)
        :param nprocs (str or int): number of processors to run the calculation on
        (instance nprocs will be used if not specified)
        :param mem (str or int): amount of memory requested (in GB or MB) to run the job.
        (instance mem will be used if not specified)
        :param option (str): Extraplotation technique to use.  Supported options are ('2','3','ep2','ep3'). See Orca manual for details.
        :param basis_family (str): basis sets to use for calculation. Supported basis families are ('cc','aug-cc', 'cc-core', 'ano', 'saug-ano', 'aug-ano', 'def2'). Default is aug-cc.
        :param scf_convergence : convergence option for scf. supported options are (normalscf,loosescf,sloppyscf,strongscf,tightscf,verytightscf,extremescf). Default is 'verytightscf'
        :param method: A method to use for single point calculations. See Orca manual for supported methods. Default is DLPNO-CCSD(T).
        :param method_details (str): Additional details for method. Default is tightpno.
        :param n (int): Cardinal number of smaller basis set (ex. cc-pvtz = 3)
        :param m (int): Cardinal number of larger basis set (ex. cc-pvqz = 4)

        Returns label for file_name
        """

        # Get mem_per_proc needed to write input
        if (mem or nprocs) is not None:
            if (mem and nprocs) is not None:
                mem_per_proc = self.get_mem_per_proc(mem=mem, nprocs=nprocs)
            elif nprocs is not None:
                mem = self.mem
                mem_per_proc = self.get_mem_per_proc(nprocs=nprocs)
            else:
                nprocs = self.nprocs
                mem_per_proc = self.get_mem_per_proc(mem=mem)
        else:
            mem_per_proc = self.mem_per_proc
            mem = self.mem
            nprocs = self.nprocs

        # if directory is not specified,
        # Use Orca instance directory
        if directory is None:
            directory = self.directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Convert paramters to lowercase strings
        option = str(option).lower()
        basis_family = basis_family.lower()
        method = method.lower()
        scf_convergence= scf_convergence.lower()
        mem = mem.lower()
        
        # Asset scf_convergence is supported
        scf_convergence_options = 'normalscf loosescf sloppyscf strongscf tightscf verytightscf extremescf'
        assert scf_convergence in scf_convergence_options

        # if 2 or ep2 option and m is None, m=4
        assert option in ['2','3','ep2','ep3']
        if option in ['2', 'ep2']:
            if m is None:
                m = 4
        if option in ['ep2','ep3']:
            assert method in ['dlpno-ccsd(t)','mp2']        

        if n is None:
            n = 3
        else:
            n = int(n)

        if m is not None:
            m = int(m)
            assert m == n+1

        # Assert basis set familt is in supported families
        assert basis_family in ['cc','aug-cc', 'cc-core', 'ano', 'saug-ano', 'aug-ano', 'def2']
        
        if int(self.mult) == 1:
            hf = 'rhf'
        else:
            hf = 'uhf'

        if option in ['2','ep2']:
            file_name = self.label + '_' + method + '-extrapolate-' + option + '{' + str(basis_family) + '}' + str(n)+'/'+str(m) + '.inp'
        else:
            file_name = self.label + '_' + method + '-extrapolate-' + option + '{' + str(basis_family) + '}' + str(n) + '.inp'

        if '/' in file_name:
            file_name = file_name.replace('/', '-')

        label = file_name.strip('.inp')

        file_path = os.path.join(directory, file_name)

        base = self.base + method + '_extrapolate_' + option + basis_family 
        if '(' in base or '#' in base or '/' in base:
            base = base.replace('(', '{').replace(')', '}').replace(
                '#', '=-').replace('/', '-')

        with open(file_path, 'w+') as f:
            f.write('# {0}/{1} extrapolation {2} for {3} \n'.format(method, basis_family, option,self.label))
            if option == 'ep3':
                f.write('! {0} Extrapolate{1}({2},{3},{4}) {5}'.format(
                    hf.upper(),option.upper(),basis_family,method,method_details,scf_convergence)
                )
            elif option == 'ep2':
                f.write('! {0} Extrapolate{1}({2}/{3},{4},{5},{6}) {7}'.format(
                    hf.upper(),option.upper(),n,m,basis_family,method,method_details,scf_convergence)
                )
            elif option == '2':
                f.write('! {0} {1} AutoAux Extrapolate({2}/{3},{4}) {5}'.format(
                    hf.upper(),method.upper(),n,m,basis_family,scf_convergence)
                )
            elif option == '3':
                f.write('! {0} {1} AutoAux Extrapolate({2},{3}) {4}'.format(
                    hf.upper(),method.upper(),n,basis_family,scf_convergence)
                )
            f.write('\n')
            f.write('%pal nprocs {0} end \n'.format(nprocs))
            f.write('%maxcore {0}\n'.format(mem_per_proc))
            f.write('%base "{0}" \n'.format(base))
            f.write('*xyz {0} {1}\n'.format(self.charge, self.mult))
            f.write(self.coords)
            f.write('*\n')

        return label

    def check_NormalTermination(self,path):
        """
        checks if an Orca job terminated normally.
        Returns True is normal termination and False if something went wrong.
        """
        assert os.path.exists(path), 'It seems {} is not a valid path'.format(path)

        lines = open(path,'r').readlines()[-5:]
        complete = False
        for line in lines:
            if "ORCA TERMINATED NORMALLY" in line:
                complete = True
                break
        return complete
        
    def read_fod_log(self,path):
        """
        Reads an FOD log to get the FOD number.
        Returns FOD number if log terminated normally and FOD number can be found.
        """
        assert os.path.exists(path),'It seems {} is not a valid path'.format(path)

        if self.check_normal_termination(path):
            N_FOD = None
            for line in open(path,'r').readlines():
                if 'N_FOD =' in line:
                    N_FOD = float(line.split(" ")[-1])
                    break
            if N_FOD:
                logging.info("the FOD number is {}".format(N_FOD))
                return N_FOD
            else:
                logging.info("It appears that the orca terminated normally for {}, but we couldn't find the FOD number".format(path))
        else:
            logging.info('It appears the orca FOD job for {} did not terminate normally'.format(path))
            