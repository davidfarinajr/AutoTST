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

    def __init__(self,directory='.',conformer=None):
        
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
        
    def __repr__(self):
        return '<Orca Calculator>'

    def load_conformer_attributes(self):
        assert self.conformer is not None,'Must provide an AutoTST conformer object'
        assert isinstance(self.conformer,Conformer),'conformer must be an autotst conformer object'
        if isinstance(self.conformer, TS):
            self.label = self.conformer.reaction_label
        else:
            self.label = self.conformer.smiles

        if '(' in self.label or '#' in self.label:
            self.base = label.replace('(', '{').replace(')', '}').replace('#', '=-')
        else:
            self.base = self.label

        try:
            self.charge = self.conformer.rmg_molecule.getNetCharge()
        except:
            logging.warning('could not get charge for conformer...setting charge to None')
            self.charge = None
        try:
            self.mult = self.conformer.rmg_molecule.multiplicity
        except:
            logging.warning('could not get mulitpicity of conformer...setting multiplicty to None')
            self.mult = None
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

        assert None not in [self.mult,self.charge,self.coords]
        
        if directory is None:
            directory = self.directory
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)

        outfile = os.path.join(directory,self.label+'_fod.inp')

        with open(outfile, 'w') as f:
            f.write('# FOD anaylsis for {} \n'.format(self.label))
            f.write('! FOD \n')
            f.write('\n')
            f.write('%pal nprocs 4 end \n')
            f.write('%scf\n  MaxIter  600\nend\n')
            f.write('%base "{}_fod" \n'.format(self.base))
            f.write('*xyz {} {}\n'.format(self.charge, self.mult))
            f.write(self.coords)
            f.write('*\n')

    def write_sp_input(self, directory=None, nprocs=20, mem='110gb', method = 'ccsd(t)-f12', basis= 'cc-pvdz-f12',
                        scf_convergence = 'verytightscf', max_iter = '600'):
        """
        A method to write single point energy calculation input files for ORCA.
        
        :param path: path for input file
        :param settings:
            :param nprocs: number of processors to run the calculation on (default is 20)
            :param mem: amount of memory requested (in gb or mb) to run the job (default is 110gb)
            :param method: calculation method to run.  Supported methods are (hf,ccsd(t),ccsd(t)-f12). Default is ccsd(t)-f12
            :param basis: basis set for calculation. Default is cc-pvdz-f12
            :param scf_convergence : convergence option for scf. supported options are (normalscf,loosescf,sloppyscf,strongscf,tightscf,verytightscf,extremescf). Default is 'verytightscf'
            :param max_iter : maximum number of scf iterations to reach convergence criteria. (default is 600)
        """
        
        assert None not in [self.mult, self.charge, self.coords]

        if directory is None:
            directory = self.directory
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)

        nprocs = int(nprocs)
        mem = mem.lower()
        method = method.lower()
        basis = basis.lower()
        scf_convergence = scf_convergence.lower()
        max_iter = max_iter.lower()

        suppported_methods = ['hf','ccsd(t)','ccsd(t)-f12']

        assert method in suppported_methods,'is appears {0} is not a supported method. Please select a method from {1}'.format(method,suppported_methods)
        
        scf_convergence_options = 'normalscf loosescf sloppyscf strongscf tightscf verytightscf extremescf'
        assert scf_convergence in scf_convergence_options


        if 'gb' in mem:
            mem_mb = float(mem.strip('gb')) * 1000
        elif 'mb' in mem:
            mem_mb = float(mem.strip('mb'))
        else: #assume GB
            mem_mb = float(mem) * 1000
        mem_proc = int(mem_mb/nprocs)

        if 'dz' in basis:
            basis_label = 'dz'
        elif 'tz' in basis:
            basis_label = 'tz'
        elif 'qz' in basis:
            basis_label = 'qz'
        elif '5z' in basis:
            basis_label = '5z'
        elif '6z' in basis:
            basis_label = '6z'
        else:
            basis_label = basis
        if 'aug' in basis:
            basis_label = 'a' + basis_label

        if 'f12' in method and 'f12' not in basis:
            logging.warning('An F12 method was called, but an f12 basis set was not chosen')
            logging.info('trying to find an f12 basis...')
            if 'cc' in basis:
                basis = 'cc-pv{}-f12'.format(basis_label.strip('a'))
                logging.info('{} will be used for the f12 calculation'.format(basis))
            else:
                logging.info(
                    'Could not find f12 basis set. {} will be used for the f12 calculation'.format(basis))

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

        auxiliary_basis_sets_dict = {
            'cc-pvdz-f12': 'CC-PVDZ-F12-CABS CC-PVTZ/C',
            'cc-pvtz-f12': 'CC-PVTZ-F12-CABS CC-PVQZ/C',
            'cc-pvqz-f12': 'CC-PVQZ-F12-CABS CC-PVQZ/C'}

        if basis in auxiliary_basis_sets_dict.keys():
            aux_basis = auxiliary_basis_sets_dict.get(basis)
            if int(self.mult) == 1 or 'hf' in method:
                aux_basis = aux_basis.split(' ')[0]
        else:
            aux_basis = ''

        if method == 'ccsd(t)-f12' and int(self.mult) != 1:
            method = 'ccsd(t)-f12/ri'

        file_name = self.label + '_' + method + '{' + str(basis) + '}' + '.inp'
        if '/' in file_name:
            file_name = file_name.replace('/','-')

        file_path = os.path.join(directory,file_name)

        base = self.base + '_' + method + '_' + basis
        if '(' in base or '#' in base or '/' in base:
            base = base.replace('(', '{').replace(')', '}').replace('#', '=-').replace('/','-')

        with open(file_path, 'w') as f:
            f.write('# {0}/{1} calculation for {2} \n'.format(method,basis,self.label))
            f.write('! {0} {1} {2} {3} {4}\n'.format(hf.upper(),method.upper(),basis.upper(),aux_basis.upper(),scf_convergence.upper()))
            f.write('\n')
            f.write('%pal nprocs {0} end \n'.format(nprocs))
            f.write('%maxcore {0}\n'.format(mem_proc))
            f.write('%scf\n  MaxIter  {0}\nend\n'.format(max_iter))
            f.write('%base "{0}" \n'.format(base))
            f.write('*xyz {0} {1}\n'.format(self.charge, self.mult))
            f.write(self.coords)
            f.write('*\n')
        
        return file_name
    
    def write_extrapolation_input(self, directory='.', nprocs=20, mem='110gb', option='EP3', basis_family='aug-cc', 
                                scf_convergence='Tightscf', method='DLPNO-CCSD(T)', method_details='tightpno', 
                                n=3, m=4):

        if directory is None:
            directory = self.directory
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        option = str(option).lower()
        basis_family = basis_family.lower()
        method = method.lower()
        scf_convergence= scf_convergence.lower()
        mem = mem.lower()

        if 'gb' in mem:
            mem_mb = float(mem.strip('gb')) * 1000
        elif 'mb' in mem:
            mem_mb = float(mem.strip('mb'))
        else:  # assume GB
            mem_mb = float(mem) * 1000
        mem_proc = int(mem_mb/nprocs)
        
        scf_convergence_options = 'normalscf loosescf sloppyscf strongscf tightscf verytightscf extremescf'
        assert scf_convergence in scf_convergence_options

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

        file_path = os.path.join(directory, file_name)

        base = self.base + method + '_extrapolate_' + option + basis_family 
        if '(' in base or '#' in base or '/' in base:
            base = base.replace('(', '{').replace(')', '}').replace(
                '#', '=-').replace('/', '-')

        with open(file_path, 'w') as f:
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
            f.write('%maxcore {0}\n'.format(mem_proc))
            f.write('%base "{0}" \n'.format(base))
            f.write('*xyz {0} {1}\n'.format(self.charge, self.mult))
            f.write(self.coords)
            f.write('*\n')

        return file_name

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

        if self.check_NormalTermination(path):
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
            

        


            



    
