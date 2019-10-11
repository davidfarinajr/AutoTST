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
import logging
import rmgpy
from rmgpy.molecule import Molecule
# from arkane.main import Arkane as RMGArkane, KineticsJob, StatMechJob, ThermoJob
from ase import Atom, Atoms
from cclib.io import ccread

from autotst.species import Species, Conformer

FORMAT = "%(filename)s:%(lineno)d %(funcName)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

class Arkane_Input():

    def __init__(self, molecule=None, modelChemistry=None, directory=None, gaussian_log_path=None, energy_log_path=None, geometry_log_path=None, frequencies_log_path=None):
        
        self.molecule = molecule
        self.modelChemistry = modelChemistry
        self.smiles = self.molecule.toSMILES()
        self.label = self.smiles + '_arkane'
        self.directory = directory
        if gaussian_log_path:
            self.energy_log_path = self.geometry_log_path = self.frequencies_log_path = gaussian_log_path
        else:
            self.energy_log_path = energy_log_path
            self.geometry_log_path = geometry_log_path
            self.frequencies_log_path = frequencies_log_path

        self.command = '$RMGpy/Arkane.py'

    def write_molecule_file(self):

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        linear = self.molecule.isLinear()
        sym_num = self.molecule.calculateSymmetryNumber()
        mult = self.molecule.multiplicity
        path = os.path.join(self.directory,self.smiles+'.py')
        energy_path = os.path.basename(self.energy_log_path)
        geometry_path = os.path.basename(self.geometry_log_path)
        frequencies_path = os.path.basename(self.frequencies_log_path)

        with open(path,'w+') as f:
            f.write('#SMILES = {}\n'.format(self.smiles))
            f.write('linear = {}\n'.format(linear))
            f.write('externalSymmetry = {}\n'.format(sym_num))
            f.write('spinMultiplicity = {}\n'.format(mult))
            f.write('opticalIsomers = 1\n')
            f.write('geometry = GaussianLog("{}")\n'.format(geometry_path))
            f.write('frequencies = GaussianLog("{}")\n'.format(frequencies_path))
            f.write('energy = GaussianLog("{}")\n'.format(energy_path))
            f.close()

        return path

    def write_arkane_input(self,frequency_scale_factor=1.0, useIsodesmicReactions=False, useAtomCorrections=False, useBondCorrections=False, 
                            useHinderedRotors=False, constraint_classes=None, n_reactions_max = 50, 
                            max_ref_uncertainty = None, deviation_coeff = 3.0):
        
        molecule_file_path = os.path.join(self.directory,self.smiles+'.py')
        if not os.path.exists(molecule_file_path):
            molecule_file_path = self.write_molecule_file()
        
        arkane_input_path = os.path.join(self.directory,'arkane_input' +'.py')
    
        if useIsodesmicReactions is True:
            useAtomCorrections = useBondCorrections = False

        with open(arkane_input_path,'w+') as f:
            f.write('modelChemistry = "{}"\n'.format(self.modelChemistry))
            f.write('frequencyScaleFactor = {}\n'.format(frequency_scale_factor))
            f.write('useAtomCorrections = {}\n'.format(useAtomCorrections))
            f.write('useBondCorrections = {}\n'.format(useBondCorrections))
            f.write('useHinderedRotors = {}\n'.format(useHinderedRotors))
            if useIsodesmicReactions is True:
                f.write('useIsodesmicReactions = {}\n'.format(useIsodesmicReactions))
                f.write('n_reactions_max = {}\n'.format(n_reactions_max))
                f.write('max_ref_uncertainty = {}\n'.format(max_ref_uncertainty))
                f.write('deviation_coeff = {}\n'.format(deviation_coeff))
            f.write('\n')
            f.write('species("{0}","./{1}.py",structure = SMILES("{1}"))\n'.format(1,self.smiles))
            f.write('\n')
            f.write('thermo("{}","NASA")\n'.format(1))
            f.close()
        
        return arkane_input_path
    