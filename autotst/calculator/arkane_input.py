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

    def __init__(self, conformer=None, modelChemistry=None, directory=None, gaussian_log_path=None, energy_log_path=None, geometry_log_path=None, frequencies_log_path=None, rotors_dir=None):
        
        self.conformer = conformer
        self.modelChemistry = modelChemistry
        self.label = self.conformer.smiles + '_arkane'
        self.directory = directory
        if gaussian_log_path:
            self.energy_log_path = self.geometry_log_path = self.frequencies_log_path = gaussian_log_path
        else:
            self.energy_log_path = energy_log_path
            self.geometry_log_path = geometry_log_path
            self.frequencies_log_path = frequencies_log_path
        
        self.rotors_dir = rotors_dir
        self.command = '$RMGpy/Arkane.py'

    def get_rotor_info(self,torsion):
        """
        Formats and returns info about torsion as it should appear in an Arkane species.py

        The following are needed for an Arkane input file:
        - scanLog :: Gaussian output log of freq calculation on optimized geometry
        - pivots  :: torsion center: j,k of i,j,k,l (Note Arkane begins indexing with 1)
        - top     :: ID of all atoms in one top (Note Arkane begins indexing with 1)

        Parameters:
        - conformer (Conformer): autotst conformer object
        - torsion (Torsion): autotst torsion object 


        Returns:
        - info (str): a string containing all of the relevant information for a hindered rotor scan
        """

        _, j, k, _ = torsion.atom_indices

        # Adjusted since mol's IDs start from 0 while Arkane's start from 1
        tor_center_adj = [j+1, k+1]

   
        tor_log = os.path.join(
            self.rotors_dir,
            self.conformer.smiles + '_36by10_{0}_{1}.log'.format(j, k)
        )

        if not os.path.exists(tor_log):
            logging.info(
                "Torsion log file does not exist for {}".format(torsion))
            return ""

        tor_log = os.path.join(os.path.basename(self.rotors_dir), self.conformer.smiles + '_36by10_{0}_{1}.log'.format(j, k))
        top_IDs = []
        for num, tf in enumerate(torsion.mask):
            if tf:
                top_IDs.append(num)

        # Adjusted to start from 1 instead of 0
        top_IDs_adj = [ID+1 for ID in top_IDs]

        info = "     HinderedRotor(scanLog=Log('{0}'), pivots={1}, top={2}, fit='best'),".format(
            tor_log, tor_center_adj, top_IDs_adj)

        return info

    def write_molecule_file(self):

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        linear = self.conformer.rmg_molecule.is_linear()
        sym_num = self.conformer.rmg_molecule.calculate_symmetry_number()
        mult = self.conformer.rmg_molecule.multiplicity
        path = os.path.join(
            self.directory, self.conformer.rmg_molecule.smiles +'.py')
        energy_path = os.path.basename(self.energy_log_path)
        geometry_path = os.path.basename(self.geometry_log_path)
        frequencies_path = os.path.basename(self.frequencies_log_path)

        if sym_num == 0.5:
            logging.warning("RMG sym number is 0.5!  Setting sym number to 1.0")
            sym_num = 1.0

        with open(path,'w') as f:
            f.write('#SMILES = {}\n'.format(self.conformer.rmg_molecule.smiles))
            f.write('linear = {}\n'.format(linear))
            f.write('externalSymmetry = {}\n'.format(sym_num))
            f.write('spinMultiplicity = {}\n'.format(mult))
            f.write('opticalIsomers = 1\n')
            f.write('geometry = GaussianLog("{}")\n'.format(geometry_path))
            f.write('frequencies = GaussianLog("{}")\n'.format(frequencies_path))
            f.write('energy = GaussianLog("{}")\n'.format(energy_path))
            if self.rotors_dir is not None:
                f.write('rotors = [\n')
                for torsion in self.conformer.torsions:
                    logging.info(torsion)
                    info = self.get_rotor_info(torsion)
                    logging.info(info)
                    if len(info) == 0:
                        continue
                    f.write(info + '\n')
                f.write(']')
            f.close()

        return path

    def write_arkane_input(self, frequency_scale_factor=1.0, useIsodesmicReactions=False, useAtomCorrections=False, useBondCorrections=False,
                            useHinderedRotors=False, constraint_classes=None, n_reactions_max = 50, 
                            max_ref_uncertainty = None, deviation_coeff = 3.0):
        
        molecule_file_path = os.path.join(
            self.directory, self.conformer.rmg_molecule.smiles + '.py')
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
            f.write('species("{0}","./{1}.py",structure = SMILES("{1}"))\n'.format(
                1, self.conformer.rmg_molecule.smiles))
            f.write('\n')
            f.write('thermo("{}","NASA")\n'.format(1))
            f.close()
        
        return arkane_input_path
    
