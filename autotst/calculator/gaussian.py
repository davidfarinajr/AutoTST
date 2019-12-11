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

from rmgpy.molecule import Atom as RMGAtom
import os
import itertools
import logging
import numpy as np
from cclib.io import ccread

import autotst
from autotst.reaction import Reaction, TS
from autotst.species import Species, Conformer
from autotst.geometry import Torsion
from autotst.utils.periodic_table import atomic_number_symbol_dict

from ase import Atom, Atoms
from ase.io.gaussian import read_gaussian, read_gaussian_out
from ase.calculators.gaussian import Gaussian as ASEGaussian

import rmgpy
from rmgpy.molecule import Molecule as RMGMolecule
from rmgpy.reaction import Reaction as RMGReaction
from rmgpy.molecule import Atom as RMGAtom
from rmgpy.molecule import Bond

from shutil import move
import yaml

def read_log(file_path):
        """
        A helper method that allows one to easily parse log files
        """
        symbol_dict = {
            53: "I",
            35: "Br",
            17: "Cl",
            9:  "F",
            8:  "O",
            7:  "N",
            6:  "C",
            1:  "H",
        }
        atoms = []

        parser = ccread(file_path, loglevel=logging.ERROR)

        for atom_num, coords in zip(parser.atomnos, parser.atomcoords[-1]):
            atoms.append(Atom(symbol=symbol_dict[atom_num], position=coords))

        return Atoms(atoms)

def write_input(conformer, ase_calculator):
    """
    A helper method that will write an input file and move it to the correct scratch directory
    """

    ase_calculator.write_input(conformer.ase_molecule)
    try:
        os.makedirs(ase_calculator.scratch)
    except OSError:
        pass

    move(
        ase_calculator.label + ".com",
        os.path.join(
            ase_calculator.scratch,
            ase_calculator.label + ".com"
        ))

    move(
        ase_calculator.label + ".ase",
        os.path.join(
            ase_calculator.scratch,
            ase_calculator.label + ".ase"
        ))


class Gaussian():

    def __init__(self,
                 conformer= None, # Either a transition state or a conformer
                 settings={
                     "method": "m062x",
                     "basis": "cc-pVTZ",
                     "dispersion": None,
                     "mem": "5GB",
                     "sp": 'G4',
                     "convergence": "",
                     "nprocshared": 20,
                     "time": "24:00:00",
                     "partition": 'general,west'
                 },
                 directory=".", #where you want input and log files to be written, default is current directory
                 scratch=None  #where you want temporary files to be written
                 ):

        default_settings = {
            "method": "m062x",
            "basis": "cc-pVTZ",
            "mem": "5GB",
            "nprocshared": 24,
        }

        self.conformer = conformer

        #setting the settings accordingly
        for setting, value in list(default_settings.items()):
            if setting in list(settings.keys()):
                assert isinstance(settings[setting], type(
                    value)), "{} is not a proper instance..."
            else:
                logging.info("{} not specified, setting it to the default value of {}".format(
                    setting, value))
                settings[setting] = value


        self.command = "g16"
        self.settings = settings

        self.settings["convergence"] = settings["convergence"].lower()
        convergence_options = ["", "verytight", "tight", "loose"]
        assert self.settings["convergence"] in convergence_options,"{} is not among the supported convergence options {}".format(settings["convergence"],convergence_options)
    
        self.directory = directory

        if self.settings["dispersion"]:
            dispersion = self.settings["dispersion"].upper()
        else:
            disperion = None

        if settings["dispersion"]:
            dispersion = settings["dispersion"].upper()
            assert dispersion in ['GD3','GD3BJ','GD2'],'Acceptable keywords for dispersion are GD3, GD3BJ, or GD2'
            self.opt_method = settings["method"].upper() + '-' + dispersion + '_' + settings["basis"].upper()
        else:
            self.opt_method = settings["method"].upper() + '_' + settings["basis"].upper()
   
        try: 
            if scratch is None:
                self.scratch = os.environ['GAUSS_SCRDIR']
            else:
                self.scratch = os.environ['GAUSS_SCRDIR'] = scratch
        except:
            if scratch is None:
                self.scratch = '.'
            else:
                self.scratch = os.environ['GAUSS_SCRDIR'] = scratch

    def __repr__(self):
        if isinstance(self.conformer, TS):
            return '<Gaussian Calculator {}>'.format(self.conformer.reaction_label)
        elif isinstance(self.conformer, Conformer):
            return '<Gaussian Calculator {}>'.format(self.conformer.smiles)
        else:
            return '<Gaussian Calculator>'

    def read_nbo_log(self,path):
        """
        Method to determine the representative Lewis structure from a Gaussian NBO calculation.
        Writes a yml file with results from nbo calculation.
        
        Args:
            - path (str): path to gaussian nbo log file
        
        Returns:
            - RMG Molecule (rmgpy.molecule.molecule.Molecule)
        """
        
        data = ccread(path)

        atoms = []
        bonds = {}

        for i, atom_number in enumerate(data.atomnos):
            atoms.append(RMGAtom(id=i+1, element=atomic_number_symbol_dict[atom_number],
                            coords=data.atomcoords[-1][i]))

        lines = open(path, 'r').readlines()
        _ = False

        for line in lines:
            if 'Natural Bond Orbitals (Summary):' in line:
                _ = True
            if _ is True:
                if 'BD' in line:
                    atom1 = atoms[int(line.split('-')[0].split()[-1])-1]
                    atom2 = atoms[int(line.split('-')[1].split()[1])-1]
                    occupancy = float(line.split('-')[1].split()[2])
                    if (atom1.id, atom2.id) not in list(bonds.keys()):
                        bond = Bond(atom1, atom2)
                        bond.order = occupancy/2
                        if '*' in line:
                            bond.order = -bond.order
                        atom1.bonds[atom2] = bond
                        atom2.bonds[atom1] = bond
                        bonds[(atom1.id, atom2.id)] = bond
                    else:
                        bond = bonds[(atom1.id, atom2.id)]
                        if '*' in line:
                            bond.order -= occupancy/2
                        else:
                            bond.order += occupancy/2
                if 'LP'in line:
                    atom = atoms[int(line.split(')')[1].split()[1])-1]
                    orbital = int(line.split(')')[0].split()[-1])
                    occupancy = float(line.split(')')[1].split()[2])
                    orbital = "LP_{}".format(orbital)
                    if orbital not in list(atom.props.keys()):
                        atom.props[orbital] = occupancy
                    else:
                        if '*' in line:
                            atom.props[orbital] -= occupancy
                        else:
                            atom.props[orbital] += occupancy
            if 'Charge unit' in line:
                _ = False

        for atom in atoms:
            if len(atom.props.keys()) > 0:
                for orbital, occupancy in atom.props.items():
                    if atom.lone_pairs == -100:
                        atom.lone_pairs = 0
                    if round(occupancy) == 2:
                        atom.increment_lone_pairs()
                    elif 0 < occupancy <= 1.5:
                        atom.increment_radical()
                atom.update_charge()

        mol = RMGMolecule(atoms=atoms)
        mol.update_lone_pairs()
        mol.update_multiplicity()
        mol.update_connectivity_values()

        assert mol.multiplicity == data.mult,\
            "Multiplicities do not match ({} != {})".format(
                mol.multiplicity, data.mult)

        mol_copy = mol.copy(deep=True)
        for bond in mol_copy.get_all_edges():
            bond.order = round(bond.order)

        info = {
            'info': data.metadata,
            'smiles': mol_copy.to_smiles(),
            'adj_list': mol.to_adjacency_list(),
            'mult': data.mult,
            'inchi': mol_copy.to_inchi(),
            'inchi_key': mol_copy.to_inchi_key(),
            'atom_numbers': data.atomnos.tolist(),
            'coords': data.atomcoords.tolist(),
            'natural_charges': data.atomcharges['natural'].tolist()
        }

        try:
            with open('{}_nbo.yml'.format(mol_copy.to_smiles()), 'w') as f:
                yaml.safe_dump(info, f)
        except:
            pass

        return mol_copy

    def get_rotor_calc(self,
                       torsion_index=0,
                       steps=36,
                       step_size=10.0):
        """
        A method to create all of the calculators needed to perform hindered rotor calculations given a `Conformer` and a `Torsion`.

        Parameters:
        - conformer (Conformer): A `Conformer` object that you want to perform hindered rotor calculations on
        - torsion (Torsion): A `Torsion` object that you want to perform hindered rotor calculations about
        - settings (dict): a dictionary of settings containing method, basis, mem, nprocshared
        - scratch (str): a directory where you want log files to be written to
        - steps (int): the number of steps you want performed in this scan
        - step_size (float): the size, in degrees, of the step you to scan along

        Returns:
        - calc (ASEGaussian): an ASEGaussian calculator with all of the proper setting specified
        """
        torsion = self.conformer.torsions[torsion_index]

        assert (torsion and (isinstance(torsion, Torsion))
                ), "To create a rotor calculator, you must provide a Torsion object."

        assert isinstance(
            self.conformer, Conformer), "A Conformer object was not provided..."

        addsec = ""
        for bond in self.conformer.bonds:
            i, j = bond.atom_indices
            addsec += "B {} {}\n".format(i + 1, j + 1)

        i, j, k, l = torsion.atom_indices
        addsec += "D {} {} {} {} S {} {}\n".format(
            i + 1, j + 1, k + 1, l + 1, steps, float(step_size))

        if isinstance(self.conformer, TS):
            extra = "Opt=(ts,CalcFC,ModRedun)"
            label = conformer_dir = self.conformer.reaction_label
            label += "_{}by{}_{}_{}".format(steps, int(step_size), j, k)
            conformer_type = "ts"
        elif isinstance(self.conformer, Conformer):
            label = conformer_dir = self.conformer.smiles
            label += "_{}by{}_{}_{}".format(steps, int(step_size), j, k)
            conformer_type = "species"
            extra = "Opt=(CalcFC,ModRedun)"

        for locked_torsion in self.conformer.torsions:  # TODO: maybe doesn't work;
            if sorted(locked_torsion.atom_indices) != sorted(torsion.atom_indices):
                a, b, c, d = locked_torsion.atom_indices
                addsec += 'D {0} {1} {2} {3} F\n'.format(a+1, b+1, c+1, d+1)

        self.conformer.rmg_molecule.update_multiplicity()
        mult = self.conformer.rmg_molecule.multiplicity

        new_scratch = os.path.join(
            self.directory,
            conformer_type,
            conformer_dir,
            "rotors"
        )

        ase_gaussian = ASEGaussian(
            mem=self.settings["mem"],
            nprocshared=self.settings["nprocshared"],
            label=label,
            scratch=new_scratch,
            method=self.settings["method"],
            basis=self.settings["basis"],
            extra=extra,
            multiplicity=mult,
            addsec=[addsec[:-1]])

        ase_gaussian.atoms = self.conformer.ase_molecule
        del ase_gaussian.parameters['force']
        return ase_gaussian

    def get_conformer_calc(self):
        """
        A method that creates a calculator for a `Conformer` that will perform a geometry optimization

        Parameters:
        - conformer (Conformer): A `Conformer` object that you want to perform hindered rotor calculations on
        - torsion (Torsion): A `Torsion` object that you want to perform hindered rotor calculations about
        - settings (dict): a dictionary of settings containing method, basis, mem, nprocshared
        - scratch (str): a directory where you want log files to be written to
        - convergence (str): ['verytight','tight','' (default)], specifies the convergence criteria of the geometry optimization

        Returns:
        - calc (ASEGaussian): an ASEGaussian calculator with all of the proper setting specified
        """

        method = self.settings["method"].upper()
        basis = self.settings["basis"].upper()
        if self.settings["dispersion"]:
            dispersion = 'EmpiricalDispersion={}'.format(self.settings["dispersion"].upper())
        else: 
            dispersion = ''

        convergence = self.settings["convergence"].upper()

        self.settings["mem"] = '10GB'
        num_atoms = self.conformer.rmg_molecule.get_num_atoms()
        
        if num_atoms <= 4:
            self.settings["nprocshared"] = 1
            self.settings["time"] = '12:00:00'
        elif num_atoms <= 8:
            self.settings["nprocshared"] = 2
            self.settings["time"] = '12:00:00'
        elif num_atoms <= 15:
            self.settings["nprocshared"] = 4
            self.settings["time"] = '12:00:00'
        elif num_atoms <= 20:
            self.settings["nprocshared"] = 8
            self.settings["time"] = '12:00:00'
            self.settings["mem"] = '20GB'
        else:
            self.settings["nprocshared"] = 12
            self.settings["time"] = '24:00:00'
            self.settings["mem"] = '20GB'

        if isinstance(self.conformer, TS):
            logging.info(
                "TS object provided, cannot obtain a species calculator for a TS")
            return None

        assert isinstance(
            self.conformer, Conformer), "A Conformer object was not provided..."

        #self.conformer.rmg_molecule.update_multiplicity()

        label = "{}_{}_{}_optfreq".format(self.conformer.smiles, self.conformer.index, self.opt_method)

        new_scratch = os.path.join(
            self.directory,
            "species",
            self.opt_method,
            self.conformer.smiles,
            "conformers"
        )

        try:
            os.makedirs(new_scratch)
        except OSError:
            pass

        ase_gaussian = ASEGaussian(
            mem=self.settings["mem"],
            nprocshared=self.settings["nprocshared"],
            label=label,
            scratch=new_scratch,
            method=method,
            basis=basis,
            extra="opt=(calcfc,maxcycles=900,{}) {} freq IOP(7/33=1,2/16=3) scf=(maxcycle=900)".format(convergence,dispersion),
            multiplicity=self.conformer.rmg_molecule.multiplicity)
        ase_gaussian.atoms = self.conformer.ase_molecule
        ase_gaussian.directory = new_scratch
        ase_gaussian.label = label
        ase_gaussian.parameters["partition"] = self.settings["partition"]
        ase_gaussian.parameters["time"] = self.settings["time"]
        del ase_gaussian.parameters['force']
        return ase_gaussian

    def get_nbo_calc(self):
        """
        A method that creates a calculator for a `Conformer` that will perform a Natural Bond Orbital (NBO) population calculation

        Returns:
        - calc (ASEGaussian): an ASEGaussian calculator with all of the proper setting specified
        """

        self.settings["mem"] = '10GB'
        method = self.settings["method"].upper()
        basis = self.settings["basis"].upper()

        num_atoms = self.conformer.rmg_molecule.get_num_atoms()

        if num_atoms <= 4:
            self.settings["nprocshared"] = 1
            self.settings["time"] = '6:00:00'
        elif num_atoms <= 10:
            self.settings["nprocshared"] = 2
            self.settings["time"] = '6:00:00'
        else:
            self.settings["nprocshared"] = 4
            self.settings["time"] = '6:00:00'

        label = "{}_nbo".format(self.conformer.smiles)

        new_scratch = os.path.join(
            self.directory,
            "species",
            self.opt_method,
            self.conformer.smiles,
            "nbo"
        )

        try:
            os.makedirs(new_scratch)
        except OSError:
            pass

        ase_gaussian = ASEGaussian(
            mem=self.settings["mem"],
            nprocshared=self.settings["nprocshared"],
            label=label,
            scratch=new_scratch,
            method=method,
            basis=basis,
            extra="pop=nbo",
            multiplicity=self.conformer.rmg_molecule.multiplicity)
        ase_gaussian.atoms = self.conformer.ase_molecule
        ase_gaussian.directory = new_scratch
        ase_gaussian.label = label
        ase_gaussian.parameters["partition"] = self.settings["partition"]
        ase_gaussian.parameters["time"] = self.settings["time"]
        del ase_gaussian.parameters['force']
        return ase_gaussian

    def get_sp_calc(self):

        method = self.settings["sp"].upper()
        convergence = self.settings["convergence"].upper()

        gaussian_methods = [
            "G1","G2","G3","G4","G2MP2","G3MP2","G3B3","G3MP2B3","G4","G4MP2",
            "W1","W1U","W1BD","W1RO",
            "CBS-4M","CBS-QB3","CBS-APNO",
        ]
        assert method in gaussian_methods

        self.settings["time"] = "24:00:00"
        num_atoms = self.conformer.rmg_molecule.get_num_atoms()
        
        if num_atoms <= 18:
            self.settings["mem"] = '100GB'
            self.settings["nprocshared"] = 16
        else:
            self.settings["mem"] = '300GB'
            self.settings["nprocshared"] = 16
            
        if isinstance(self.conformer, TS):
            logging.info(
                "TS object provided, cannot obtain a species calculator for a TS")
            return None

        assert isinstance(
            self.conformer, Conformer), "A Conformer object was not provided..."

        #self.conformer.rmg_molecule.update_multiplicity()

        label = "{}_{}".format(self.conformer.smiles, method)

        new_scratch = os.path.join(
            self.directory,
            "species",
            self.opt_method,
            self.conformer.smiles,
            "sp"
        )

        try:
            os.makedirs(new_scratch)
        except OSError:
            pass

       
        ase_gaussian = ASEGaussian(
            mem=self.settings["mem"],
            nprocshared=self.settings["nprocshared"],
            label=label,
            scratch=new_scratch,
            method= method,
            basis = '',
            extra="opt=(calcfc,maxcycles=900,{}) IOP(7/33=1,2/16=3) scf=(maxcycle=900)".format(convergence),
            multiplicity=self.conformer.rmg_molecule.multiplicity)
        ase_gaussian.atoms = self.conformer.ase_molecule
        ase_gaussian.directory = new_scratch
        ase_gaussian.label = label
        ase_gaussian.parameters["partition"] = self.settings["partition"]
        ase_gaussian.parameters["time"] = self.settings["time"]
        del ase_gaussian.parameters['force']
        return ase_gaussian

    def get_shell_calc(self):
        """
        A method to create a calculator that optimizes the reaction shell of a `TS` object

        Parameters:
        - ts (TS): A `TS` object that you want to perform calculations on
        - direction (str): the forward or reverse direction of the `TS` object
        - settings (dict): a dictionary of settings containing method, basis, mem, nprocshared
        - directory (str): a directory where you want log files to be written to

        Returns:
        - calc (ASEGaussian): an ASEGaussian calculator with all of the proper setting specified
        """
        assert isinstance(self.conformer, TS), "A TS object was not provided..."
        assert self.conformer.direction.lower() in ["forward", "reverse"]

        self.conformer.rmg_molecule.update_multiplicity()

        label = self.conformer.reaction_label + "_" + self.conformer.direction.lower() + "_shell_" + str(self.conformer.index)

        new_scratch = os.path.join(
            self.directory,
            "ts",
            self.conformer.reaction_label,
            "conformers"
        )

        try:
            os.makedirs(new_scratch)
        except OSError:
            pass
        
        #if self.conformer.reaction_family != "Some reaction family with 4 labeled atoms..."
        if self.conformer.reaction_family.lower() in ["h_abstraction", "intra_h_migration", "r_addition_multiplebond"]:
            ind1 = self.conformer.rmg_molecule.get_labeled_atoms("*1")[0].sorting_label
            ind2 = self.conformer.rmg_molecule.get_labeled_atoms("*2")[0].sorting_label
            ind3 = self.conformer.rmg_molecule.get_labeled_atoms("*3")[0].sorting_label
        else:
            logging.error("Reaction family {} is not supported...".format(self.conformer.reaction_family))
            raise AssertionError

        combos = ""
        combos += "{0} {1} F\n".format(ind1+1, ind2+1)
        combos += "{0} {1} F\n".format(ind2+1, ind3+1)
        combos += "{0} {1} {2} F".format(ind1+1, ind2+1, ind3+1)

        ase_gaussian = ASEGaussian(
            mem=self.settings["mem"],
            nprocshared=self.settings["nprocshared"],
            label=label,
            scratch=new_scratch,
            method=self.settings["method"],
            basis=self.settings["basis"],
            extra="Opt=(ModRedun,Loose,maxcycles=900) Int(Grid=SG1) scf=(maxcycle=900)",
            multiplicity=self.conformer.rmg_molecule.multiplicity,
            addsec=[combos]
        )
        ase_gaussian.atoms = self.conformer.ase_molecule
        del ase_gaussian.parameters['force']

        return ase_gaussian

    def get_center_calc(self):
        """
        A method to create a calculator that optimizes the reaction center of a `TS` object

        Parameters:
        - ts (TS): A `TS` object that you want to perform calculations on
        - direction (str): the forward or reverse direction of the `TS` object
        - settings (dict): a dictionary of settings containing method, basis, mem, nprocshared
        - scratch (str): a directory where you want log files to be written to

        Returns:
        - calc (ASEGaussian): an ASEGaussian calculator with all of the proper setting specified
        """

        assert self.conformer.direction.lower() in ["forward", "reverse"]

        assert isinstance(self.conformer, TS), "A TS object was not provided..."

        indicies = []
        for i, atom in enumerate(self.conformer.rmg_molecule.atoms):
            if not (atom.label != ""):
                indicies.append(i)

        addsec = ""
        for combo in list(itertools.combinations(indicies, 2)):
            a, b = combo
            addsec += "{0} {1} F\n".format(a + 1, b + 1)

        self.conformer.rmg_molecule.update_multiplicity()

        label = self.conformer.reaction_label + "_" + self.conformer.direction.lower() + "_center_" + str(self.conformer.index)

        new_scratch = os.path.join(
            self.directory,
            "ts",
            self.conformer.reaction_label,
            "conformers"
        )

        try:
            os.makedirs(new_scratch)
        except OSError:
            pass

        ase_gaussian = ASEGaussian(
            mem=self.settings["mem"],
            nprocshared=self.settings["nprocshared"],
            label=label,
            scratch=new_scratch,
            method=self.settings["method"],
            basis=self.settings["basis"],
            extra="Opt=(ts,calcfc,noeigentest,ModRedun,maxcycles=900) scf=(maxcycle=900)",
            multiplicity=self.conformer.rmg_molecule.multiplicity,
            addsec=[addsec[:-1]]
        )
        ase_gaussian.atoms = self.conformer.ase_molecule
        del ase_gaussian.parameters['force']

        return ase_gaussian

    def get_overall_calc(self):
        """
        A method to create a calculator that optimizes the overall geometry of a `TS` object

        Parameters:
        - ts (TS): A `TS` object that you want to perform calculations on
        - direction (str): the forward or reverse direction of the `TS` object
        - settings (dict): a dictionary of settings containing method, basis, mem, nprocshared
        - scratch (str): a directory where you want log files to be written to

        Returns:
        - calc (ASEGaussian): an ASEGaussian calculator with all of the proper setting specified
        """

        assert isinstance(self.conformer, TS), "A TS object was not provided..."

        self.conformer.rmg_molecule.update_multiplicity()

        label = self.conformer.reaction_label + "_" + self.conformer.direction.lower() + "_" + str(self.conformer.index)

        new_scratch = os.path.join(
            self.directory,
            "ts",
            self.conformer.reaction_label,
            "conformers"
        )

        try:
            os.makedirs(new_scratch)
        except OSError:
            pass

        ase_gaussian = ASEGaussian(
            mem=self.settings["mem"],
            nprocshared=self.settings["nprocshared"],
            label=label,
            scratch=new_scratch,
            method=self.settings["method"],
            basis=self.settings["basis"],
            extra="opt=(ts,calcfc,noeigentest,maxcycles=900) freq scf=(maxcycle=900) IOP(7/33=1,2/16=3)",
            multiplicity=self.conformer.rmg_molecule.multiplicity)
        ase_gaussian.atoms = self.conformer.ase_molecule
        del ase_gaussian.parameters['force']

        return ase_gaussian

    def get_irc_calc(self):
        """
        A method to create a calculator that runs an IRC calculation the overall geometry of a `TS` object

        Parameters:
        - ts (TS): A `TS` object that you want to perform calculations on
        - direction (str): the forward or reverse direction of the `TS` object
        - settings (dict): a dictionary of settings containing method, basis, mem, nprocshared
        - scratch (str): a directory where you want log files to be written to

        Returns:
        - calc (ASEGaussian): an ASEGaussian calculator with all of the proper setting specified
        """

        assert isinstance(self.conformer, TS), "A TS object was not provided..."

        self.conformer.rmg_molecule.update_multiplicity()
        label = self.conformer.reaction_label + "_irc_" + self.conformer.direction + "_" + str(self.conformer.index)

        new_scratch = os.path.join(
            self.directory,
            "ts",
            self.conformer.reaction_label,
            "irc"
        )
        try:
            os.makedirs(new_scratch)
        except OSError:
            pass

        ase_gaussian = ASEGaussian(
            mem=self.settings["mem"],
            nprocshared=self.settings["nprocshared"],
            label=label,
            scratch=new_scratch,
            method=self.settings["method"],
            basis=self.settings["basis"],
            extra="irc=(calcall)",
            multiplicity=self.conformer.rmg_molecule.multiplicity
        )
        ase_gaussian.atoms = self.conformer.ase_molecule
        del ase_gaussian.parameters['force']

        return ase_gaussian

    def verify_output_file(self, path):
        """
        A method to verify output files and make sure that they successfully converged, if not, re-running them

        Returns a tuple where the first entry indicates if the file is complete, the second indicates if it was successful
        """

        if not os.path.exists(path):
            logging.info("Not a valid path, cannot be verified...")
            return (False, False)

        f = open(path, "r")
        file_lines = f.readlines()[-10:]
        verified = (False, False)
        for file_line in file_lines:
            if " Normal termination" in file_line:
                verified = (True, True)
            if " Error termination" in file_line:
                verified = (True, False)

        return verified

    def validate_irc(self):
        """
        A method to verify an IRC calc
        """

        logging.info("Validating IRC file...")
        irc_path = os.path.join(
            self.directory,
            "ts",
            self.conformer.reaction_label,
            "irc",
            self.conformer.reaction_label + "_irc_" + self.conformer.direction + "_" + str(self.conformer.index) + ".log"
        )

        label = self.conformer.reaction_label + "_" + self.conformer.direction + "_" + str(self.conformer.index)

        complete, converged = self.verify_output_file(irc_path)
        if not complete:
            logging.info(
                "It seems that the IRC claculation did not complete")
            return False
        if not converged:
            logging.info("The IRC calculation did not converge...")
            return False

        pth1 = list()
        steps = list()
        with open(irc_path) as output_file:
            for line in output_file:
                line = line.strip()

                if line.startswith('Point Number:'):
                    if int(line.split()[2]) > 0:
                        if int(line.split()[-1]) == 1:
                            pt_num = int(line.split()[2])
                            pth1.append(pt_num)
                        else:
                            pass
                elif line.startswith('# OF STEPS ='):
                    num_step = int(line.split()[-1])
                    steps.append(num_step)
        # This indexes the coordinate to be used from the parsing
        if steps == []:
            logging.error('No steps taken in the IRC calculation!')
            return False
        else:
            pth1End = sum(steps[:pth1[-1]])
            # Compare the reactants and products
            irc_parse = ccread(irc_path, loglevel=logging.ERROR)


            atomcoords = irc_parse.atomcoords
            atomnos = irc_parse.atomnos

            mol1 = RMGMolecule()
            mol1.from_xyz(atomnos, atomcoords[pth1End])
            mol2 = RMGMolecule()
            mol2.from_xyz(atomnos, atomcoords[-1])

            test_reaction = RMGReaction(
                reactants=mol1.split(),
                products=mol2.split(),
            )

            r, p = self.conformer.reaction_label.split("_")

            reactants = []
            products = []

            for react in r.split("+"):
                react = RMGMolecule(SMILES=react)
                reactants.append(react)

            for prod in p.split("+"):
                prod = RMGMolecule(SMILES=prod)
                products.append(prod)

            possible_reactants = []
            possible_products = []
            for reactant in reactants:
                possible_reactants.append(
                    reactant.generate_resonance_structures())

            for product in products:
                possible_products.append(
                    product.generate_resonance_structures())

            possible_reactants = list(itertools.product(*possible_reactants))
            possible_products = list(itertools.product(*possible_products))

            for possible_reactant in possible_reactants:
                reactant_list = []
                for react in possible_reactant:
                    reactant_list.append(react.to_single_bonds())

                for possible_product in possible_products:
                    product_list = []
                    for prod in possible_product:
                        product_list.append(prod.to_single_bonds())

                    target_reaction = RMGReaction(
                        reactants=list(reactant_list),
                        products=list(product_list)
                    )

                    if target_reaction.is_isomorphic(test_reaction):
                        logging.info("IRC calculation was successful!")
                        return True
            logging.info("IRC calculation failed for {} :(".format(irc_path))
            return False
