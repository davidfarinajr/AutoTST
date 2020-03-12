#!/usr/bin/python
# -*- coding: utf-8 -*-

##########################################################################
#
#   AutoTST - Automated Transition State Theory
#
#   Copyright (c) 2015-2020 Richard H. West (r.west@northeastern.edu)
#   and the AutoTST Team
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
import ase
import cclib.io

from ..reaction import Reaction, TS
from ..species import Species, Conformer

import rmgpy
import arkane.main

FORMAT = "%(filename)s:%(lineno)d %(funcName)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class StatMech():

    def __init__(
            self,
            reaction,
            directory=".",
            model_chemistry="M06-2X/cc-pVTZ",
            freq_scale_factor=0.982):
        """
        A class to perform Arkane calculations:
        :param: reaction: (Reaction) The reaction of interest
        :param: output_directory: (str) The directory where you would like output files written to
        :param: model_chemistry: (str) The supported model_chemistry described by http://reactionmechanismgenerator.github.io/RMG-Py/users/arkane/input.html#model-chemistry
        :param: freq_scale_factor: (float) The scaling factor corresponding to the model chemistry - source:https://comp.chem.umn.edu/freqscale/version3b1.htm
        """

        self.reaction = reaction
        self.directory = directory

        self.kinetics_job = arkane.main.Arkane()
        self.thermo_job = arkane.main.Arkane()
        self.model_chemistry = model_chemistry
        self.freq_scale_factor = freq_scale_factor

    def get_atoms(self, conformer):
        """
        A method to create an atom dictionary for an rmg molecule from a `Conformer` object.

        Parameters:
        - conformer (Conformer): a conformer object that you want atom info from

        Returns:
        - atom_dict (dict): a dictionary containing counts of different atom types
        """
        atom_dict = {}
        for atom in conformer.rmg_molecule.atoms:
            if atom.is_carbon():
                atom_type = "C"
            if atom.is_hydrogen():
                atom_type = "H"
            if atom.is_oxygen():
                atom_type = "O"

            try:
                atom_dict[atom_type] += 1
            except KeyError:
                atom_dict[atom_type] = 1

        return atom_dict

    def get_bonds(self, conformer):
        """
        A method to create a bond dictionary for an rmg molecule from a `Conformer` object.

        Parameters:
        - conformer (Conformer): a conformer object that you want bond info from

        Returns:
        - bond_dict (dict): a dictionary containing counts of different bond types
        """

        bonds = conformer.rmg_molecule.get_all_edges()
        bond_dict = {}
        for bond in bonds:
            if bond.is_single():
                if bond.atom1.symbol == 'C' and bond.atom2.symbol == 'C':
                    bond_type = 'C-C'
                elif (bond.atom1.symbol == 'H' and bond.atom2.symbol == 'H'):
                    bond_type = 'H-H'
                elif (bond.atom1.symbol == 'C' and bond.atom2.symbol == 'H') or (bond.atom1.symbol == 'H' and bond.atom2.symbol == 'C'):
                    bond_type = 'C-H'
                elif (bond.atom1.symbol == 'O' and bond.atom2.symbol == 'O'):
                    bond_type = 'O-O'
                elif (bond.atom1.symbol == 'C' and bond.atom2.symbol == 'O') or (bond.atom1.symbol == 'O' and bond.atom2.symbol == 'C'):
                    bond_type = 'C-O'
                elif (bond.atom1.symbol == 'H' and bond.atom2.symbol == 'O') or (bond.atom1.symbol == 'O' and bond.atom2.symbol == 'H'):
                    bond_type = 'O-H'
                elif bond.atom1.symbol == 'N' and bond.atom2.symbol == 'N':
                    bond_type = 'N-N'
                elif (bond.atom1.symbol == 'C' and bond.atom2.symbol == 'N') or (bond.atom1.symbol == 'N' and bond.atom2.symbol == 'C'):
                    bond_type = 'N-C'
                elif (bond.atom1.symbol == 'O' and bond.atom2.symbol == 'N') or (bond.atom1.symbol == 'N' and bond.atom2.symbol == 'O'):
                    bond_type = 'N-O'
                elif (bond.atom1.symbol == 'H' and bond.atom2.symbol == 'N') or (bond.atom1.symbol == 'N' and bond.atom2.symbol == 'H'):
                    bond_type = 'N-H'
                elif bond.atom1.symbol == 'S' and bond.atom2.symbol == 'S':
                    bond_type = 'S-S'
                elif (bond.atom1.symbol == 'H' and bond.atom2.symbol == 'S') or (bond.atom1.symbol == 'S' and bond.atom2.symbol == 'H'):
                    bond_type = 'S-H'
            elif bond.is_double():
                if bond.atom1.symbol == 'C' and bond.atom2.symbol == 'C':
                    bond_type = 'C=C'
                elif (bond.atom1.symbol == 'O' and bond.atom2.symbol == 'O'):
                    bond_type = 'O=O'
                elif (bond.atom1.symbol == 'C' and bond.atom2.symbol == 'O') or (bond.atom1.symbol == 'O' and bond.atom2.symbol == 'C'):
                    bond_type = 'C=O'
                elif bond.atom1.symbol == 'N' and bond.atom2.symbol == 'N':
                    bond_type = 'N=N'
                elif (bond.atom1.symbol == 'C' and bond.atom2.symbol == 'N') or (bond.atom1.symbol == 'N' and bond.atom2.symbol == 'C'):
                    bond_type = 'N=C'
                elif (bond.atom1.symbol == 'O' and bond.atom2.symbol == 'N') or (bond.atom1.symbol == 'N' and bond.atom2.symbol == 'O'):
                    bond_type = 'N=O'
                elif (bond.atom1.symbol == 'O' and bond.atom2.symbol == 'S') or (bond.atom1.symbol == 'S' and bond.atom2.symbol == 'O'):
                    bond_type = 'S=O'
            elif bond.is_triple():
                if bond.atom1.symbol == 'C' and bond.atom2.symbol == 'C':
                    bond_type = 'C#C'
                elif bond.atom1.symbol == 'N' and bond.atom2.symbol == 'N':
                    bond_type = 'N#N'
                elif (bond.atom1.symbol == 'C' and bond.atom2.symbol == 'N') or (bond.atom1.symbol == 'N' and bond.atom2.symbol == 'C'):
                    bond_type = 'N#C'
            try:
                bond_dict[bond_type] += 1
            except KeyError:
                bond_dict[bond_type] = 1

        return bond_dict

    def write_species_files(self, species):
        """
        A method to write Arkane files for all conformers in a Species object

        Parameters:
        - species (Species): a species object that you want to write arkane files for
        - scratch (str): the directory where you want to write arkane files to, there should be a 'species/SMILES/' subdirectory

        Returns:
        - None
        """

        for smiles, confs in list(species.conformers.items()):
            if os.path.exists(os.path.join(self.directory, "species", smiles, smiles + ".log")):
                logging.info(
                    f"Lowest energy conformer log file exists for {smiles}")
                self.write_conformer_file(conformer=confs[0])
            else:
                logging.info(
                    f"Lowest energy conformer log file DOES NOT exist for {smiles}")

    def write_conformer_file(self, conformer):
        """
        A method to write Arkane files for a single Conformer object

        Parameters:
        - conformer (Conformer): a Conformer object that you want to write an Arkane file for
        - scratch (str): the directory where you want to write arkane files to, there should be a 'species/SMILES/' subdirectory

        Returns:
        - None
        """
        label = conformer.smiles

        if not os.path.exists(os.path.join(self.directory, "species", label, label + ".log")):
            logging.info("There is no lowest energy conformer file...")
            return False

        if os.path.exists(os.path.join(self.directory, "species", label, label + '.py')):
            logging.info("Species input file already written... Not doing anything")
            return True

        parser = cclib.io.ccread(os.path.join(
            self.directory, "species", label, label + ".log"), loglevel=logging.ERROR)
        symbol_dict = {
            35: "Br",
            17: "Cl",
            9:  "F",
            8:  "O",
            7:  "N",
            6:  "C",
            1:  "H",
        }

        atoms = []

        for atom_num, coords in zip(parser.atomnos, parser.atomcoords[-1]):
            atoms.append(ase.Atom(symbol=symbol_dict[atom_num], position=coords))

        conformer._ase_molecule = ase.Atoms(atoms)
        conformer.update_coords_from("ase")
        mol = conformer.rmg_molecule
        output = ['#!/usr/bin/env python',
                  '# -*- coding: utf-8 -*-', '', 'atoms = {']

        atom_dict = self.get_atoms(conformer=conformer)  # Fix this

        for atom, count in atom_dict.items():
            output.append(f"    '{atom}': {count},")
        output = output + ['}', '']

        bond_dict = self.get_bonds(conformer=conformer) 
        if bond_dict != {}:
            output.append('bonds = {')
            for bond_type, num in bond_dict.items():
                output.append(f"    '{bond_type}': {num},")
            output.append("}")
        else:
            output.append('bonds = {}')

        external_symmetry = conformer.calculate_symmetry_number()

        output += ["",
                   f"linear = {conformer.rmg_molecule.is_linear()}",
                   "",
                   f"externalSymmetry = {external_symmetry}",
                   "",
                   f"spinMultiplicity = {conformer.rmg_molecule.multiplicity}",
                   "",
                   "opticalIsomers = 1",
                   ""]

        output += ["energy = {", f"    '{self.model_chemistry}': Log('{label}.log'),", "}", ""]  # fix this

        output += [f"geometry = Log('{label}.log')", ""]

        output += [
            f"frequencies = Log('{label}.log')", ""]

        """
        TODO: add rotor information @carl
        output += ["rotors = ["]
        for torsion in conf.torsions:
            output += [self.get_rotor_info(conf, torsion)]
        output += ["]"]
        """
        input_string = ""

        for t in output:
            input_string += t + "\n"

        with open(os.path.join(self.directory, "species", label, label + '.py'), "w") as f:
            f.write(input_string)
        return True

    def get_rotor_info(self, conformer, torsion_index):
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
        torsion = conformer.torsions[torsion_index]
        _, j, k, _ = torsion.atom_indices

        # Adjusted since mol's IDs start from 0 while Arkane's start from 1
        tor_center_adj = [j+1, k+1]

        if isinstance(conformer, TS):
            tor_log = os.path.join(
                scratch,
                "ts",
                conformer.reaction_label,
                "torsions",
                comformer.reaction_label + f"_36by10_{j}_{k}.log"
            )
        else:
            tor_log = os.path.join(
                scratch,
                "species",
                conformer.smiles,
                "torsions",
                conformer.smiles + f'_36by10_{j}_{k}.log'
            )

        if not os.path.exists(tor_log):
            logging.info(
                f"Torsion log file does not exist for {torsion}")
            return ""

        top_IDs = []
        for num, tf in enumerate(torsion.mask):
            if tf:
                top_IDs.append(num)

        # Adjusted to start from 1 instead of 0
        top_IDs_adj = [ID+1 for ID in top_IDs]

        info = f"     HinderedRotor(scanLog=Log('{tor_log}'), pivots={tor_center_adj}, top={top_IDs_adj}, fit='fourier'),"

        return info

    def write_ts_input(self, transitionstate):
        """
        A method to write Arkane files for a single TS object

        Parameters:
        - transitionstate (TS): a TS object that you want to write an Arkane file for
        - scratch (str): the directory where you want to write arkane files to, there should be a 'ts/REACTION_LABEL/' subdirectory

        Returns:
        - None
        """

        label = transitionstate.reaction_label

        if os.path.exists(os.path.join(self.directory, "ts", label, label + '.py')):
            logging.info("TS input file already written... Not doing anything")
            return True

        if not os.path.exists(os.path.join(self.directory, "ts", label, label + ".log")):
            logging.info("There is no lowest energy conformer file...")
            return False

        parser = cclib.io.ccread(os.path.join(self.directory, "ts", label, label + ".log"), loglevel=logging.ERROR)
        symbol_dict = {
            17: "Cl",
            9:  "F",
            8:  "O",
            7:  "N",
            6:  "C",
            1:  "H",
        }

        atoms = []
        for atom_num, coords in zip(parser.atomnos, parser.atomcoords[-1]):
            atoms.append(ase.Atom(symbol=symbol_dict[atom_num], position=coords))

        transitionstate._ase_molecule = ase.Atoms(atoms)
        transitionstate.update_coords_from("ase")

        output = ['#!/usr/bin/env python',
                  '# -*- coding: utf-8 -*-', '', 'atoms = {']

        atom_dict = self.get_atoms(conformer=transitionstate)  # need to fix

        for atom, count in atom_dict.items():
            output.append(f"    '{atom}': {count},")
        output = output + ['}', '']

        bond_dict = self.get_bonds(conformer=transitionstate)  # need to fix
        if bond_dict != {}:
            output.append('bonds = {')
            for bond_type, num in bond_dict.items():
                output.append(f"    '{bond_type}': {num},")

            output.append("}")
        else:
            output.append('bonds = {}')
        transitionstate.rmg_molecule.update_multiplicity()

        external_symmetry = transitionstate.calculate_symmetry_number()

        output += ["",
                   "linear = False",
                   "",
                   f"externalSymmetry = {external_symmetry}",
                   "",
                   f"spinMultiplicity = {transitionstate.rmg_molecule.multiplicity}",
                   "",
                   "opticalIsomers = 1",
                   ""]

        output += ["energy = {", f"    '{self.model_chemistry}': Log('{label}.log'),", "}", ""]  # fix this

        output += [f"geometry = Log('{label}.log')", ""]

        output += [
            f"frequencies = Log('{label}.log')", ""]

        output += ["rotors = []", ""]  # TODO: Fix this

        input_string = ""

        for t in output:
            input_string += t + "\n"

        with open(os.path.join(self.directory, "ts", label, label + '.py'), "w") as f:
            f.write(input_string)
        return True

    def write_kinetics_input(self):
        """
        A method to write Arkane file to obtain kinetics for a Reaction object

        Parameters:
        - reaction (Reaction): a Reaction object that you want to write an Arkane kinetics job file for.
        - scratch (str): the directory where you want to write arkane files to, there should be a 'species/SMILES/' subdirectory

        Returns:
        - None
        """

        top = [
            "#!/usr/bin/env python",
            "# -*- coding: utf-8 -*-",
            "",
            f'modelChemistry = "{self.model_chemistry}"',  # fix this
            f"frequencyScaleFactor = {self.freq_scale_factor}",  # fix this
            "useHinderedRotors = False",  # fix this @carl
            "useBondCorrections = False",
            ""]

        labels = []
        r_smiles = []
        p_smiles = []
        for i, react in enumerate(self.reaction.reactants):
            lowest_energy = 1e5
            lowest_energy_conf = None

            if len(list(react.conformers.keys())) > 1:
                for smiles in list(react.conformers.keys()):
                    path = os.path.join(self.directory, "species",
                                        smiles, smiles + ".log")
                    if not os.path.exists(path):
                        logging.info(
                            f"It looks like {smiles} doesn't have any optimized geometries")
                        continue

                    parser = cclib.io.ccread(path, loglevel=logging.ERROR)
                    energy = parser.scfenergies[-1]
                    if energy < lowest_energy:
                        lowest_energy = energy
                        lowest_energy_conf = react.conformers[smiles][0]

            else:
                smiles = list(react.conformers.keys())[0]
                path = os.path.join(self.directory, "species",
                                    smiles, smiles + ".log")
                if not os.path.exists(path):
                    logging.info(
                        f"It looks like {smiles} doesn't have any optimized geometries")
                    continue

                parser = cclib.io.ccread(path, loglevel=logging.ERROR)
                lowest_energy = parser.scfenergies[-1]
                lowest_energy_conf = list(react.conformers.values())[0][0]

            # r_smiles.append(lowest_energy_conf.smiles)
            r_smiles.append(f"react_{i}")
            label = lowest_energy_conf.smiles
            if label in labels:
                continue
            else:
                labels.append(label)
            p = os.path.join(self.directory, "species", label, label + ".py")
            line = f"species('react_{i}', '{p}', structure=SMILES('{label}'))"
            top.append(line)

        for i, prod in enumerate(self.reaction.products):

            lowest_energy = 1e5
            lowest_energy_conf = None

            if len(list(prod.conformers.keys())) > 1:
                for smiles in list(prod.conformers.keys()):
                    path = os.path.join(self.directory, "species",
                                        smiles, smiles + ".log")
                    if not os.path.exists(path):
                        logging.info(
                            f"It looks like {smiles} doesn't have any optimized geometries")
                        continue

                    parser = cclib.io.ccread(path, loglevel=logging.ERROR)
                    energy = parser.scfenergies[-1]
                    if energy < lowest_energy:
                        lowest_energy = energy
                        lowest_energy_conf = prod.conformers[smiles][0]

            else:
                smiles = list(prod.conformers.keys())[0]
                path = os.path.join(self.directory, "species",
                                    smiles, smiles + ".log")
                if not os.path.exists(path):
                    logging.info(
                        f"It looks like {smiles} doesn't have any optimized geometries")
                    continue

                parser = cclib.io.ccread(path, loglevel=logging.ERROR)
                lowest_energy = parser.scfenergies[-1]
                lowest_energy_conf = list(prod.conformers.values())[0][0]

            # p_smiles.append(lowest_energy_conf.smiles)
            p_smiles.append(f"prod_{i}")
            label = lowest_energy_conf.smiles
            if label in labels:
                continue
            else:
                labels.append(label)
            p = os.path.join(self.directory, "species", label, label + ".py")
            line = f"species('prod_{i}', '{p}', structure=SMILES('{label}'))"
            top.append(line)
        p = os.path.join(self.directory, "ts", self.reaction.label, self.reaction.label + ".py")
        line = f"transitionState('TS', '{p}')"
        top.append(line)

        line = ["",
                "reaction(",
                f"    label = '{self.reaction.label}',",
                f"    reactants = {r_smiles},",
                f"    products = {p_smiles},",
                "    transitionState = 'TS',",
                "    tunneling = 'Eckart',",
                ")",
                "",
                "statmech('TS')",
                f"kinetics('{self.reaction.label}')"]

        top += line

        input_string = ""

        for t in top:
            input_string += t + "\n"

        with open(os.path.join(self.directory, "ts", self.reaction.label, self.reaction.label + ".kinetics.py"), "w") as f:
            f.write(input_string)

    def write_thermo_input(self, conformer):
        """
        A method to write Arkane file to obtain thermochemistry for a Conformer object

        Parameters:
        - conformer (Conformer): a Conformer object that you want to write an Arkane thermo job file for.
        - scratch (str): the directory where you want to write arkane files to, there should be a 'species/SMILES/' subdirectory

        Returns:
        - None
        """

        model_chemistry = self.model_chemistry
        freq_scale_factor = self.freq_scale_factor

        top = [
            "#!/usr/bin/env python",
            "# -*- coding: utf-8 -*-",
            "",
            f'modelChemistry = "{model_chemistry}"',  # fix this
            f"frequencyScaleFactor = {freq_scale_factor}",  # fix this
            "useHinderedRotors = False",  # fix this @carl
            "useBondCorrections = False",
            ""]
        p = os.path.join(conformer.smiles + ".py")
        line = f"species('species', '{f}', structure=SMILES('{conformer.smiles}'))"
        top.append(line)

        top.append("statmech('species')")
        top.append("thermo('species', 'NASA')")

        input_string = ""

        for t in top:
            input_string += t + "\n"

        with open(os.path.join(self.directory, "species", conformer.smiles, conformer.smiles + ".thermo.py"), "w") as f:
            f.write(input_string)

    def write_files(self):
        """
        A method to write all species, transition state, and kinetics job files to obtain kinetic parameters

        Parameters:
        - None

        Returns:
        - None
        """

        for mol in self.reaction.reactants:
            for smiles, confs in list(mol.conformers.items()):
                conf = confs[0]
                self.write_conformer_file(conf)

        for mol in self.reaction.products:
            for smiles, confs in list(mol.conformers.items()):
                conf = confs[0]
                self.write_conformer_file(conf)

        self.write_ts_input(
            self.reaction.ts["forward"][0])

        self.write_kinetics_input()

    def run(self):
        """
        A method to write run a kinetics job from all of the files written `write_files`

        Parameters:
        - None

        Returns:
        - None
        """

        self.kinetics_job.input_file = os.path.join(
            self.directory, "ts", self.reaction.label, self.reaction.label + ".kinetics.py")
        self.kinetics_job.plot = False
        self.kinetics_job.output_directory = os.path.join(self.directory, "ts", self.reaction.label)

        self.kinetics_job.execute()

        for job in self.kinetics_job.job_list:
            if isinstance(job, arkane.main.KineticsJob):
                self.kinetics_job = job
            elif isinstance(job, arkane.main.ThermoJob):
                self.thermo_job = job

    def set_results(self):
        """
        A method to set the RMGReaction from the kinetics job to the RMGReaction of the input Reaction

        Parameters:
        - None

        Returns:
        - None
        """

        for reactant in self.reaction.rmg_reaction.reactants:
            for r in self.kinetics_job.reaction.reactants:
                if reactant.to_smiles() == r.label:
                    r.molecule = [reactant]

        for product in self.reaction.rmg_reaction.products:
            for p in self.kinetics_job.reaction.products:
                if product.to_smiles() == p.label:
                    p.molecule = [product]

        self.reaction.rmg_reaction = self.kinetics_job.reaction

        return self.reaction
