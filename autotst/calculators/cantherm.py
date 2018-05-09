from rmgpy.cantherm import CanTherm, KineticsJob, StatMechJob
from rmgpy.cantherm import *
from autotst.reaction import *

class AutoTST_CanTherm():

    def __init__(self, reaction, output_directory="."):

        self.reaction = reaction

        self.cantherm_job = CanTherm()
        self.output_directory=output_directory
        self.cantherm_job.outputDirectory = self.output_directory

    def get_atoms(self, mol):
        atom_dict={}
        if isinstance(mol, AutoTST_Molecule):
            rmg_mol = mol.rmg_molecule
        elif isinstance(mol, AutoTST_Reaction):
            rmg_mol = mol.ts.rmg_ts
        elif isinstance(mol, AutoTST_TS):
            rmg_mol = mol.rmg_ts
        for atom in rmg_mol.atoms:
            if atom.isCarbon():
                atom_type = "C"
            if atom.isHydrogen():
                atom_type = "H"
            if atom.isOxygen():
                atom_type = "O"

            try:
                atom_dict[atom_type] += 1
            except KeyError:
                atom_dict[atom_type] = 1

        return atom_dict

    def get_bonds(self, mol):
        bondList = []
        if isinstance(mol, AutoTST_Molecule):
            rmg_mol = mol.rmg_molecule
        elif isinstance(mol, AutoTST_Reaction):
            rmg_mol = mol.ts.rmg_ts
        elif isinstance(mol, AutoTST_TS):
            rmg_mol = mol.rmg_ts
        for atom in rmg_mol.atoms:
            for bond in atom.bonds.values():
                bondList.append(bond)
        bonds = list(set(bondList))
        bondDict = {}
        for bond in bonds:
            if bond.isSingle():
                if bond.atom1.symbol=='C' and bond.atom2.symbol=='C':
                    bondType = 'C-C'
                elif (bond.atom1.symbol=='H' and bond.atom2.symbol=='H'):
                    bondType = 'H-H'
                elif (bond.atom1.symbol=='C' and bond.atom2.symbol=='H') or (bond.atom1.symbol=='H' and bond.atom2.symbol=='C'):
                    bondType = 'C-H'
                elif (bond.atom1.symbol=='O' and bond.atom2.symbol=='O'):
                    bondType = 'O-O'
                elif (bond.atom1.symbol=='C' and bond.atom2.symbol=='O') or (bond.atom1.symbol=='O' and bond.atom2.symbol=='C'):
                    bondType = 'C-O'
                elif (bond.atom1.symbol=='H' and bond.atom2.symbol=='O') or (bond.atom1.symbol=='O' and bond.atom2.symbol=='H'):
                    bondType = 'O-H'
                elif bond.atom1.symbol=='N' and bond.atom2.symbol=='N':
                    bondType = 'N-N'
                elif (bond.atom1.symbol=='C' and bond.atom2.symbol=='N') or (bond.atom1.symbol=='N' and bond.atom2.symbol=='C'):
                    bondType = 'N-C'
                elif (bond.atom1.symbol=='O' and bond.atom2.symbol=='N') or (bond.atom1.symbol=='N' and bond.atom2.symbol=='O'):
                    bondType = 'N-O'
                elif (bond.atom1.symbol=='H' and bond.atom2.symbol=='N') or (bond.atom1.symbol=='N' and bond.atom2.symbol=='H'):
                    bondType = 'N-H'
                elif bond.atom1.symbol=='S' and bond.atom2.symbol=='S':
                    bondType = 'S-S'
                elif (bond.atom1.symbol=='H' and bond.atom2.symbol=='S') or (bond.atom1.symbol=='S' and bond.atom2.symbol=='H'):
                    bondType = 'S-H'
            elif bond.isDouble:
                if bond.atom1.symbol=='C' and bond.atom2.symbol=='C':
                    bondType = 'C=C'
                elif (bond.atom1.symbol=='O' and bond.atom2.symbol=='O'):
                    bondType = 'O=O'
                elif (bond.atom1.symbol=='C' and bond.atom2.symbol=='O') or (bond.atom1.symbol=='O' and bond.atom2.symbol=='C'):
                    bondType = 'C=O'
                elif bond.atom1.symbol=='N' and bond.atom2.symbol=='N':
                    bondType = 'N=N'
                elif (bond.atom1.symbol=='C' and bond.atom2.symbol=='N') or (bond.atom1.symbol=='N' and bond.atom2.symbol=='C'):
                    bondType = 'N=C'
                elif (bond.atom1.symbol=='O' and bond.atom2.symbol=='N') or (bond.atom1.symbol=='N' and bond.atom2.symbol=='O'):
                    bondType = 'N=O'
                elif (bond.atom1.symbol=='O' and bond.atom2.symbol=='S') or (bond.atom1.symbol=='S' and bond.atom2.symbol=='O'):
                    bondType = 'S=O'
            elif bond.isTriple:
                if bond.atom1.symbol=='C' and bond.atom2.symbol=='C':
                    bondType = 'C#C'
                elif bond.atom1.symbol=='N' and bond.atom2.symbol=='N':
                    bondType = 'N#N'
                elif (bond.atom1.symbol=='C' and bond.atom2.symbol=='N') or (bond.atom1.symbol=='N' and bond.atom2.symbol=='C'):
                    bondType = 'N#C'
            try:
                bondDict[bondType] += 1
            except KeyError:
                bondDict[bondType] = 1

        return bondDict

    def write_cantherm_for_reacts_and_prods(self, mol):

        output = ['#!/usr/bin/env python', '# -*- coding: utf-8 -*-', '', 'atoms = {']

        atom_dict = self.get_atoms(mol)

        for atom, count in atom_dict.iteritems():
            output.append("    '{0}': {1},".format(atom, count))
        output = output + ['}', '']

        bond_dict = self.get_bonds(mol)
        if bond_dict != {}:
            output.append('bonds = {')
            for bond_type, num in bond_dict.iteritems():
                output.append("    '{0}': {1},".format(bond_type, num))
            output.append("}")
        else:
            output.append('bonds = {}')



        output += ["","linear = False","","externalSymmetry = 1", "", "spinMultiplicity = {}".format(mol.rmg_molecule.multiplicity), "", "opticalIsomers = 1", ""]

        output += ["energy = {","    'M06-2X/cc-pVTZ': GaussianLog('{}.log'),".format(mol.smiles),"}",""]

        output += ["geometry = GaussianLog('{}.log')".format(mol.smiles), ""]

        output += ["frequencies = GaussianLog('{}.log')".format(mol.smiles), ""]

        output += ["rotors = []"]

        input_string = ""

        for t in output:
            input_string += t +"\n"

        with open(os.path.join(mol.smiles+".py"), "w") as f:
            f.write(input_string)

    def write_statmech_ts(self, rxn):
        output = ['#!/usr/bin/env python', '# -*- coding: utf-8 -*-', '', 'atoms = {']


        atom_dict = self.get_atoms(rxn)

        for atom, count in atom_dict.iteritems():
            output.append("    '{0}': {1},".format(atom, count))
        output = output + ['}', '']


        bond_dict = self.get_bonds(rxn)
        if bond_dict != {}:
            output.append('bonds = {')
            for bond_type, num in bond_dict.iteritems():
                output.append("    '{0}': {1},".format(bond_type, num))

            output.append("}")
        else:
            output.append('bonds = {}')



        output += ["","linear = False","","externalSymmetry = 1", "", "spinMultiplicity = {}".format(rxn.ts.rmg_ts.multiplicity), "", "opticalIsomers = 1", ""]

        output += ["energy = {","    'M06-2X/cc-pVTZ': GaussianLog('{}_overall.log'),".format(rxn.label),"}",""]

        output += ["geometry = GaussianLog('{}_overall.log')".format(rxn.label), ""]

        output += ["frequencies = GaussianLog('{}_overall.log')".format(rxn.label), ""]

        output += ["rotors = []", ""]

        input_string = ""

        for t in output:
            input_string += t +"\n"

        with open(os.path.join(rxn.label +".py"), "w") as f:
            f.write(input_string)

    def write_cantherm_ts(self, rxn):
        top = ["#!/usr/bin/env python", "# -*- coding: utf-8 -*-", "", 'modelChemistry = "M06-2X/cc-pVTZ"', "frequencyScaleFactor = 0.982", "useHinderedRotors = False", "useBondCorrections = False", ""]

        scratch ="."
        for react in rxn.reactant_mols:
            line = "species('{0}', '{1}')".format(react.smiles, os.path.join(scratch, react.smiles +".py"))
            top.append(line)

        for prod in rxn.product_mols:
            line = "species('{0}', '{1}')".format(prod.smiles, os.path.join(scratch, prod.smiles +".py"))
            top.append(line)

        line = "transitionState('TS', '{0}')".format(os.path.join(scratch, rxn.label +".py"))
        top.append(line)


        line = ["",
        "reaction(",
        "    label = '{0}',".format(rxn.label),
        "    reactants = ['{0}', '{1}'],".format(rxn.reactant_mols[0].smiles, rxn.reactant_mols[1].smiles),
        "    products = ['{0}', '{1}'],".format(rxn.product_mols[0].smiles, rxn.product_mols[1].smiles),
        "    transitionState = 'TS',",
        "    tunneling = 'Eckart',",
        ")",
        "",
        "statmech('TS')",
        "kinetics('{0}')".format(rxn.label)]
        top += line


        input_string = ""

        for t in top:
            input_string += t +"\n"

        with open(rxn.label +".cantherm.py", "w") as f:
            f.write(input_string)

    def write_files(self):
        for mol in self.reaction.reactant_mols:
            self.write_cantherm_for_reacts_and_prods(mol)

        for mol in self.reaction.product_mols:
            self.write_cantherm_for_reacts_and_prods(mol)

        self.write_statmech_ts(self.reaction)

        self.write_cantherm_ts(self.reaction)

    def run(self):

        self.cantherm_job.inputFile = self.reaction.label + ".cantherm.py"
        self.cantherm_job.plot = False
        self.cantherm_job.execute()

        for job in self.cantherm_job.jobList:
            if isinstance(job, KineticsJob):
                self.kinetics_job = job

    def set_reactants_and_products(self):

        for reactant in self.reaction.reactant_mols:
            for r in self.kinetics_job.reaction.reactants:
                if reactant.smiles == r.label:
                    r.molecule = [reactant.rmg_molecule]

        for product in self.reaction.product_mols:
            for p in self.kinetics_job.reaction.products:
                if product.smiles == p.label:
                    p.molecule = [product.rmg_molecule]

        self.reaction.rmg_reaction = self.kinetics_job.reaction

        return self.reaction