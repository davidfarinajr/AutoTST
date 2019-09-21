from autotst.calculator.gaussian import read_log,write_input,Gaussian
from autotst.calculator.orca import Orca
from autotst.calculator.arkane_input import Arkane_Input
from autotst.species import Species, Conformer
from autotst.geometry import Bond, Angle, Torsion, CisTrans, ChiralCenter
import cclib
from cclib.io import ccread
from rmgpy.molecule import Molecule as RMGMolecule
from rmgpy.species import Species as RMGSpecies
import rmgpy
from ase.calculators.gaussian import Gaussian as ASEGaussian
from ase.atoms import Atom, Atoms
import ase
import rdkit.Chem.rdDistGeom
import rdkit.DistanceGeometry
from rdkit.Chem.Pharm3D import EmbedLib
from rdkit.Chem import AllChem
from rdkit import Chem
import rdkit
import os
import time
import yaml
from shutil import move, copyfile
import numpy as np
import pandas as pd
import subprocess
import multiprocessing
from multiprocessing import Process, Manager
import logging
FORMAT = "%(filename)s:%(lineno)d %(funcName)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

def get_discovery_username():
    home = os.environ["HOME"]
    discovery_username = home.split('/')[-1]
    try:
        discovery_users = os.listdir('/scratch')
        if discovery_username not in discovery_users:
            logging.warning('It appears that {} is not a valid discovery username'.format(self.discovery_username))
    except:
        pass
    return discovery_username

def check_complete(label, user):
    """
    A method to determine if a job is still running
    """
    command = """squeue -n "{}" -u "{}" """.format(label,user)
    output = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE).communicate()[0]
    
    if len(output.split("\n")) <= 2:
        return True
    else:
        return False

def check_isomorphic(conformer,log_path):
    """
    Compares whatever is in the log file  
    to the SMILES of the passed in 'conformer'
    """
    starting_molecule = RMGMolecule(SMILES=conformer.smiles)
    starting_molecule = starting_molecule.toSingleBonds()

    atoms = read_log(log_path)

    test_molecule = RMGMolecule()
    test_molecule.fromXYZ(
        atoms.arrays["numbers"],
        atoms.arrays["positions"]
    )
    if not starting_molecule.isIsomorphic(test_molecule):
        logging.info(
            "Output geometry of {} is not isomorphic with input geometry".format(log_path))
        return False
    else:
        logging.info(
        "{} was successful and was validated!".format(log_path))
        return True

class ThermoJob():
    """
    A class to deal with the input and output of calculations
    """

    def __init__(
            self,
            species=None,
            calculator=None,
            conformer_calculator = None, # an ASE Calculator object
            partition="general", # The partition to run calculations on
            discovery_username= None, # discovery user account (for checking on jobs in the queue)
            directory = None, # where to save your files 
            scratch = None # a directory for temporary files generated by calculators
            ):

        self.discovery_username = discovery_username
        if self.discovery_username is None:
            self.discovery_username = get_discovery_username()

        assert isinstance(species,Species),'Species must be an AutoTST species, {} was provided'.format(type(species))
        self.species = species 

        self.calculator = calculator
        self.sp_calculator = None
        self.method_name = None
        if self.calculator:
            if directory is None:
                logging.info("Job directory not specified...setting Job directory to calculator directory")
                self.directory = self.calculator.directory
            else:
                logging.info("Setting calculator directory to Job directory")
                self.directory = self.calculator.directory = directory
            if scratch is None:
                logging.info("Job scratch directory not specified...setting Job scratch to calculator scratch")
                self.scratch = self.calculator.scratch
            else:
                logging.info("Setting calculator scratch to Job scratch")
                self.scratch = self.calculator.scratch = scratch
        else:
            logging.info("No calculator specified")
            if directory is None:
                logging.info("No directory specified...setting directory to .")
                self.directory = '.'
            else:
                self.directory = directory
            if scratch is None:
                logging.info("No scratch directory specified...setting scratch to .")
                self.scratch = '.'
            else:
                self.scratch = scratch

        self.conformer_calculator = conformer_calculator
        if self.conformer_calculator:
            self.conformer_calculator.directory = self.scratch
            
        self.partition = partition

        manager = multiprocessing.Manager()
        global results
        results = manager.dict()

        if self.scratch and not os.path.exists(self.scratch):
            os.makedirs(self.scratch)
        if self.directory and not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def __repr__(self):
        return "< ThermoJob '{}'>".format(self.species)

    def _submit_conformer(self, conformer, calc, restart=False):
        """
        A methods to submit a job based on the calculator and partition provided
        """
        assert conformer, "Please provide a conformer to submit a job"

        #self.calculator.conformer = conformer
        #ase_calculator = self.calculator.get_conformer_calc()
        write_input(conformer, calc)
        label = calc.label
        log_path = os.path.join(calc.scratch,label + '.log')

        #label = conformer.smiles + "_{}".format(conformer.index)

        #log_path = os.path.join(ase_calculator.scratch, label)

        #gaussian_scratch = os.environ['GAUSS_SCRDIR']
        gaussian_scratch = '/scratch/westgroup/GAUSS_SCRDIR/'
        os.environ['GAUSS_SCRDIR'] = gaussian_scratch
        if not os.path.exists(gaussian_scratch):
            os.makedirs(gaussian_scratch)

        os.environ["COMMAND"] = self.calculator.command  # only using gaussian for now
        os.environ["FILE_PATH"] = label
        
        attempted = False
        if os.path.exists(log_path):
            attempted = True
            if not restart:
                logging.info(
                    "It appears that this job has already been run, not running it a second time.")

        if not check_complete(label=label, user=self.discovery_username):
            logging.info("It appears that {} is already in the queue...not submitting".format(label))
            return label

        if restart or not attempted:
            if restart:
                logging.info(
                    "Restarting calculations for {}.".format(conformer)
                )
            else:
                logging.info("Starting calculations for {}".format(conformer))
            subprocess.Popen(
                """sbatch --exclude=c5003 --job-name="{0}" --output="{0}.log" --error="{0}.slurm.log" -p {1} -N 1 -n {2} -t {3} --mem={4} $AUTOTST/autotst/job/submit.sh""".format(
                    label,calc.parameters["partition"],calc.parameters["nprocshared"],calc.parameters["time"],calc.parameters["mem"]), shell=True, cwd=calc.scratch)

        return label


    def calculate_conformer(self, conformer, method, basis_set, dispersion):
        """
        A method that optimizes a conformer and performs frequency analysis.
        If the conformer does not convergenge with tight convergence criteria,
        the convergence criteria is loosened to Gaussian's default criteria,
        and the optimization is rerun using the geometry from the last step of the Tight convergence optimization.
        Returns True if Gaussian log is complete and converged, and False if incomplete of unconverged.
        """

        self.calculator.conformer = conformer
        #self.calculator.convergence = "Tight"
        calc = self.calculator.get_conformer_calc(method = method, basis_set = basis_set, convergence = 'Tight', dispersion=dispersion)
        label = calc.label
        # scratch_dir = os.path.join(
        #     self.calculator.directory,
        #     "species",
        #     conformer.smiles,
        #     "conformers")
        log_path = os.path.join(calc.scratch,calc.label + ".log")
        #f = calc.label + ".log"
        logging.info(
            "Submitting conformer calculation for {}".format(calc.label))
        label = self._submit_conformer(conformer,calc)
        time.sleep(15)
        while not check_complete(label=label,user=self.discovery_username):
            time.sleep(15)

        complete, converged = self.calculator.verify_output_file(log_path)

        if not complete:
            logging.info(
                "It seems that the file never completed for {} completed, running it again".format(calc.label))
            calc.parameters["time"] = "12:00:00"
            calc.parameters["nprocshared"] = 12
            label = self._submit_conformer(conformer,calc,restart=True)
            time.sleep(10)
            while not check_complete(label=label,user=self.discovery_username):
                time.sleep(15)

            complete, converged = self.calculator.verify_output_file(log_path)

        #####

        if (complete and converged):
            return check_isomorphic(conformer=conformer,log_path=log_path)

        if not complete: # try again
            logging.info(
                "It appears that {} was killed prematurely".format(calc.label))
            calc.parameters["time"] = "24:00:00"
            calc.parameters["nprocshared"] = 16
            label = self._submit_conformer(conformer,calc, restart=True)
            time.sleep(10)
            while not check_complete(label=label,user=self.discovery_username):
                time.sleep(15)

            complete, converged = self.calculator.verify_output_file(log_path)
            if (complete and converged):
                return check_isomorphic(conformer=conformer,log_path=log_path)
            elif not complete:
                logging.info(
                    "It appears that {} was killed prematurely or never completed :(".format(calc.label))
                return False
            # else complete but not converged

        if not converged:
            logging.info("{} did not converge, trying it as a looser convergence criteria".format(calc.label))

            logging.info("Resubmitting {} with default convergence criteria".format(conformer))
            atoms = read_log(log_path)
            conformer.ase_molecule = atoms
            conformer.update_coords_from("ase")
            self.calculator.conformer = conformer # again, be careful setting this in multiple processes?
            calc = self.calculator.get_conformer_calc(method = method, basis_set = basis_set, convergence = '', dispersion=dispersion)

            logging.info("Removing the old log file that didn't converge, restarting from last geometry")
            os.remove(log_path)

            label = self._submit_conformer(conformer,calc)
            time.sleep(10)
            while not check_complete(label=label,user=self.discovery_username):
                time.sleep(15)

            if not os.path.exists(log_path):
                logging.info(
                "It seems that {}'s loose optimization was never run...".format(calc.label))
                return False

            complete, converged = self.calculator.verify_output_file(log_path)

            if not complete:
                logging.info(
                "It appears that {} was killed prematurely or never completed :(".format(calc.label))
                return False

            elif not converged:
                logging.info("{} failed second QM optimization :(".format(calc.label))
                return False

            else:
                return check_isomorphic(conformer=conformer,log_path=log_path)

        raise Exception("Shoudn't reach here")

    def calculate_sp(self,conformer,method_name,single_point_method):
 
        self.calculator.conformer = conformer
        calc = self.calculator.get_SP_calc(method=single_point_method,convergence='Tight')
        calc.scratch = calc.directory = os.path.join(
            self.directory,
            "species",
            method_name,
            conformer.smiles,
            "sp"
        )
        label = calc.label
        log_path = os.path.join(calc.scratch,calc.label + ".log")
        logging.info(
            "Submitting {} calculation".format(calc.label))
        label = self._submit_conformer(conformer,calc)
        time.sleep(10)
        while not check_complete(label=label,user=self.discovery_username):
            time.sleep(15)

        complete, converged = self.calculator.verify_output_file(log_path)

        if (complete and converged):
            return True

        if not complete: # try again
            logging.info(
                "It appears that {} was killed prematurely".format(calc.label))
            calc.parameters["time"] = "24:00:00"
            calc.parameters["nprocshared"] = 16
            calc.parameters["mem"] = "300Gb"
            label = self._submit_conformer(conformer,calc, restart=True)
            time.sleep(10)
            while not check_complete(label=label,user=self.discovery_username):
                time.sleep(15)

            complete, converged = self.calculator.verify_output_file(log_path)
            
            if (complete and converged):
                return True
            elif not complete:
                logging.info(
                    "It appears that {} was killed prematurely or never completed :(".format(calc.label))
                return False
            # else complete but not converged

        if not converged:
            logging.info("{} did not converge, trying it as a looser convergence criteria".format(calc.label))
            logging.info("Resubmitting {} with default convergence criteria".format(conformer))
            atoms = read_log(log_path)
            conformer.ase_molecule = atoms
            conformer.update_coords_from("ase")
            self.calculator.conformer = conformer 
            calc = self.calculator.get_SP_calc(method=single_point_method,convergence='')
            calc.scratch = calc.directory = os.path.join(
                self.directory,
                "species",
                method_name,
                conformer.smiles,
                "sp"
            )
            label = calc.label

            logging.info("Removing the old log file that didn't converge, restarting from last geometry")
            os.remove(log_path)

            label = self._submit_conformer(conformer,calc)
            time.sleep(10)
            while not check_complete(label=label,user=self.discovery_username):
                time.sleep(15)

            if not os.path.exists(log_path):
                logging.info(
                "It seems that {}'s loose optimization was never run...".format(calc.label))
                return False

            complete, converged = self.calculator.verify_output_file(log_path)

            if not complete:
                logging.info(
                "It appears that {} was killed prematurely or never completed :(".format(calc.label))
                return False

            elif not converged:
                logging.info("{} failed second QM optimization :(".format(calc.label))
                return False

            else:
                return True

    def _calculate_fod(self,conformer,method_name):
        """
        Runs finite temperaure DFT to determine the Fractional Occupation number weighted Density (FOD number).
        Uses Orca's the default functional, basis set, and smear temperature (TPSS, def2-TZVP, 5000 K).
        """
        # specify a directory for FOD input and output
        fod_dir = os.path.join(self.directory, "species",method_name, conformer.smiles, "fod")
        dir = os.path.join(self.directory, "species", method_name,
                            conformer.smiles)
        label = conformer.smiles + "_fod"

        if not os.path.exists(fod_dir):
            os.makedirs(fod_dir)

        # Get orca calculator instance 
        # for lowest energy conformer
        orca_calc = Orca(directory=fod_dir,conformer=conformer)
        orca_calc.write_fod_input()

        # Assign FOD label for calulation and filepath 
        # to save input and output
        file_path = os.path.join(fod_dir, label)

        # Assign environment variables with orca command and path
        os.environ["label"] = label

        # Do not run orca if log file is already there
        attempted = False
        complete = False

        if os.path.exists(file_path + ".log") or os.path.exists(os.path.join(dir, label + ".log")):
            attempted = True
            logging.info(
                "It appears that this job has already been run")
            if os.path.exists(os.path.join(dir, label + ".log")):
                if orca_calc.check_NormalTermination(os.path.join(dir, label + ".log")):
                    complete = True
                    logging.info("The FOD calculation completed!")
                    return True
            elif os.path.exists(file_path + ".log"):
                if orca_calc.check_NormalTermination(file_path + ".log"):
                    complete = True
                    logging.info("The FOD calculation completed!")
                    copyfile(
                        file_path + ".log", os.path.join(dir, label + ".log"))
                    return True
            else:
                logging.info("It appears the FOD job did not terminate normally! Trying FOD job again")
                complete = False

        # In log file does not exist, run Orca
        if not attempted or not complete:
            logging.info(
                "Starting FOD calculation for {}".format(conformer))
            subprocess.Popen(
                """sbatch --exclude=c5003,c3040 --job-name="{0}" --output="{0}.log" --error="{0}.slurm.log" -p test,general -N 1 -n 4 -t 10:00 --mem=5GB $AUTOTST/autotst/job/orca_submit.sh""".format(
                    label), shell=True, cwd=orca_calc.directory)
            time.sleep(10)
        # wait unitl the job is done
        while not check_complete(label=label, user=self.discovery_username):
            time.sleep(10)

        # If the log file exits, check to see if it terminated normally
        if os.path.exists(file_path + ".log"):
            if orca_calc.check_NormalTermination(file_path + ".log"):
                logging.info("The FOD calculation completed! The FOD log file is {}".format(
                    file_path + ".log"))
                copyfile(
                    file_path + ".log", os.path.join(dir, label + ".log"))
                return True
            else:
                logging.info("It appears the FOD job did not terminate normally! The FOD log file is {}".format(
                    file_path + ".log"))
                return False
        else:
            logging.info("It appears the FOD orca job never ran")
            return False

    def calculate_species(self, method = 'm062x', basis_set = 'cc-pvtz', dispersion= None,
                          optimize = True, multiplicity = True,
                          recalculate=False, calculate_fod=True, single_point_method=None,
                          arkane_dft = True, arkane_sp = True):
        """
        Calculates the energy and harmonic frequencies of the lowest energy conformer of a species:
        1) Systematically generates low energy conformers for a given species with an ASE calculator.
        2) Optimizes each low energy conformer with provided Gaussian AutoTST calculator.
        3) Saves the gaussian optimization and frequency analysis log file for the lowest energy conformer of the species.
        """
        species = self.species
        method = method.upper()
        basis_set = basis_set.upper()

        if dispersion:
            dispersion = dispersion.upper()
            method_name = method + '-' + dispersion + '_' + basis_set
            self.method_name = method_name
        else:
            method_name = method + '_' + basis_set
            self.method_name = method_name
        
        if optimize:
            for smiles in self.species.smiles:
                got_one = False
                label =  "{}_{}_optfreq".format(smiles,method_name)
                log_path = os.path.join(self.calculator.directory,"species",method_name,smiles,label+".log")
                if os.path.exists(log_path) and not recalculate:
                    logging.info('It appears we already calculated this species')
                    logging.info('Checking to see if the log is complete and converge...')
                    complete, converged = self.calculator.verify_output_file(log_path)
                    
                    if (complete and converged):
                        logging.info('creating a sample conformer for isomorphism test...')
                        conf = Conformer(smiles=smiles)
                        if check_isomorphic(conformer=conf, log_path=log_path):
                            got_one = True
                            logging.info('The existing log has been verified')
                        else:
                            logging.info('removing existing log and restarting calculation...')
                            os.remove(log_path)
                        
                    else:
                        logging.info('the existing log did not complete or converge')
                        logging.info('removing existing log and restarting calculation...')
                        os.remove(log_path)

                if recalculate or not got_one:
                    logging.info("Calculating geometries for {}".format(species))

                    if self.conformer_calculator:
                        species.generate_conformers(ase_calculator=self.conformer_calculator,multiplicity=multiplicity)

                    currently_running = []
                    processes = {}
                    #for smiles, conformers in list(species.conformers.items()):
                    #for conformers in list(species.conformers[smiles]):
                    
                    for conformer in species.conformers[smiles]:
                        process = Process(target=self.calculate_conformer, args=(
                            conformer,method,basis_set,dispersion))
                        processes[process.name] = process

                    # This loop will block until everything in processes 
                    # has been started, and added to currently_running
                    for name, process in list(processes.items()):
                        # while len(currently_running) >= 50:
                        #     for running in currently_running:
                        #         if not running.is_alive():
                        #             currently_running.remove(name)
                        #     time.sleep(15)
                        process.start()
                        currently_running.append(name)

                    # This loop will block until everything in currently_running
                    # has finished.
                    while len(currently_running) > 0:
                        for name, process in list(processes.items()):
                            if not (name in currently_running):
                                continue
                            if not process.is_alive():
                                currently_running.remove(name)
                        time.sleep(15)

                    results = []
                    #for smiles, conformers in list(species.conformers.items()):
                    #for conformers in species.conformers[smiles]:
                    for conformer in list(species.conformers[smiles]):
                        scratch_dir = os.path.join(
                            self.directory,
                            "species",
                            method_name,
                            conformer.smiles,
                            "conformers"
                        )
                        f = "{}_{}_{}_optfreq.log".format(conformer.smiles, conformer.index, method_name)
                        path = os.path.join(scratch_dir, f)
                        if not os.path.exists(path):
                            logging.info(
                                "It seems that {} was never run...".format(f))
                            continue
                        try:
                            parser = ccread(path, loglevel=logging.ERROR)
                            if parser is None:
                                logging.info(
                                    "Something went wrong when reading in results for {} using cclib...".format(f))
                                continue
                            energy = parser.scfenergies[-1]
                        except:
                            logging.info(
                                "The parser does not have an scf energies attribute, we are not considering {}".format(f))
                            energy = 1e5

                        results.append([energy, conformer, f])

                    results = pd.DataFrame(
                        results, columns=["energy", "conformer", "file"]).sort_values("energy").reset_index()

                    if results.shape[0] == 0:
                        logging.info(
                            "No conformer for {} was successfully calculated... :(".format(species))
                        return False
                    
                    conformer = results['conformer'][0]
                    lowest_energy_file = results['file'][0]

                    # for index in range(results.shape[0]):
                    #     conformer = results.conformer[index]
                    #     lowest_energy_file = results.file[index]
                    #     break

                    logging.info(
                        "The lowest energy conformer is {}".format(lowest_energy_file))

                    lowest_energy_file_path = os.path.join(self.calculator.directory, "species",method_name, conformer.smiles,"conformers",lowest_energy_file)
                    label =  "{}_{}_optfreq".format(conformer.smiles,method_name)
                    dest = os.path.join(self.calculator.directory,"species",method_name,conformer.smiles,label+".log")

                    try:
                        copyfile(lowest_energy_file_path,dest)
                    except IOError:
                        os.makedirs(os.path.dirname(dest))
                        copyfile(lowest_energy_file_path,dest)

                    logging.info("The lowest energy file is {}! :)".format(
                        lowest_energy_file))

                    parser = ccread(dest, loglevel=logging.ERROR)
                    xyzpath = os.path.join(self.calculator.directory,"species",method_name,conformer.smiles,label+".xyz")
                    parser.writexyz(xyzpath)

                    logging.info("The lowest energy xyz file is {}!".format(
                        xyzpath))

        if calculate_fod:  # We will run an orca FOD job
            
            method_name = self.method_name
            # Update the lowest energy conformer 
            # with the lowest energy logfile
            for smiles in self.species.smiles:
                label =  "{}_{}".format(smiles,method_name)
                conformer = Conformer(smiles=smiles)
                log = os.path.join(self.calculator.directory,"species",method_name,conformer.smiles,label+"_optfreq.log")
                assert os.path.exists(log),"It appears the calculation failed for {}...cannot calculate fod".format(conformer.smiles)
                atoms = read_log(log)
                mult = ccread(log,loglevel=logging.ERROR).mult
                conformer.ase_molecule = atoms
                conformer.update_coords_from("ase")
                conformer.rmg_molecule.multiplicity = mult
                self._calculate_fod(conformer=conformer,method_name=method_name)

        if single_point_method:
            method_name = self.method_name
            for smiles in self.species.smiles:
                label =  "{}_{}".format(smiles,method_name)
                conformer = Conformer(smiles=smiles)
                log = os.path.join(self.directory,"species",method_name,conformer.smiles,label+"_optfreq.log")
                assert os.path.exists(log), "It appears the calculation failed for {}...cannot perform single point calculations".format(conformer.smiles)
                complete, converged = self.calculator.verify_output_file(log)
                assert all([complete, converged]), "It appears the log file in incomplete or did not converge"
                atoms = read_log(log)
                mult = ccread(log,loglevel=logging.ERROR).mult
                conformer.ase_molecule = atoms
                conformer.update_coords_from("ase")
                conformer.rmg_molecule.multiplicity = mult

                if isinstance(single_point_method,str):
                    single_point_methods = [single_point_method]
                else: 
                    single_point_methods = single_point_method
                sp_dir = os.path.join(self.directory,"species",method_name,conformer.smiles,"sp")
                if not os.path.exists(sp_dir):
                    os.makedirs(sp_dir)

                currently_running = []
                processes = {}
                
                for sp_method in single_point_methods:
                    process = Process(target=self.calculate_sp, args=(conformer,method_name,
                        sp_method))
                    processes[process.name] = process

                # This loop will block until everything in processes 
                # has been started, and added to currently_running
                for name, process in list(processes.items()):
                    process.start()
                    currently_running.append(name)

                # This loop will block until everything in currently_running
                # has finished.
                while len(currently_running) > 0:
                    for name, process in list(processes.items()):
                        if not (name in currently_running):
                            continue
                        if not process.is_alive():
                            currently_running.remove(name)
                    time.sleep(15)

                if arkane_sp:
                    
                    arkane_dir = os.path.join(
                    self.directory,
                    "species",
                    method_name,
                    smiles,
                    "sp",
                    'arkane'
                )

                    if not os.path.exists(arkane_dir):
                        os.makedirs(arkane_dir)
                    for sp_method in single_point_methods:
                        label = smiles + '_' + sp_method
                        log_path = os.path.join(sp_dir,label + '.log')
                        complete, converged = self.calculator.verify_output_file(log_path)
                        if not all([complete,converged]):
                            logging.info("It seems the log file {} is incomplete or didnt converge".format(log_path))
                            continue
                        #sp_log = [f for f in os.listdir(sp_dir) if f.endswith('.log') and ('slurm' not in f) and (sp_method in f)]
                        #label =  "{}_{}_optfreq".format(smiles,method_name)
                        # log_path = os.path.join(self.directory,"species",method_name,smiles,label+".log")
                        # copyfile(log_path,os.path.join(arkane_dir,label + ".log"))
                        molecule = self.species.rmg_species[0]
                        if molecule.toSMILES() != smiles:
                            for mol in self.species.rmg_species:
                                if mol.toSMILES() == smiles:
                                    molecule = mol
                                    break
                        molecule.multiplicity = mult
                        copyfile(log_path,
                        os.path.join(arkane_dir,label+'.log'))
                        model_chem = sp_method
                        arkane_calc = Arkane_Input(molecule=molecule,modelChemistry=model_chem,directory=arkane_dir,
                        gaussian_log_path=log_path)
                        arkane_calc.write_molecule_file()
                        if 'G' in sp_method:
                            arkane_calc.write_arkane_input(frequency_scale_factor=0.9854)
                        else:
                            arkane_calc.write_arkane_input()
                        subprocess.Popen(
                            """sbatch --exclude=c5003,c3040 --job-name="{0}" --output="{0}.log" --error="{0}.slurm.log" -p test,general,west -N 1 -n 1 -t 10:00 --mem=1GB $RMGpy/Arkane.py arkane_input.py""".format(arkane_calc.label), 
                            shell=True, cwd=arkane_calc.directory)
                        time.sleep(10)
                        while not check_complete(label=arkane_calc.label, user=self.discovery_username):
                            time.sleep(10)

                        yml_file = os.path.join(arkane_calc.directory,'species','1.yml')
                        os.remove(os.path.join(arkane_dir,label + ".log"))

                        dest = os.path.expandvars(os.path.join('$halogen_data','reference_species',"{}".format(sp_method)))
                        if not os.path.exists(dest):
                            os.makedirs(dest)

                        if os.path.exists(yml_file):
                            copyfile(yml_file,os.path.join(dest,smiles + '.yml'))
                            copyfile(
                                os.path.join(self.directory,"species",method_name,smiles,"sp",'arkane',smiles+'.py'),
                                os.path.join(dest,smiles + '.py')
                            )
                            logging.info('Arkane job completed successfully!')

                        else:
                            logging.info('It appears the arkane job failed or was never run for {}'.format(smiles))


        if arkane_dft:
            ##### run Arkane
            for i,smiles in enumerate(self.species.smiles):
                arkane_dir = os.path.join(
                    self.directory,
                    "species",
                    method_name,
                    smiles,
                    'arkane'
                )
                if not os.path.exists(arkane_dir):
                    os.makedirs(arkane_dir)

                label =  "{}_{}_optfreq".format(smiles,method_name)
                log_path = os.path.join(self.directory,"species",method_name,smiles,label+".log")
                mult = ccread(log_path,loglevel=logging.ERROR).mult
                copyfile(log_path,os.path.join(arkane_dir,label + ".log"))
                molecule = self.species.rmg_species[i]
                if molecule.toSMILES() != smiles:
                    for mol in self.species.rmg_species:
                        if mol.toSMILES() == smiles:
                            molecule = mol
                            break
                molecule.multiplicity = mult
                arkane_calc = Arkane_Input(molecule=molecule,modelChemistry=method_name,directory=arkane_dir,gaussian_log_path=log_path)
                arkane_calc.write_molecule_file()
                arkane_calc.write_arkane_input()
                subprocess.Popen(
                    """sbatch --exclude=c5003,c3040 --job-name="{0}" --output="{0}.log" --error="{0}.slurm.log" -p test,general,west -N 1 -n 1 -t 10:00 --mem=1GB $RMGpy/Arkane.py arkane_input.py""".format(arkane_calc.label), 
                    shell=True, cwd=arkane_calc.directory)
                time.sleep(15)
                while not check_complete(label=arkane_calc.label, user=self.discovery_username):
                    time.sleep(10)

                yml_file = os.path.join(arkane_calc.directory,'species','1.yml')
                os.remove(os.path.join(arkane_dir,label + ".log"))
                dest = os.path.join(
                    self.directory,
                    "species",
                    method_name,
                    smiles,
                    smiles + '.yml'
                )


                dest2 = os.path.expandvars(os.path.join('$halogen_data','reference_species',method_name))
                if not os.path.exists(dest2):
                    os.makedirs(dest2)

                if os.path.exists(yml_file):
                    copyfile(yml_file,dest)
                    copyfile(yml_file,os.path.join(dest2,smiles + '.yml'))
                    copyfile(
                        os.path.join(self.directory,"species",method_name,smiles,smiles + '_fod.log'),
                        os.path.join(dest2,smiles + '_fod.log')
                    )  
                    copyfile(
                        os.path.join(self.directory,"species",method_name,smiles,smiles + '_' + method_name + '_optfreq.log'),
                        os.path.join(dest2,smiles + '_' + method_name + '_optfreq.log')
                    )
                    copyfile(
                        os.path.join(self.directory,"species",method_name,smiles,'arkane',smiles+'.py'),
                        os.path.join(dest2,smiles + '.py')
                    )
                    logging.info('Arkane job completed successfully!')

                else:
                    logging.info('It appears the arkane job failed or was never run for {}'.format(smiles))




            

