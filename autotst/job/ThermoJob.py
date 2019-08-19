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

def check_complete(label, user, partition):
    """
    A method to determine if a job is still running
    """
    command = """squeue -n "{}" -u "{}" -p "{}" """.format(label,user,partition)
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
            sp_calculator=None,
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
        
        self.sp_calculator = sp_calculator
        if self.sp_calculator:
            assert isinstance(self.sp_calculator,Orca)

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

        gaussian_scratch = os.environ['GAUSS_SCRDIR']
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

        if not check_complete(label=label, user=self.discovery_username, partition=self.partition):
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
                """sbatch --exclude=c5003,c3040 --job-name="{0}" --output="{0}.log" --error="{0}.slurm.log" -p {1} -N 1 -n 20 -t 12:00:00 --mem=60GB $AUTOTST/autotst/job/submit.sh""".format(
                    label, self.partition), shell=True, cwd=calc.scratch)

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
        while not check_complete(label=label,user=self.discovery_username,partition=self.partition):
            time.sleep(15)

        complete, converged = self.calculator.verify_output_file(log_path)

        if not complete:
            logging.info(
                "It seems that the file never completed for {} completed, running it again".format(calc.label))
            label = self._submit_conformer(conformer,calc,restart=True)
            time.sleep(10)
            while not check_complete(label=label,user=self.discovery_username,partition=self.partition):
                time.sleep(15)

            complete, converged = self.calculator.verify_output_file(log_path)

        #####

        if (complete and converged):
            return check_isomorphic(conformer=conformer,log_path=log_path)

        if not complete: # try again
            logging.info(
                "It appears that {} was killed prematurely".format(calc.label))
            label = self._submit_conformer(conformer,calc, restart=True)
            time.sleep(10)
            while not check_complete(label=label,user=self.discovery_username,partition=self.partition):
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
            while not check_complete(label=label,user=self.discovery_username,partition=self.partition):
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
                """sbatch --exclude=c5003,c3040 --job-name="{0}" --output="{0}.log" --error="{0}.slurm.log" -p general -N 1 -n 4 -t 10:00 --mem=5GB $AUTOTST/autotst/job/orca_submit.sh""".format(
                    label), shell=True, cwd=orca_calc.directory)
            time.sleep(60)
        # wait unitl the job is done
        while not check_complete(label=label, user=self.discovery_username, partition="general"):
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

    def _generate_sp_inputs(self,conformer,method_name):

        if isinstance(self.sp_calculator,Orca):
            orca_calc = self.sp_calculator

            hf = orca_calc.write_sp_input(method='hf',basis='ma-def2-qzvpp')
        
            ex1 = orca_calc.write_extrapolation_input(option='ep3')
            # ex2 = orca_calc.write_extrapolation_input(option='3',method='ccsd(t)')

            # sp1 = orca_calc.write_sp_input(
            #     method='ccsd(t)', basis='aug-cc-pvtz', atom_basis={'Cl':'cc-pvt(+d)z','H':'cc-pvtz'}, use_atom_basis=True)
            # sp2 = orca_calc.write_sp_input(
            #     method='ccsd(t)', basis='aug-cc-pvqz', atom_basis={'Cl':'cc-pvq(+d)z','H':'cc-pvqz'}, use_atom_basis=True)

            if 35 not in conformer.ase_molecule.get_atomic_numbers():
                sp3 = orca_calc.write_sp_input(method='ccsd(t)-f12', basis='cc-pvdz-f12', atom_basis={'Cl': 'cc-pvt(+d)z'}, use_atom_basis=True)
                sp4 = orca_calc.write_sp_input(method='ccsd(t)-f12', basis='cc-pvtz-f12', atom_basis={'Cl': 'cc-pvq(+d)z'}, use_atom_basis=True)
                # labels = [hf,sp1,sp2,sp3,sp4,ex1,ex2]
                labels = [hf,ex1,sp3,sp4]
            else:
                # labels = [hf,sp1,sp2,ex1,ex2]
                labels = [hf,ex1]

            return labels
        
        else:
            logging.info("We currently do not support single point calculations for {}".format(self.sp_calculator))
            return None



    def _calculate_sp(self,conformer,label):
    
 
        if isinstance(self.sp_calculator,Orca):
            orca_calc = self.sp_calculator

        file_path = os.path.join(orca_calc.directory, label)

        os.environ["label"] = label

        # Do not run orca if log file is already there
        attempted = False
        complete = False
        if os.path.exists(file_path + ".log"):
            attempted = True
            logging.info(
                "It appears that this job already ran")
            if orca_calc.check_NormalTermination(file_path + ".log"):
                complete = True
                logging.info("The single point calculation completed! The log file is {}".format(
                    file_path + ".log"))
                copyfile(
                    file_path + ".log", os.path.join(self.directory, "species", conformer.smiles, label + ".log"))
                return True
                
            else:
                logging.info(
                    "It appears {} did not terminate normally! Rerunning calculation").format(file_path + ".log")
                complete = False

        # In log file does not exist, run Orca
        if not attempted or not complete:
            logging.info(
                "Starting {} single point calculation for {}".format(file_path,conformer))
            if int(self.sp_calculator.nprocs) >= 20:
                exclusive = '--exclusive'
            else:
                exclusive = ''
            subprocess.Popen(
                """sbatch {0} --exclude=c5003,c3040 --job-name="{1}" --output="{1}.log" --error="{1}.slurm.log" -p {2} -N 1 -n {3} -t {4} --mem={5} $AUTOTST/autotst/job/orca_submit.sh""".format(
                    exclusive,label,orca_calc.partition,orca_calc.nprocs,orca_calc.time,orca_calc.mem), shell=True, cwd=orca_calc.directory)  
            time.sleep(15)

        while not check_complete(label=label, user=self.discovery_username, partition=orca_calc.partition):
            time.sleep(15)

                # If the log file exits, check to see if it terminated normally
        if os.path.exists(file_path + ".log"):
            if orca_calc.check_NormalTermination(file_path + ".log"):
                logging.info("The single point calculation {} completed! The log file is {}".format(label,
                    file_path + ".log"))
                copyfile(
                    file_path + ".log", os.path.join(self.directory, "species", conformer.smiles, label + ".log"))
                return True
            else:
                logging.info("It appears the single point calculation {} did not terminate normally".format(label))
                return False
        else:
            logging.info("It appears the single point calculation {} never ran".format(label))
            return False

    def calculate_species(self, method = 'm062x', basis_set = 'cc-pvtz', dispersion= '',
                          recalculate=False, calculate_fod=False, single_point=False):
        """
        Calculates the energy and harmonic frequencies of the lowest energy conformer of a species:
        1) Systematically generates low energy conformers for a given species with an ASE calculator.
        2) Optimizes each low energy conformer with provided Gaussian AutoTST calculator.
        3) Saves the gaussian optimization and frequency analysis log file for the lowest energy conformer of the species.
        """
        species = self.species
        method = method.upper()
        basis_set = basis_set.upper()
        dispersion = dispersion.upper()

        if dispersion == '':
            method_name = method + '_' + basis_set
        else:
            method_name = method + '-' + dispersion + '_' + basis_set

        got_one = False
        label =  "{}_{}_optfreq".format(self.species.smiles[0],method_name)
        log_path = os.path.join(self.calculator.directory,"species",self.species.smiles[0],label+".log")
        if os.path.exists(log_path) and not recalculate:
            logging.info('It appears we already calculated this species')
            logging.info('Checking to see if the log is complete and converge...')
            complete, converged = self.calculator.verify_output_file(log_path)
            
            if (complete and converged):
                logging.info('creating a sample conformer for isomorphism test...')
                conf = Conformer(smiles=species.smiles[0])
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
                species.generate_conformers(ase_calculator=self.conformer_calculator)

            currently_running = []
            processes = {}
            for smiles, conformers in list(species.conformers.items()):

                for conformer in conformers:

                    process = Process(target=self.calculate_conformer, args=(
                        conformer,method,basis_set,dispersion))
                    processes[process.name] = process

            # This loop will block until everything in processes 
            # has been started, and added to currently_running
            for name, process in list(processes.items()):
                while len(currently_running) >= 50:
                    for running in currently_running:
                        if not running.is_alive():
                            currently_running.remove(name)
                    time.sleep(15)
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
            for smiles, conformers in list(species.conformers.items()):
                for conformer in conformers:
                    scratch_dir = os.path.join(
                        self.calculator.directory,
                        method_name,
                        "species",
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

            lowest_energy_file_path = os.path.join(self.calculator.directory, method_name, "species",conformer.smiles,"conformers",lowest_energy_file)
            label =  "{}_{}_optfreq".format(conformer.smiles,method_name)
            dest = os.path.join(self.calculator.directory,method_name, "species",conformer.smiles,label+".log")

            try:
                copyfile(lowest_energy_file_path,dest)
            except IOError:
                os.makedirs(os.path.dirname(dest))
                copyfile(lowest_energy_file_path,dest)

            logging.info("The lowest energy file is {}! :)".format(
                lowest_energy_file))

            parser = ccread(dest, loglevel=logging.ERROR)
            xyzpath = os.path.join(self.calculator.directory,method_name,"species",conformer.smiles,label+".xyz")
            parser.writexyz(xyzpath)

            logging.info("The lowest energy xyz file is {}!".format(
                xyzpath))

        if calculate_fod:  # We will run an orca FOD job

            # Update the lowest energy conformer 
            # with the lowest energy logfile
            label =  "{}_{}_{}".format(self.species.smiles[0],method_name,basis_set)
            conformer = Conformer(smiles=species.smiles[0])
            log = os.path.join(self.calculator.directory,method_name,"species",conformer.smiles,label+".log")
            assert os.path.exists(log),"It appears the calculation failed for {}...cannot calculate fod".format(conformer.smiles)
            atoms = read_log(log)
            conformer.ase_molecule = atoms
            conformer.update_coords_from("ase")
            self._calculate_fod(conformer=conformer,method_name=method_name)

        if single_point:
            
            label =  "{}_{}_{}".format(self.species.smiles[0],method_name,basis_set)
            conformer = Conformer(smiles=species.smiles[0])
            log = os.path.join(self.directory,"species",conformer.smiles,label+".log")
            assert os.path.exists(log), "It appears the calculation failed for {}...cannot perform single point calculations".format(conformer.smiles)
            atoms = self.read_log(log)
            conformer.ase_molecule = atoms
            conformer.update_coords_from("ase")

            valence_dict = {
            1:1,
            6:4,
            7:5,
            8:6,
            9:7,
            17:7,
            35:7
            }

            total_valence = 0
            for atom in conformer.ase_molecule.get_atomic_numbers():
                total_valence += valence_dict.get(atom)

            if total_valence <= 8:
                nprocs = 2
                mem = '12GB'
                t = '01:00:00'
                partition = 'test'
            elif total_valence >8 and total_valence <=14:
                nprocs = 10
                mem = '60GB'
                t = '1-00:00:00'
                partition = 'general'
            else:
                nprocs = 20
                mem = '110GB'
                t = '1-00:00:00'
                partition = 'general'        

            sp_dir = os.path.join(self.directory,method_name,"species",conformer.smiles,"sp")
            if not os.path.exists(sp_dir):
                os.makedirs(sp_dir)
            if self.sp_calculator is None:
                self.sp_calculator = Orca(directory=sp_dir,conformer=conformer,nprocs=nprocs,mem=mem,time=t,partition=partition)
            else:
                if isinstance(self.sp_calculator,Orca):
                    self.sp_calculator.directory = sp_dir
                    self.sp_calculator.conformer = conformer
                    self.sp_calculator.nprocs = nprocs
                    self.sp_calculator.mem = mem
                    self.sp_calculator.mem_per_proc = self.sp_calculator.get_mem_per_proc()
                    self.sp_calculator.time = t
                    self.sp_calculator.partition = partition
                    self.sp_calculator.load_conformer_attributes()

            labels = self._generate_sp_inputs(conformer=conformer)

            if labels is not None:

                currently_running = []
                processes = {}

                for label in labels:

                    process = Process(target=self._calculate_sp, args=(
                            conformer,label))
                    processes[process.name] = process

                for name, process in list(processes.items()):
                    # while len(currently_running) >= 50:
                    #     for running in currently_running:
                    #         if not running.is_alive():
                    #             currently_running.remove(name)
                    process.start()
                    currently_running.append(name)

                while len(currently_running) > 0:
                    for name, process in list(processes.items()):
                        if not (name in currently_running):
                            continue
                        if not process.is_alive():
                            currently_running.remove(name)
            else:
                logging.info('Could not perform single point calcs because single point calculator provided is not currently supported')

        ##### run Arkane
        smiles = self.species.smiles[0]
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
        molecule = self.species.rmg_species[0]
        arkane_calc = Arkane_Input(molecule=molecule,modelChemistry=method_name,directory=arkane_dir,gaussian_log_path=log_path)
        arkane_calc.write_molecule_file()
        arkane_calc.write_arkane_input()
        subprocess.Popen(
                """sbatch --exclude=c5003,c3040 --job-name="{}" --output="{}.log" --error="{}.slurm.log" -p general -N 1 -n 1 -t 10:00 --mem=1GB $RMGpy/Arkane.py arkane_input.py""".format(
                    arkane_calc.label), shell=True, cwd=arkane_calc.directory)
        time.sleep(15)
        while not check_complete(label=arkane_calc.label, user=self.discovery_username, partition='general'):
            time.sleep(10)

        yml_file = os.path.join(arkane_calc.directory,'species','1.yml')
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
            logging.info('It appears the arkane job failed or was never run')




        
