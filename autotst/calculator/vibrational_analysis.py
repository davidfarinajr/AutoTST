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
import pandas as pd
import numpy as np
from cclib.io import ccread
from autotst.reaction import Reaction, TS
from autotst.species import Species, Conformer


def percent_change(original, new):
    "A function to calculate the percent change between two values"
    percent_change = (abs(new - original) / original) * 100
    return percent_change


class VibrationalAnalysis():
    """
    A class that allows one to perform vibrational analysis. It takes a
    TS and uses it to parse out the finalized geometry from a
    corresponding log file, and then compares the geometry before and after
    displacement from the imaginary frequency.
    """

    def __init__(self, ts=None, scratch="."):
        """
        Variables:
        - ts (TS): A TS object that you want to run vibratinal analysis on
        - scratch (str): the directory where log files of the TS are located
        """
        self.ts = ts
        self.scratch = scratch

    def __repr__(self):
        if self.ts is None:
            label = None
        else:
            label = self.ts.reaction_label
        return '<Vibrational Analysis "{0}">'.format(
            label)

    def get_log_file(self, scratch=".", ts=None):
        """
        This method obtains the logfile name from the TS

        Variables:
        - scratch (str): The directory where the log files are located in
        - ts (TS): the TS object of interest

        Returns:
        - log_file (str): a path to the log file of interest
        """
        log_file = os.path.join(
            scratch,
            "ts",
            ts.reaction_label,
            "conformers",
            "{}_{}_{}.log".format(ts.reaction_label, ts.direction, ts.index))
        return log_file

    def parse_vibrations(self, log_file=None):
        """
        This method obtains the vibrations from the log file of interest using
        cclib. It then creates a zipped list with the vibrational frequencies
        and their corresponding displacements.

        Variables:
        - log_file (str): the log file you want to obtain vibrations from. Often found from `get_log_file`

        Returns:
        - vibrations (list): a list of the vibrations and their corresponding displacements in XYZ coordinates
        """

        if not log_file:
            log_file = self.log_file

        assert os.path.exists(log_file), "Log file provided does not exist"

        log_file_info = ccread(log_file)
        vibrations = list(zip(log_file_info.vibfreqs, log_file_info.vibdisps))

        return vibrations

    def obtain_geometries(self, ts, vibrations):
        """
        This method obtains the previbrational geometry (the geometry returned
        by a quantum optimizer), and the postvibrational geometry.

        Variables:
        - ts (TS): A transition state object of interest
        - vibrations (list): a list of the vibrations and their corresponding displacements in XYZ coords. Often from `parse_vibrations`

        Returns:
        - pre_geometry (ASEAtoms): an ASEAtoms object containing the geometry before applying the vibrations
        - post_geometry (ASEAtoms): an ASEAtoms object containing the geometry after applying the vibrations
        """

        assert isinstance(ts, TS)

        pre_geometry = ts.ase_molecule.copy()
        post_geometry = ts.ase_molecule.copy()

        for vib, displacements in vibrations:
            if vib < 0:  # Finding the imaginary frequency
                post_geometry.arrays["positions"] -= displacements

        return pre_geometry, post_geometry

    def obtain_percent_changes(self, ts, before_geometry, post_geometry):
        """
        This method takes the connectivity of a TS and then uses
        that to identify the percent change of the bonds
        between the before_geometry and the post_geometryself.

        This returns a dataframe with the type of geometry, the indicies in it,
        if it is close to the reaction center, and the percent change of that
        geometry.

        Variables:
        - ts (TS): the transition state of interest
        - berfore_geometry (ASEAtoms): the ASEAtoms object describing the geometry before applying the vibrations
        - post_geometry (ASEAtoms): the ASEAtoms object describing the geometry after applying the vibrations

        Returns:
        - results (DataFrame): a pandas dataframe containing the bond index, atom indicies of those bonds,
            if the bond is in the reaction center, and percent change of all bonds

        """

        results = []

        for bond in ts.bonds:
            i, j = bond.atom_indices
            before = before_geometry.get_distance(i, j)
            after = post_geometry.get_distance(i, j)
            results.append([bond.index, bond.atom_indices,
                            bond.reaction_center, percent_change(before, after)])

        results = pd.DataFrame(results)
        results.columns = ["index", "atom_indices", "center", "percent_change"]

        return results

    def validate_ts(self, scratch=None, ts=None):
        """
        A method designed to run the above and return a bool that states if we
        have arrived at a TS. We say we have arrived at a TS if the average
        change of geometries in the reaction center is one order of magnitude
        geater elsewhere.

        Variables:
        - scratch (str): the path to where the log file of the optimized transition state is
        - ts (TS): the transition state of interest

        Returns:
        - (bool): True if the TS can be validated via vibrational analysis and False if it cannot
        """
        if scratch is None:
            scratch = self.scratch
        if ts is None:
            ts = self.ts
        try:
            self.log_file = self.get_log_file(self.scratch, ts)

            self.vibrations = self.parse_vibrations(self.log_file)

            self.pre_geometry, self.post_geometry = self.obtain_geometries(
                self.ts, self.vibrations)

            self.percent_changes = self.obtain_percent_changes(
                self.ts, self.pre_geometry, self.post_geometry)

            center_values = np.log(
                self.percent_changes[self.percent_changes.center].percent_change.mean())
            shell_values = np.log(
                self.percent_changes[self.percent_changes.center != True].percent_change.mean())

            if center_values > shell_values + 1:
                logging.info("Vibrational analysis was successful")
                return True

            else:
                logging.info(
                    "Cannot reasonably say that we have arrived at a TS through vibrational analysis.")
                return False
        except:
            logging.info("Something went wrong...")
            logging.info("Cannot verify via vibrational analysis")
            return False
