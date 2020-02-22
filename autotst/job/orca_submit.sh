#!/bin/sh

module purge
module load orca/4.0.1
module load gcc/7.2.0
module load openmpi/2.0.4
/shared/apps/orca/orca_4_0_1_linux_x86-64_openmpi202/orca "$FILE_PATH.inp"