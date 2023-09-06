### ----------------------------- Author: Etinosa Osaro, Resources: Center for Research Computing, University of Notre Dame-------------------------------------###

### --------------------------------------------------- Objective ------------------------------------------------------------ ###
## This script allows user to run all AL in all 1800 MOFs using different resources as specified in run.sh

#!/bin/bash

for d in $(cat mof.txt)
do
cd $d
mkdir RQ
cd RQ

cp ../../GP.py .
cp ../../run.sh .

sed -i 's/INDEX/'$d'/' run.sh
qsub run.sh
cd ../../
done
