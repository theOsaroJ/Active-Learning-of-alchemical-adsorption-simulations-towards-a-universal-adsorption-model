#!/bin/bash

#Making a datafile
data="Prior.csv"
data2="Add.csv"

#Declaring a string variable Fin which would compare if the active learning was successful or not
Fin="NOT_DONE"

#Loading modules
module load python

# funneling the python output to output
python3 GP.py > result.txt
#Taking the GP relative error and sending to document

som=$(awk 'FNR==3 {print $1}' result.txt)
echo $som >> cummulative.csv

rrsq=$(awk 'FNR==5 {print $2}' result.txt)
echo $rrsq >> squares.csv

mae=$(awk 'FNR==4 {print $1}' result.txt)
echo $mae >> absm.csv

#Initialising variables that will store the array Index for max. uncertainty, and the flag which tells if the code has converged or not
Fugacity=$(awk 'FNR==1 {print $1}' result.txt)
Charge=$(awk 'FNR==1 {print $2}' result.txt)
BL=$(awk 'FNR==1 {print $3}' result.txt)
Epsilon=$(awk 'FNR==1 {print $4}' result.txt)
Siggma=$(awk 'FNR==1 {print $5}' result.txt)
PC11=$(awk 'FNR==1 {print $6}' result.txt)
PC22=$(awk 'FNR==1 {print $7}' result.txt)
Flag=$(awk 'FNR==2 {print $1}' result.txt)
Uptake=$(awk 'FNR==1 {print $8}' result.txt)
mmax=$(awk 'FNR==4 {print $1}' result.txt)

#Removing brackets in the mean.csv from rrmse output from error_estimator code
echo " " >> mean.csv
sed -i 's/[][]//' mean.csv
sed -i 's/[][]//' mean.csv

##Checking if the uncertainty (sigma) is lower than the limit; if not we need to do more simulations
if [[ $Flag == $Fin ]];
then
        # Printing whether the code has converged or not, and the index with max. uncertainty
        echo "Active learning still not finished!"
        echo "CDIR,$mmax,$Fugacity,$Charge,$BL,$Epsilon,$Siggma,$Uptake,$PC11,$PC22" >> ../${data2}
        break
else
                #In case If doesn't satisfy, (which means the uncertainty is lower than 2% for all points), break out of this loop and finish Active learning, the model is ready
echo "Active learning still is finished here!"
echo "CRDIR,$mmax,$Fugacity,$Charge,$BL,$Epsilon,$Siggma,$Uptake,$PC11,$PC22" >> ../${data2}
break
fi
