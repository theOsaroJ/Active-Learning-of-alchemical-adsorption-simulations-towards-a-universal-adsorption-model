#!/bin/bash
#$ -q hpc@@colon
#$ -N DL
#$ -pe smp 64

###----------------This is to create folder with MOF name, CompleteData.csv and Prior.csv----------###
##Removing exponential
sed -i 's/1e+05/100000/' Todo_Complete_training.csv
sed -i 's/5e+05/500000/' Todo_Complete_training.csv
sed -i 's/1e+06/1000000/' Todo_Complete_training.csv
sed -i 's/5e+06/1000000/' Todo_Complete_training.csv
sed -i 's/1e+07/10000000/' Todo_Complete_training.csv


#Splitting the data by MOF using the split.py
python3 split.py

#Creating a folder and putting each file into folder
for f in OPT*.csv
do
  subdir=${f%%.*}
  [ ! -d "$subdir" ] && mkdir -- "$subdir"
  mv -- "$f" "$subdir"
done

#Entering into each MOF folder to execute a command
for d in *; do
  if [ -d "$d" ]; then
    ( cd "$d" &&
#Creating a copy of file to replace fugacity,eps,sig,charge and bondlength with placehold

head -n 2 OPT*.csv > one.csv

#picking the placeholders index now
vvv=$(awk 'FNR==2 {print $1}' one.csv | awk -F',' '{print $28}')
www=$(awk 'FNR==2 {print $1}' one.csv | awk -F',' '{print $29}')
xxx=$(awk 'FNR==2 {print $1}' one.csv | awk -F',' '{print $30}')
yyy=$(awk 'FNR==2 {print $1}' one.csv | awk -F',' '{print $31}')
zzz=$(awk 'FNR==2 {print $1}' one.csv | awk -F',' '{print $32}')

#Creating a new file to delete and paste later.
cat one.csv > two.csv
cut -d, -f 1-27 two.csv > three.csv

#passing the values to a file
echo "Fugacity,chg,bond_length,eps_eff,sig_eff" > four.csv
echo "$vvv,$www,$xxx,$yyy,$zzz" >> four.csv

#replacing the indexes by placeholders index with placeholders
sed -i 's/'${vvv}'/AAA/' four.csv
sed -i 's/'${www}'/BBB/' four.csv
sed -i 's/'${xxx}'/CCC/' four.csv
sed -i 's/'${yyy}'/DDD/' four.csv
sed -i 's/'${zzz}'/EEE/' four.csv

#appending file
paste three.csv four.csv -d','> DL_AL.csv

#converting name of main files
mv OPT*.csv CompleteData.csv

#removing unwanted files
rm one.csv two.csv three.csv four.csv
 )
  fi
done

#Entering into each MOF folder to create the Prior.csv after copying split python file
for d in *; do
  if [ -d "$d" ]; then
    ( cp prior.py "$d" &&
cd "$d"
#taking columns that matter
cat CompleteData.csv > one.csv
cut -d, -f 28-33 one.csv > two.csv

module load python
python3 prior.py

#deleting the header
sed -i '1d' 1*csv 2*csv 5*csv 7*csv

#deleting everything except the first and last line
sed -i -n '1p;$p' 1*csv 2*csv 5*csv 7*csv

#Creating the Prior now
echo "Fugacity,chg,bond_length,eps_eff,sig_eff,prediciton" > three.csv
cat 1*csv 2*csv 5*csv 7*csv >> three.csv

sort -n three.csv > Prior.csv
#Removing the annoying ^M character
sed -i "s/\r//g" Prior.csv

rm 1*csv 2*csv 5*csv 7*csv one.csv two.csv three.csv
 )
  fi
done
