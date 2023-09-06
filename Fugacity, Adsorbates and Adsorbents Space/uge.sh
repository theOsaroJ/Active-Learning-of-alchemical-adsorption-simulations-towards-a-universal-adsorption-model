#!/bin/bash
#$ -q hpc@@colon
#$ -pe smp 1
#$ -N AL

for ((n=1 ; n<=100; n++));
do

for i in $(cat list.txt)
do

cd $i

cp ../Essentials/ActiveLearning.sh .
cp ../Essentials/GP.py .
cp ../Essentials/Prior.csv .

sed -i 's/CDIR/'$i'/' ActiveLearning.sh
sed -i 's/mmm/'$i'/' GP.py
sed -i 's/CRDIR/'$i'/' ActiveLearning.sh

sh ActiveLearning.sh

cd ..
done

cut -d, -f 3-10 Add.csv > adjust.csv
if [[ -f arr.csv ]]; then
rm arr.csv
fi

if [[ -f Add.csv ]]; then
python3 update.py
cat arr.csv >> Essentials/Prior.csv
fi

rm Add.csv

done
