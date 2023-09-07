#!/bin/bash
#$ -q hpc@@colon
#$ -pe smp 1
#$ -N bagging

for i in $(cat mof.txt)
do
cd $i

rm sfile_*

cp ../bagging.py .

cp *.csv data.csv
sed -i '1, 1d' data.csv

python3 bagging.py

cd ..
done

for i in $(cat mof.txt)
do

cd $i

for ((n=1; n<=50; n++)); do
cat "sfile_$n.csv" >> "../data_$n.csv"
done

cd ../
done

mkdir Bagging
mv data_* Bagging

cd Bagging

for ((n=1; n<=50; n++)); do
cut -d, -f 28-35 data_$n.csv > chunk_$n.csv
done

mkdir Chunks
mv chunk_* Chunks
