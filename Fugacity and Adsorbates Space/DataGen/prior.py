#!/usr/bin/env python
# coding: utf-8

#Importing important libraries
import csv

with open('two.csv') as fin:
    csvin = csv.DictReader(fin)
    # Spliiting by Fugacity
    outputs = {}
    for row in csvin:
        cat = row['Fugacity']
        # Open a new file and write the header
        if cat not in outputs:
            fout = open('{}.csv'.format(cat), 'w')
            dw = csv.DictWriter(fout, fieldnames=csvin.fieldnames)
            dw.writeheader()
            outputs[cat] = fout, dw
        # Always write the row
        outputs[cat][1].writerow(row)
    # Close all the files
    for fout, _ in outputs.values():
        fout.close()
