#!/bin/bash

esl-sfetch --index $1
cmsearch --nohmmonly --rfam --cut_ga -o $3/$2.out --tblout $3/$2.tblout cms/$2.cm $1
grep -v "^#" $3/$2.tblout | awk '{print $1"/"$8"-"$9, $8, $9, $1}' | esl-sfetch -Cf $1 - > $3/$2.fasta