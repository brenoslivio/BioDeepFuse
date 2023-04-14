seqkit rmdup -s raw/Cis-reg.fasta > preprocessed/Cis-reg.fasta
seqkit rmdup -s raw/rRNA.fasta > preprocessed/rRNA.fasta
seqkit rmdup -s raw/sRNA.fasta > preprocessed/sRNA.fasta
seqkit rmdup -s raw/tRNA.fasta > preprocessed/tRNA.fasta

seqkit stats -a preprocessed/Cis-reg.fasta
seqkit stats -a preprocessed/rRNA.fasta
seqkit stats -a preprocessed/sRNA.fasta
seqkit stats -a preprocessed/tRNA.fasta