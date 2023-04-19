from Bio import SeqIO
import re
import os
import numpy as np

def preprocess():
    alphabets = {'nt': re.compile('^[acgtu]*$', re.I)}
    files = {"train": "data/nRC/dataset_Rfam_6320_13classes.fasta",
             "test": "data/nRC/dataset_Rfam_validated_2600_13classes.fasta"}

    types = set()

    for record in SeqIO.parse(files["train"], "fasta"):
        if alphabets['nt'].search(str(record.seq)) is not None:
            types.add(record.description.split()[-1])

    for fasta in files:
        print(f"FASTA: {files[fasta]}")

        for type in types:
            with open(f"data/nRC/{fasta}/{type}.fasta", "a") as f:
                for record in SeqIO.parse(f"{files[fasta]}", "fasta"):
                    if alphabets['nt'].search(str(record.seq)) is not None:
                        type_file = record.description.split()[-1]
                        if type == type_file:
                            f.write(f">{record.description}\n")
                            f.write(f"{record.seq.back_transcribe()}\n")

if __name__ == '__main__':
    preprocess()