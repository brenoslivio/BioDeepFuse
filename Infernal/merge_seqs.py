from Bio import SeqIO
import re
import os

def merge(genomeseqs_folder):
    alphabets = {'nt': re.compile('^[acgtu]*$', re.I)}

    for genome in os.listdir(genomeseqs_folder):
        genome_folder = os.path.join(genomeseqs_folder, genome)
        for file in os.listdir(genome_folder):
            if file.endswith(".fasta"):

                output_file = os.path.join('raw', file)

                with open(output_file, "a") as f:
                    for record in SeqIO.parse(os.path.join(genome_folder, file), "fasta"):
                        if alphabets['nt'].search(str(record.seq)) is not None:
                            f.write(f">{record.id}\n")
                            f.write(f"{record.seq.back_transcribe()}\n")

if __name__ == '__main__':
    merge('sequences/')