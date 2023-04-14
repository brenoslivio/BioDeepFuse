import subprocess
import os

def extract_seqs(genomes_folder, output_folder, labels):

    genome_files = []

    for file in os.listdir(genomes_folder):
        if file.endswith(".fasta"):
            genome_files.append(file)

    for genome_file in genome_files:
        genome_name = os.path.splitext(genome_file)[0]

        genome_path = os.path.join(genomes_folder, genome_file)
        seq_folder = os.path.join(output_folder, genome_name)
        os.makedirs(seq_folder, exist_ok=True)

        for label in labels:
            subprocess.run(['./run_infernal.sh', genome_path, label, seq_folder])

def fetch_cm(rfam_cm, labels):
    for label in labels:
        cm_output = open(f'cms/{label}.cm', 'a')
        subprocess.run(['cmfetch', '-f', rfam_cm, f'accessions/{label}.txt'], stdout=cm_output)

if __name__ == '__main__':
    labels = ['rRNA', 'tRNA', 'sRNA', 'Cis-reg']

    # fetch_cm('Infernal/Rfam.cm', labels)
    extract_seqs("genomes", "sequences", labels)