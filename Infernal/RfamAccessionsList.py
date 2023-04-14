from mysql.connector import connection
import pandas as pd

def query_sql(rna_types):
    conn = connection.MySQLConnection(user='rfamro',
                                    host='mysql-rfam-public.ebi.ac.uk',
                                    database='Rfam', port = 4497)

    for rna in rna_types:
        df = pd.read_sql(f"""SELECT f.rfam_acc
                            FROM family f
                            WHERE f.type LIKE '%{rna}%'""", conn)

        with open(f'Infernal/accessions/{rna}.txt', 'a') as filename:
            for i in range(len(df)):
                filename.write(str(df.iloc[i, 0]) + '\n')

    conn.close()

if __name__ == "__main__":
    rna_types = ['rRNA', 'tRNA', 'sRNA', 'Cis-reg']

    query_sql(rna_types)