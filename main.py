from Bio import SeqIO
from Bio import AlignIO


import glob

import csv

import pandas as pd
import numpy as np
import gzip
import shutil
import tarfile
import os

# Cases
classes = {
    "TCGA.BRCA": "1",
    "TCGA.LUAD": "2",
    "TCGA.UCEC": "3",
    "TCGA.LGG": "4",
    "TCGA.HNSC": "5",
    "TCGA.PRAD": "6",
    "TCGA.LUSC": "7",
    "TCGA.THCA": "8",
    "TCGA.SKCM": "9",
    "TCGA.OV": "10",
    "TCGA.STAD": "11",
    "TCGA.COAD": "12",
    "TCGA.BLCA": "13",
    "TCGA.GBM": "14",
    "TCGA.LIHC": "15",
    "TCGA.KIRC": "16",
    "TCGA.CESC": "17",
    "TCGA.KIRP": "18",
    "TCGA.SARC": "19",
    "TCGA.ESCA": "20",
    "TCGA.PAAD": "21",
    "TCGA.PCPG": "22",
    "TCGA.READ": "23",
    "TCGA.TGCT": "24",
    "TCGA.LAML": "25",
    "TCGA.THYM": "26",
    "TCGA.ACC": "27",
    "TCGA.MESO": "28",
    "TCGA.UVM": "29",
    "TCGA.KICH": "30",
    "TCGA.UCS": "31",
    "TCGA.CHOL": "32",
    "TCGA.DLBC": "33"
}
chromosomes = {"chr1": "NC_000001.11",
               "chr2": "NC_000002.12",
               "chr3": "NC_000003.12",
               "chr4": "NC_000004.12",
               "chr5": "NC_000005.10",
               "chr6": "NC_000006.12",
               "chr7": "NC_000007.14",
               "chr8": "NC_000008.11",
               "chr9": "NC_000009.12",
               "chr10": "NC_000010.11",
               "chr11": "NC_000011.10",
               "chr12": "NC_000012.12",
               "chr13": "NC_000013.11",
               "chr14": "NC_000014.9",
               "chr15": "NC_000015.10",
               "chr16": "NC_000016.10",
               "chr17": "NC_000017.11",
               "chr18": "NC_000018.10",
               "chr19": "NC_000019.10",
               "chr20": "NC_000020.11",
               "chr21": "NC_000021.9",
               "chr22": "NC_000022.11",
               "chrX": "NC_000023.11",
               "chrY": "NC_000024.10"}


def extract_files():
    files = glob.glob('data/**/*.gz', recursive=True)
    files = list(dict.fromkeys(files))
    for file in files:
        print(file)
        if file.endswith("tar.gz"):
            tar = tarfile.open(file, "r:gz")
            tar.extractall("data")
            tar.close()
        elif file.endswith("tar"):
            tar = tarfile.open(file, "r:")
            tar.extractall("data")
            tar.close()
        else:
            os.system("cd data")
            os.system("gzip -dk " + file)
            os.system("cd ..")


def prepare_data():
    ref = SeqIO.parse(open("GRCh38_latest_genomic.fna"), 'fasta')
    for r in ref:
        print(r.id)
    return 
    sequences = {}
    for chromosome in chromosomes:
        sequences[chromosome] = next(
            (str(x.seq) for i, x in enumerate(ref) if x.id == chromosomes[chromosome]), None)
    files = glob.glob('./data/**/**.maf', recursive=True)
    files = list(dict.fromkeys(files))
    frames = list()
    for file in files:
        frame = pd.read_csv(file, sep='\t', skiprows=5)
        frame = frame[['Hugo_Symbol', 'NCBI_Build', 'Chromosome', 'Start_Position', 'End_Position', 'Strand', 'Variant_Classification',
                       'Variant_Type', 'Reference_Allele',	'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2', 'Allele',	'Gene',	'Feature', 'Feature_type']]
        frame['Sequence'] = frame.apply(lambda row: (str(sequences[row['Chromosome']])[
                                        row['Start_Position']-1:row['End_Position']]), axis=1)
        for c in classes:
            if c in file:
                frame['cancer_type'] =  classes[c]
        frames.append(frame)
    data = pd.concat(frames)
    data.to_csv("data.csv", index=False)
prepare_data()


#extract_files()
