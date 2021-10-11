from bigmd import *
from tqdm import tqdm
from time import time as t
import pickle as p
import os

"""
This is a script to kick off BRAF and keep track of what's been done.
Steps:
    1. Extract the FASTAs from the PDBs I have (cross-check with notes)
    2. Perform a BLAST search for the FASTAs and note the matches
        a) How many identical matches?
        b) Distribution of matches scores
        c) Variation across families of proteins
        d) Matches of the key conserved areas (DFG etc.)
    3. Download these PDBs
    4. Align the PDBs 
    5. Measure key observables and RCs
"""
# This uses my aligner
def align_hits(bs,sequence):
    n_hits = len(bs.hits)
    alignr = aligner(sequence,"")
    alignments = []
    for i,x in enumerate(bs.hits):
        fst = x.pdb.get_fasta(sort_by="chainID")
        seq2 = fst[x.pdb_chain]
        if seq2:
            alignr.set_seq(seq2)
            print(f"PDB number {i} : Sequence - ")
            print(seq2)
            print("\n")
            alignments.append(alignr.n_w()[0])
        else:
            print(f"Error for {x.pdb_id}")
            continue
    return alignemnts

# This uses MAFFT (recommended by Dan)
def blast2MAFFT(blast, f_path=None):
    """
    Function to take a blast search and generate a MAFFT input file.
    The resulting file path is then returned.
    """
    if f_path is None:
        f_path = f"{blast.name}-mafft.inp"
    names = []
    fastas = []
    for hit in blast.hits:
        print(hit.pdb_id)
        if hit.pdb:
            names.append(f"{hit.pdb_id}-{hit.pdb_chain}")
            fasta = hit.pdb.get_fasta(sort_by="chainID") 
            if isinstance(fasta,str):
                pass
            else:
                fasta = fasta[hit.pdb_chain]
            fastas.append(fasta)
        else:
            print(f"No pdb for {hit.pdb_id}")
            continue
    with open(f_path,"w") as f_open:
        for name,fasta in zip(names,fastas):
            f_open.write(f">{name}\n")
            f_open.write(fasta)
    return f_path



if __name__ == "__main__":
    blast_flag = True
    download_pdbs = True

    #1. obtain fastas
    original_topology_path = "/mnt/hdd/work/braf-craf/henryTop.pdb"
    henry_pdb = PDB_file(original_topology_path)
    BRAF_fasta, CRAF_fasta = henry_pdb.get_fasta(sort_by="chainID").values()

    our_align = aligner(BRAF_fasta, CRAF_fasta) # Check similarity between these
    our_align.n_w()

    #2 blast search
    """
        Implement pickling checks so don't have to load everytime. 
    """
    if os.path.isfile("./B_C_RAF_blast.p"):
        with open("./B_C_RAF_blast.p","rb") as f_open:
            t1 = t()
            braf_blast, craf_blast = p.load(f_open)     
            t2 = t()
            print(f"Time Taken {round(t2-t1,4)} seconds to load")
    else:
        _braf_res = "./results/blast_searches/BRAF-blast.xml"
        _craf_res = "./results/blast_searches/CRAF-blast.xml"
        _braf_put_dir = "./results/blast_searches/PDBs/braf"
        _craf_put_dir = "./results/blast_searches/PDBs/craf"
        braf_blast = blast(BRAF_fasta,"BRAF",path=_braf_res)
        craf_blast = blast(CRAF_fasta,"CRAF",path=_craf_res)
        if download_pdbs:
            braf_blast.download_pdbs(pdir=_braf_put_dir)
            craf_blast.download_pdbs(pdir=_craf_put_dir)
        braf_blast.check_for_pdbs(f_path=_braf_put_dir)
        craf_blast.check_for_pdbs(f_path=_craf_put_dir)
        with open("./B_C_RAF_blast.p","wb") as f_open:
            p.dump([braf_blast, craf_blast],f_open)
    """
    TODO : Make plots for the distribution of E-value and identities
           Compare the families of different groupings from this
    """
    #3 alignments
    #braf_alignments = align_hits(braf_blast,BRAF_fasta)
    mafft_inp = blast2MAFFT(braf_blast)
    


