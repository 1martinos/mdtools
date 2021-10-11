import pandas as pd
import xml.etree.ElementTree as ET
import os
from urllib.request import urlretrieve as download
from glob import glob as g
from Bio.Blast import NCBIWWW
from Bio.PDB import PDBList as pdbl
from bigmd import PDB_file as mypdb
from time import time as t
from tqdm import tqdm
from IPython import embed as e
from traitlets.config import get_config
c = get_config()
c.InteractiveShellEmbed.colors = "Linux"
#e(config=c)
"""
This is a quick BLAS parser for handling protein searches.
The aim is to give is a fasta sequence and it return a list of the 
PDBs matching.
There is currently no support for advanced options but to do so 
would simply require a lot of c&p from the NCBIWWW function in Bio.

TODO: Add downloading the PDBs, start analysis on the sites
"""
def merge_dicts(dict1,dict2): # First argument is merged
    for k2,v2 in dict2.items():
        if k2 in dict1.keys():
            if isinstance(dict1[k2],list):
                dict1[k2].append(v2)
            else:
                dict1[k2] = [dict1[k2],v2]
        else:
            print(f"New Key : {k2}") # testing
            dict1[k2] = v2
    return dict1


class hit:  # Holder class just for hit informations: not great usage
    def __init__(self, xml_hit):
        self.hit_num = xml_hit[0].text
        pdb_info = xml_hit[1].text.split("|")[1:]
        self.pdb_id = pdb_info[0]
        self.pdb_chain = pdb_info[1]
        self.description = xml_hit[2].text
        data = [[y.tag.replace("Hsp_",""),y.text] for y in xml_hit[5][0]]
        self.data = dict(data)
        self.pdb = None
        self._not_exists = False

    def download(self,pdb_getter,pdir=None):
    # Bit of cleaning here to swap from ent to pdb files
        f_p = pdb_getter.retrieve_pdb_file(
            self.pdb_id,file_format="pdb",pdir=pdir
            )
        if os.path.isfile(f_p):
            new_path = f_p.rsplit("/",1)[0] + f"/{self.pdb_id}.pdb"
            os.rename(f_p,new_path)
            f_p = new_path
            self._assign(f_p)
        else:
            print(f"No PDB found for {self.pdb_id}")
            self._not_exists = True

    @staticmethod
    def download2(code,pdir=None):
        base_url = "https://files.rcsb.org/download"
        pdb_url = f"{base_url}/{code}.pdb"
        pdir = f"{pdir}/{code}.pdb"
        try:
            download(pdb_url,pdir)
        except Exception:
            print(f"File {code} not found.")
        return None
        

    def _assign(self,f_p):
        self.pdb_path = f_p
        self.pdb = mypdb(f_p)

    def check_for_pdb(self,pdir=None):
        if pdir is None:
            pdir="./pdbs/"
        elif pdir[-1] != "/":
            pdir = pdir + "/"
        if not os.path.isdir(pdir):
            raise Exception
        matches =  g(pdir + f"{self.pdb_id}*")
        if matches:
            if not self.pdb:
                self._assign(matches[0])
            return True
        else:
            return False


    def __str__(self):
        return f"PDB-{self.pdb_id}-{self.pdb_chain}"
    def __repr__(self):
        return self.__str__()

class blast: # Class for handling BLAST searches of amino acid chains
    # Input the result columns
    def bsearch(self):
        if self.path and os.path.isfile(self.path):
            print("WARNING: Search exists")
            print("Input to proceed...")
            input()
        print("Searching BLAST...")
        t1 = t()
        self.results = NCBIWWW.qblast(self.program,
                                      self.database,
                                      self.sequence,hitlist_size=self.hitlen)
        t2 = t()
        print(f"Search took {round(t2-t1,4)} seconds")
        if not self.path:
            self.path = f"{self.name}-blast.xml"
        with open(self.path,"w") as output_xml:
            output_xml.write(self.results.read())
        self.parse_search()

    def parse_search(self):
        """ Aim of this is to convert an xml file into a dataframe
            I would like not to use the XML as I don't think it's elegant
            or readable here
        """
        if self.path:
            t1 = t()
            tree = ET.parse(self.path)
            iteration = tree.findall("./BlastOutput_iterations/Iteration/")
            self.query_length = iteration[3].text  #Empirical magic numbers okay?
            hits = [hit(x) for x in list(iteration[-2])]
            self.hits = hits
            mega_dict = hits[0].data
            for x in hits[1:]:
                mega_dict = merge_dicts(mega_dict,x.data)
            mega_dict["PDB ID"] = [x.pdb_id for x in hits]
            mega_dict["Chain"]  = [x.pdb_chain for x in hits]
            mega_dict["Description"] = [x.description for x in hits]
            self.df = pd.DataFrame.from_dict(mega_dict)
            t2 = t()
            print(f"Time taken to parse {t2-t1}")
        else:
            print("Error no xml file found")
            raise Exception

    def download_pdbs(self,pdir=None):
        default = "./PDBs"
        if not pdir: 
            if not os.path.isdir(default):
                os.makedirs(default)
            pdir = default
        elif pdir:
            if not os.path.isdir(pdir):
                os.makedirs(pdir)
            files = [f.split(".")[0] for f in os.listdir(pdir)]
        
        hit_bar = tqdm(self.hits)
        for x in hit_bar:
            if x.pdb_id not in files:
                hit_bar.set_description(f"Downloading {x}...")
                file = x.download2(x.pdb_id,pdir=pdir)
                if file:
                    x._assign(file)

        """
        TODO : Add method to check if PDBs exist and if they dont then 
        them from the hit list! 
        """
    def check_for_pdbs(self,f_path=None):
        # Add f_path funtionality
        for hit in self.hits:
            hit.check_for_pdb(pdir=f_path)
            if hit._not_exists:
                self.hits.remove(hit)
            


    def __init__(self,
                 sequence,
                 name,
                 program="blastp",
                 database="pdb",
                 path=None, # use this to give preexisting search results
                 search=None,hitlen=1000): # auto search if not provided path
        self.sequence = sequence
        self.program = program
        self.database = database
        self.path = path
        self.name = name
        self.hitlen = hitlen
        if self.path and os.path.isfile(self.path):
            self.parse_search()
        else:
            self.bsearch()

if __name__ == '__main__':
    braf_fasta = "KTLGRDSDWEIPDGQITVGQRIGSGSFGTVYKGKWHGDVAV"\
    "KMLNVTAPTPQLQAFKNEVGVLRKTRHVNILFMGYSTKPQLAIVTQWCEGSLYH"\
    "LHIETKFEMIKLIDIARQTAQGMDYLHAKSIHRDLKSNIFLHEDLTVKIGDFGL"\
    "ATVKSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMT"\
    "GQLPYSNINRDQIFMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKRDERPLFPQ"\
    "ILASIELARSLPKIKIRPRGQRDSYWEIE"
    name = "braf"
    # Implement check for saved results
    res_path = f"./{name}-blast.xml"
    if os.path.exists(res_path):
        bs = blast(braf_fasta,name,path=res_path) 
    else:
        bs = blast(braf_fasta,name) 
    bs.download_pdbs()
