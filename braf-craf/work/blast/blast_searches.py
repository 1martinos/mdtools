from mdtools.pdb import blast
"""
Here we perform 5 blast searches:
 1.   The BRAF monomer 
 2.   The CRAF monomer
 3,4. The activation segments
(5.   The dimer ? )
"""
braf_fasta = "KTLGRDSDWEIPDGQITVGQRIGSGSFGTVYKGKWHGDVAV"\
             "KMLNVTAPTPQLQAFKNEVGVLRKTRHVNILFMGYSTKPQLAIVTQWCEGSLYH"\
             "LHIETKFEMIKLIDIARQTAQGMDYLHAKSIHRDLKSNIFLHEDLTVKIGDFGL"\
             "ATVKSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMT"\
             "GQLPYSNINRDQIFMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKRDERPLFPQ"\
             "ILASIELARSLPKIKIRPRGQRDSYWEIE"
name = "braf"
file_path = "./blasts/braf.xml"
braf_blast = blast(braf_fasta,name,path=file_path)
input()
braf_blast.download_pdbs(pdir="./blasts/braf")

actv_fasta = "ATVKSRWSGSHQFEQLSGSILWM"
name = "actv_seg"
file_path = "./blasts/actv_seg.xml"
actv_blast = blast(actv_fasta,name,path=file_path)
actv_blast.download_pdbs(pdir="./blasts/actv_seg")

"""
Compare bit-scores for graph
"""
