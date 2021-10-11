from glob import glob
import os

def sglob(path):
    return sorted(glob(path))

def get_traj_files(folder=None):
    if folder is None:
        folder = "/mnt/hdd/work/braf-craf/actual_work/code"\
                 "/braf-craf/work/trajectories/current" 
    tops = []
    crds = []
    for root,folder,files in os.walk(folder):
        for f in files:
            full_path = os.path.join(root,f)
            extension = f.split(".",1)[-1]
            if extension == "pdb":
                tops.append(full_path)
            elif extension == "dcd":
                crds.append(full_path)
    tops = sorted(tops)
    crds = sorted(crds)
    return tops,crds

def get_name(string):
    string = string.rsplit("/",1)[-1].split(".",1)[0]
    return string

def get_atom_ids(traj):
    atoms = [atom.index for atom in traj.top.atoms]
    return atoms

def get_res_ids(traj):
    resid = [res.index for res in traj.top.residues]
    return resid
