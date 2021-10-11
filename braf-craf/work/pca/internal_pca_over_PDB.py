import numpy as np
import mdtraj as md
import os
import h5py
from glob import glob
from mdtools.pca import iPCA
from mdtools.trajectory import Trajectory
from tqdm import tqdm
from itertools import combinations
"""
This script should perform a internal coordinate PCA over the PDB
and produce a HDF file of them.

Put in this folder the PDBs used, and the fasta sequences.
- Will be helpful to go through these sequences
 Take the files found via PDB search 
 TODO : Work backwards from here and formalize the previous work 
 on BLAST, cd-hit, and MUSTANG to produce an end to end working 
 project! 
 HINT: The BLAST just needs moving from bmd!
"""
def sg(fp):
    """
    sorted glob.
    move to utils once more to put in there
    """
    return sorted(glob(fp))

def find_actv_segment(top,ca_only=True):
    """
    give a topology and it gives back the atom ids of actv_seg.
    reminder to formalize this selection soon, right now it is
    simply |(D[FG]-20):(D[FG]+20)|
    """
    atom_indices = []
    fasta = top.to_fasta()
    if len(fasta) != 1:
        print(f"Warning: Not single chained!")
        print(f"Taking first chain")
    fasta = fasta[0]
    D_index = fasta.find("DFG")
    low = D_index - 20
    hgh = D_index + 20
    res_id_range = range(low,hgh)
    for res in top.residues:
        if res.index in res_id_range:
            for atom in res.atoms:
                if ca_only:
                    if atom.name == "CA":
                        atom_indices.append(atom.index)
                else:
                    atom_indices.append(atom.index)
    return np.array(atom_indices)

def get_traj_files(folder):
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

directory = "/mnt/hdd/work/braf-craf/actual_work/results/mustang/align_actv_seg/test"
pdb_files = sg(directory+"/*.pdb")
coords = []
for pdb in pdb_files:
    traj = Trajectory(pdb)
    calphs = traj.get_calpha_index()
    coords.append(traj.load_all().xyz[:,calphs])
# Create Dummy trajectory for the iPCA
dummy_top = traj.load_all().atom_slice(calphs).top
coords = np.vstack(coords)
traj = md.Trajectory(coords,dummy_top)
pca = iPCA(traj)
pca.plots()
pca.dump("./iPCA.h5")
# Next we need to add the PCA projections of the full trajectory 
# we have!
with h5py.File("./iPCA.h5","a") as hdf:
    simdata_grp = hdf.create_group("./sim_data")
    tops,crds = get_traj_files("/mnt/hdd/work/braf-craf/actual_work/code"
                                 "/braf-craf/work/trajectories/current")
    components = pca.components
    n_components = components.shape[0]
    total_frames = 0
    vsources_proj = []
    vsources_dist = []
    names = []
    conc_traj = None
    for crd,top in tqdm(zip(crds,tops)):
        # parse name and create group
        name = get_name(crd)
        grp = simdata_grp.create_group(name)
        print(name)
        # load data and find distances,atom_ids,projs
        full_traj = Trajectory(crd,top).load_all()
        n_frames = full_traj.n_frames
        atom_indices = find_actv_segment(full_traj.top)
        atoms = [*combinations(atom_indices,2)]
        distances = md.compute_distances(full_traj,atoms)
        projs = distances.dot(components.T)
        # store the data
        grp.attrs.create("atoms",data=atoms)
        proj_dset = grp.create_dataset("proj",data=projs)
        dist_dset = grp.create_dataset("distances",data=distances)
        grp.attrs.create("atoms",data=atoms)
        # make virtual sources for virtual datasets
        names.append(name)
        vsources_proj.append(h5py.VirtualSource(proj_dset))
        vsources_dist.append(h5py.VirtualSource(dist_dset))
        total_frames += n_frames
        # make conc traj
        if conc_traj is None:
            conc_traj = full_traj
        else:
            conc_traj = md.join((conc_traj,full_traj))

    pdtype = projs.dtype
    ddtype = distances.dtype
    playout = h5py.VirtualLayout(shape=(total_frames,*projs.shape[1:]),
                                    dtype=pdtype)
    dlayout = h5py.VirtualLayout(shape=(total_frames,*distances.shape[1:]),
                                    dtype=ddtype)
    prev = 0
    for psource,dsource in zip(vsources_proj,vsources_dist):
        cur = prev + psource.shape[0]
        playout[prev:cur] = psource
        dlayout[prev:cur] = dsource
        prev = cur
    pvds = simdata_grp.create_virtual_dataset("all_proj",playout)
    pvds.attrs.create("names",data=names)
    dvds = simdata_grp.create_virtual_dataset("all_dist",dlayout)
    dvds.attrs.create("names",data=names)
    print(pvds,dvds)
    conc_traj.save("./all_frames/traj.dcd")
    conc_traj[0].save("./all_frames/traj.pdb")




