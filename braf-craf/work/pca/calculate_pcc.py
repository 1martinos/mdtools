import h5py
import numpy as np
import mdtraj as md
from mdtools.trajectory import Trajectory
from scipy.stats import pearsonr as pcc
from scipy.special import comb
from tqdm import tqdm
from itertools import combinations
"""
Make this work super slow! Only calculate one value at a time so no RAM
breakages!
Here what we want is the PCCs with the distances in the actual simulations
NOT over the PDB data!
"""

with h5py.File("./iPCA.h5","a") as hdf:
    # Load iPCA from the PDB data
    projs = hdf["sim_data/all_proj"]
    # Load BRAF trajectory
    full_traj = Trajectory("./all_frames/traj.dcd","./all_frames/traj.pdb")
    # strip to ca's
    ca_idx = full_traj.get_calpha_index()
    ca_traj = full_traj.load_all().atom_slice(ca_idx)
    # create data arrays & check sizes
    n_atoms = ca_traj.n_atoms
    n_components = projs.shape[1]
    n_distances = int(comb(n_atoms,2))
    # create new HDF for PCCs
    with h5py.File("./PCCs.h5","w") as pcc_hdf:
        pccs_dataset = pcc_hdf.create_dataset("PCCs",
                              shape=(n_distances,
                                     n_components),
                              dtype=np.float32)
        # Now on the fly calculate the distances and the 
        # PCC correlation
        print("Calculating PCCs..")
        combis = combinations(range(n_atoms),2)
        for i,(x,y) in enumerate(tqdm(combis)):
            distance = md.compute_distances(ca_traj, [[x,y]])
            distance = distance.reshape(-1)
            for j in range(n_components):
                projection = projs[:,j]
                pccs_dataset[i,j] = pcc(distance,projection)[0]
