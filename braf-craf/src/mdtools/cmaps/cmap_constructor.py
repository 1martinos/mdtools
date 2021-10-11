import numpy as np
import mdtraj as md 
import h5py as h
from tqdm import tqdm
from itertools import combinations

def traj_to_cmap(coords,hdf_group,
                 topology=None,
                 method="slow",
                 scheme='closest-heavy',
                 residues=None,
                chunks=500):
    """
    Converts trajectories into a hdf dataset of cmaps, supports one
    trajectory.
    Requires coords, a hdf-group, and a name.
    Residues is an iterable object that returns 0-based indices of
    residues involved.

    Coords can either be a MDTrajectory object or
    a file_path to a coordinate file (Topology is necessary
    in this case).

    Slow method loads each frame individually
    Fast method loads all frames (for small trajs)
    """
    # Start with checking inputs
    if isinstance(coords,str):
        if topology:
            traj = md.load(coords,top=topology)
        elif coords.rsplit(".")[-1] == "pdb":
            traj = md.load(coords)
        else:
            print("No topology and not pdb")
            raise TypeError
    elif isinstance(coords,md.Trajectory):
        traj = coords
    else:
        print(f"Input {type(coords)} not understood")
        raise TypeError
    if scheme not in ["ca", 'closest', 
                      'closest-heavy', 
                      'sidechain', 
                      'sidechain-heavy']:
        print(f"Scheme {scheme} not understood")
        raise TypeError
    if not residues:
        residues = [*range(traj.n_residues)]
    # actual code
    n_frames = traj.n_frames
    combis = np.array([*combinations(residues,2)])
    n_comb = len(combis)
    dset_shape = (n_frames,n_comb)
    dset_dtype = np.float32
    dset = hdf_group.create_dataset("cmaps",
                                    dset_shape,
                                    dset_dtype)
    if method == "slow":
        for i,frame in enumerate(tqdm(traj)):
            contacts, indx = md.compute_contacts(frame,scheme=scheme,
                                          contacts=combis)
            dset[i] = contacts
    if method == "fast":
        contacts, indx = md.compute_contacts(traj,scheme=scheme,
                                      contacts=combis)
        dset[:] = contacts
    if method == "chunks":
        for i,traj in enumerate(
                tqdm(md.iterload(coords,chunk=chunks,top=topology))
                                ):
            contacts,indx = md.compute_contacts(traj,scheme=scheme,
                                                contacts=combis)
            dset[(i*chunks):((i+1)*chunks)] = contacts
    dset = hdf_group.create_dataset("indices",
                                    data=indx)
    return dset


