import numpy as np
import h5py 
import mdtraj as md
import mdtools
from mdtools.cmaps import ContactMap
from mdtools.utils import *
from tqdm import tqdm
"""
Basically what I want to do here is create a virtual dataset of all
conformations of the activation segment!

To aid visualising this I also want it to generate trajectories
of just these sections of topology.

These should be seperate files but arent for now.

Now to begin with I will be using binary contact maps with a cutoff of
8 Angstroms just to side with Shozebs paper. Therefore the network is only 
really for use for clustering purposes.
"""
def find_actv_seg(top,inverse=False):
    """
    This custom function can be changed.
    Simply takes returns the residues of a chosen selection.
    """
    if isinstance(top,md.Trajectory):
        top = top.top
    fasta = top.to_fasta()[0]
    D_index = fasta.find("DFG")
    low = D_index 
    hgh = D_index + 26
    if inverse:
        res_inv = top._residues[:low] + top._residues[:hgh]
        return top._residues[low:hgh], res_inv
    return top._residues[low:hgh]

def slice_to_res_indices(traj,indices,inverse=True):
    """
    Simple function to take residue indices and a traj and slice it down to
    just these residues. Should be already in mdtraj tbh.
    """
    top = traj.top
    atoms = []
    for i in indices:
        for atom in top._residues[i].atoms:
            atoms.append(atom.index)
    if inverse:
        invs = [atom for atom in top.atoms if atom.index not in atoms]
        return traj.atom_slice(atoms), traj.atom_slice(invs)
    return traj.atom_slice(atoms)

def ttv_split_indices(length,training=0.8,
                             test=0.1,
                             validation=0.1):
    def choices(values,size):
        c = np.random.choice(values,size,replace=False)
        c = sorted(c)
        return c
    """
    Given the length of a 1D data array, it converts to
    the indices required for a training, test, validation split.
    Any leftover data goes to the training!
    """
    if training + test + validation != 1:
        print("Total split must add up to 1")
        raise ValueError
    indices = [*range(length)]
    training = int(training*length)
    test = int(test*length)
    validation = int(validation*length)
    total = training+test+validation
    if total != length:
        training += length - total
    train_idx = choices(indices,training)
    indicies = [indices.remove(i) for i in train_idx]
    test_idx = choices(indices,test)
    indicies = [indices.remove(i) for i in test_idx]
    valid_idx = choices(indices,validation)
    return train_idx,test_idx,valid_idx

if __name__ == "__main__":

    mono_traj_dir = "/mnt/hdd/work/braf-craf/actual_work/code/"\
                    "braf-craf/work/trajectories/braf_mono"
    sliced_traj_dir = "/mnt/hdd/work/braf-craf/actual_work/"\
                      "code/braf-craf/work/ml/vae_for_actv_segment/"\
                      "data/sliced_trajectories"
    contact_maps_DB = "/mnt/hdd/work/braf-craf/actual_work/"\
                      "code/braf-craf/work/cmaps/cmaps_braf.h5" 
    top_file = "/mnt/hdd/work/braf-craf/actual_work/"\
               "code/braf-craf/work/trajectories/"\
               "braf_mono/MyInIn.pdb" 

    # First lets find the segment we want to focus on
    braf_topology = md.load(top_file)
    chosen_residues = find_actv_seg(braf_topology)
    chosen_idxs = [r.index for r in chosen_residues]
    chosen_resSeq = [r.resSeq for r in chosen_residues]
    # inverse selection


    # next lets strip the original trajectories
    """
    TODO: Rewrite this as a higher level function because this will be
    important to do quickly later!
    """
    mono_dcds = sglob(mono_traj_dir + "/*.dcd")
    cmaps_DB = h5py.File(contact_maps_DB,"a")
    all_traj = None
    all_traj_inv = None
    print("Slicing trajectories...")
    for dcd in tqdm(mono_dcds):
        name = get_name(dcd)
        traj = md.load(dcd,top=top_file)
        traj, traj_inv = slice_to_res_indices(traj,chosen_idxs)
        traj.save(sliced_traj_dir + f"/{name}-sliced.dcd")
        # Save inverse for wacky BERT idea
        traj_inv.save(sliced_traj_dir+f"/inverse/{name}-inv.dcd")
        if all_traj is None:
            all_traj = traj
        else:
            all_traj = md.join((all_traj,traj))
            all_traj_inv = md.join((all_traj_inv,traj))
    all_traj[0].save("sliced_top.pdb")
    all_traj.save(sliced_traj_dir+"/all.dcd")
    all_traj_inv[0].save("inverse/inv_top.pdb")
    all_traj_inv.save(sliced_traj_dir+"inverse/all_inv.dcd")

    # Now lets make this training data
    # First a virtual dataset of all the data in the old hdf
    n_frames = all_traj.n_frames
    n_residues = braf_topology.n_residues
    cmap_dset_shape = (n_frames,n_residues,n_residues)
    all_cmaps = cmaps_DB.require_dataset("all_cmaps",
                                        shape=cmap_dset_shape,
                                        dtype=np.float32)

    cur = 0
    print("Creating total dataset...")
    for x in tqdm(cmaps_DB):
        print("NAME:",x)
        if isinstance(cmaps_DB[x], h5py.Group):
            dset = ContactMap(cmaps_DB[x]).all()
            # Here the times 10 converts to Angstroms! 
            # I think it's important as just to avoid getting 
            # smaller and shit results!
            # test this somehow...
            all_cmaps[cur:(cur+dset.shape[0])] = dset * 10 < 8
            cur += dset.shape[0]
        else:
            print("Is a datset.\n")
            continue
    print("All DataSet Shape:",all_cmaps.shape)
    print("n_frames:",n_frames)
    
    # Calculate shapes of test,train,valid datasets
    n_residues = len(chosen_idxs)
    train_idx,test_idx,val_idx = ttv_split_indices(n_frames)
    train_virt_shape = (len(train_idx),n_residues,n_residues)
    test_virt_shape = (len(test_idx),n_residues,n_residues)
    valid_virt_shape = (len(val_idx),n_residues,n_residues)
    # Make the HDF
    train_hdf = h5py.File("./training_data.h5","w")
    vsource = h5py.VirtualSource(all_cmaps)
    low, hgh = chosen_idxs[0], chosen_idxs[-1] + 1
    # train
    print("Making train dataset...")
    train_group = train_hdf.create_group("train")
    virtual_layout = h5py.VirtualLayout(shape=train_virt_shape,
                                        dtype=np.float32)
    virtual_layout[:] = vsource[train_idx,low:hgh,low:hgh]
    train_group.create_virtual_dataset("data",virtual_layout)
    train_group.create_dataset("indices",data=train_idx)
    # test
    print("Making test dataset...")
    test_group = train_hdf.create_group("test")
    virtual_layout = h5py.VirtualLayout(shape=test_virt_shape,
                                        dtype=np.float32)
    virtual_layout[:] = vsource[test_idx,low:hgh,low:hgh]
    test_group.create_virtual_dataset("data",virtual_layout)
    test_group.create_dataset("indices",data=test_idx)
    # valid
    print("Making validation dataset...")
    valid_group = train_hdf.create_group("validation")
    virtual_layout = h5py.VirtualLayout(shape=valid_virt_shape,
                                        dtype=np.float32)
    virtual_layout[:] = vsource[val_idx,low:hgh,low:hgh]
    valid_group.create_virtual_dataset("data",virtual_layout)
    valid_group.create_dataset("indices",data=val_idx)
    
    cmaps_DB.close()
    train_hdf.close()
