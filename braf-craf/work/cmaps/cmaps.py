from mdtools.cmaps import traj_to_cmap
from mdtools.utils import *
from time import time as t
import h5py


if __name__ == '__main__':
    tops,crds = get_traj_files("/mnt/hdd/work/braf-craf/"
                               "actual_work/code/braf-craf"
                               "/work/trajectories/braf_mono")
    dsets = []
    with h5py.File("./cmaps_braf.h5","w") as hdf:
        for top,crd in zip(tops, crds):
            t1 = t()
            print(f"Working on {crd}..")
            name = get_name(top)
            group = hdf.create_group(name)
            dsets.append(
                traj_to_cmap(crd,group,
                             topology=top,
                             method="chunks"
                ))
            t2 = t()
            print(f"Took {round(t2-t1,3)} seconds")

    
