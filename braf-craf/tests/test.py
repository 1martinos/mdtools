from mdtools.trajectory.traj import Trajectory as mdt
from glob import glob

if __name__ == '__main__':
    data = sorted(glob("/mnt/hdd/work/bmd/bigmd/tests/example_data/easy/*/*"))
    dcd = data[0]
    pdb = data[1]
    traj = mdt(dcd,pdb)
