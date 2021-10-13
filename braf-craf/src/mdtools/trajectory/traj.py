import numpy as np
import mdtraj as md

class Trajectory:
    """
    Currently, if topology is defined it will load just that!
    Try to make it only call the full trajectory when necessary.
    """
    def __init__(self, coord_file, topology_file=None):
        self.coord_file = coord_file
        self.n_frames = len(md.open(coord_file))
        if topology_file:
            self.topology_file = topology_file
            self.top = md.load_topology(topology_file)
        else:
            self.topology_file = None
            try: # if no top, try to get from first frame!
                self.top = md.load_frame(coord_file,0).topology
            except ValueError:
                print("No topology information supplied!")
                raise ValueError
        self.trajectory = None

    def load_all(self):
        if self.trajectory:
            print("already in RAM")
            return self.trajectory
        elif self.topology_file:
            traj = md.load(self.coord_file,top=self.topology_file)
        else:
            traj = md.load(self.coord_file)
        self.trajectory = traj
        return traj

    def load(self,frame,atoms=None):
        """
        Returns frames as xyz, just seems more natural.
        """
        coords = self.coord_file
        top = self.top
        if isinstance(frame,int):
            return md.load_frame(coords,frame,top,
                                 atom_indices=atoms).xyz
        elif hasattr(frame,"__getitem__"):
            if atom_indices is None:
                xyz = np.empty((len(frame),*self.top.shape[1:]))
            else:
                xyz = np.empty((len(frame),len(atom_indices),3))
            for i,frame_n in enumerate(frame):
                xyz[i] = md.load_frame(coords,frame_n,top,
                                       atom_indices=atoms)
            return xyz
        else:
            print(f"Type: {type(frame)} is not understood")
            raise TypeError

    def load_top(self):
        return md.load_topology(self.topology_file)

    def get_calpha_index(self):
        atoms = self.top.atoms
        return [atom.index for atom in atoms if atom.name == "CA"]

    def __getitem__(self,index):
        """
        Basic indexing loads frame xyz
        """
        if isinstance(index,int) or isinstance(index,list):
            return self.load(index)
        else:
            try:
                return [self.load(i) for i in index]
            except Exception:
                print(f"Not understood index: {type(index)}")
                raise TypeError
        

if __name__ == '__main__':
    """
    Unit tests!
    """
    from glob import glob
    data = glob("/mnt/hdd/work/bmd/bigmd/tests/example_data/easy/*/*")
    dcd = data[0]
    pdb = data[1]
    traj = Trajectory(dcd,pdb)
    print(traj.top)
    test_frame = traj.load(100)
    print(test_frame)
