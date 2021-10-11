import numpy as np
import mdtraj as md
import pickle
import h5py
import os 
from time import time as t
from sklearn.decomposition import PCA
from itertools import combinations
from scipy.special import comb
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr as pcc
from matplotlib import pyplot as plt
"""
Simple methods to take a trajectory object and perform a PCA
"""
def pca(data,**kwargs):
    """
    Perform standard PCA
    Data needs to fit in memory
    """
    pca = PCA(**kwargs)
    transformed = pca.fit_transform(data)
    return pca, transformed

class iPCA:
    """
    The class currently requires all data to fit into the RAM!
    """
    @classmethod
    def load(cls,hdf):
        """
        Load object from a saved hdf file.
        Constructor!
        """
        pass

    def __init__(self,traj=None,
                 atom_sel=None,
                 n_components=4,
                 **kwargs):
        """
        PCA on internal coordinates.
        Not finished, right now it just takes a traj obj, atom sel, and does it!
        """
        self.top = None
        self.dist_ids = None
        self.atoms = atom_sel
        self.pca = None
        self.transformed = None
        self.trajectory = None
        self.distances = None
        self.scores = None
        self.n_atoms = None
        self.n_comps = n_components
        if traj is not None:
            self._ipca_from_traj(traj,atom_sel=atom_sel,**kwargs)

    def _ipca_from_traj(self,traj,atom_sel=None,**kwargs):
        self.top = traj.top
        if hasattr(traj,"trajectory"):
            if traj.trajectory is None:
                traj.load_all()
            traj = traj.trajectory
        elif isinstance(traj,md.Trajectory):
            self.trajectory = traj
        else:
            print(f"Type: {type(traj)} not supported.")
        # First calculate the indices needed for distance calculations
        if atom_sel is None:
            n_atoms = self.top.n_atoms
            dist_indices = combinations([*range(n_atoms)],2)
        else:
            n_atoms = len(atom_sel)
            dist_indices = combinations(atom_sel,2)
        # Calculate the iPCA
        n_combis = comb(n_atoms,2)
        self.dist_ids = [*dist_indices]
        self.distances = md.compute_distances(traj,self.dist_ids)
        self.pca, self.transformed = pca(self.distances,
                                         n_components=self.n_comps)
        self.components = self.pca.components_
        self.exp_var = self.pca.explained_variance_ratio_
        self.n_atoms = n_atoms
        self.score()

    def score(self):
        """
        To score we first normalise the components and then
        associate them to the distances and sum up!
        """
        comps = normalize(self.components)
        dist_ids = self.dist_ids
        comp_dict = {}
        for k,(i,j) in enumerate(dist_ids):
            cur_dict = comp_dict.setdefault(i,{})
            cur_dict[j] = comps[:,k]
            cur_dict = comp_dict.setdefault(j,{})
            cur_dict[i] = comps[:,k]
        n_atoms = len(comp_dict)
        n_comps = self.n_comps
        scores = np.empty((n_comps,n_atoms),dtype=np.float32)
        for i in range(n_comps):
            for idx,(_,score_dict) in enumerate(comp_dict.items()):
                cur_score = 0
                for comp in score_dict.values():
                    cur_score += comp[i]**2
                # Here we divide because we have overcounted
                # everything twice because we wanted the dict to be
                # symmetrical
                scores[i,idx] = np.sqrt(cur_score/2)
        print("Score Norms")
        for p,score in enumerate(scores):
            print(p,score,np.linalg.norm(score))
        self.comp_dict = comp_dict
        self.scores = scores

    def PCCs(self):
        dists = self.distances
        projs = self.transformed
        n_distances = dists.shape[0]
        n_components = projs.shape[1]
        assert n_distances == projs.shape[0]
        pccs = np.empty((n_distances,n_components))
        for i in range(n_distances):
            distance = dists[:,i]
            for j in range(n_compoents):
                projection = projs[:,j]
                pccs[i] = pcc(distance, projection)
        self.pccs = pccs

    def dump(self,hdf):
        """
        Save the iPCA to a new hdf (if string) or existing file/group
        """
        if isinstance(hdf,str):
            print(f"Saving to {hdf}")
            target = h5py.File(hdf,"a")
        elif isinstance(hdf,h5py.Group) or isinstance(hdf,h5py.File):
            target = hdf
        else:
            print(f"Input {type(hdf)} not supported")
            raise TypeError
        comps = target.create_dataset("components", data=self.components)
        projs = target.create_dataset("projections",data=self.transformed)
        dists = target.create_dataset("dists",data=self.distances)
        dist_ids = target.create_dataset("dist_ids",data=self.dist_ids)
        exp_var = target.create_dataset("variance",data=self.exp_var)
        if self.atoms:
            atoms = target.create_dataset("atoms",data=self.atoms)
        if self.scores.all():
            scores = target.create_dataset("scores",data=self.scores)
        if isinstance(hdf,str):
            target.close()
        return target

    def plots(self,
             savedir="./plots"):
        """
        Plot the score plots
        """
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        combis = self.dist_ids
        n_atoms = self.n_atoms
        score = self.scores
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,sharex=True,sharey=True,figsize=(12,12))
        ax1.plot(score[0],label="Comp1",marker='o',c="red")
        ax2.plot(score[1],label="Comp2",marker='o',c="blue")
        ax3.plot(score[2],label="Comp3",marker='o',c="orange")
        ax4.plot(score[3],label="Comp4",marker='o',c="green")
        fig.legend(title='PCA scores', bbox_to_anchor=(0.9, 0.7), loc='upper left')
        plt.ylabel("Score")
        plt.xlabel("Residue Index")
        plt.savefig(savedir+f"/scores.png")

if __name__ == '__main__':
    """
    Unit tests!
    """
    from glob import glob
    from mdtools.trajectory.traj import Trajectory as Traj
    data = sorted(glob("/mnt/hdd/work/bmd/bigmd/tests/example_data/easy/*/*"))
    dcd = data[0]
    pdb = data[1]
    traj = Traj(dcd,pdb)
    data = traj.load_all().xyz
    print("Loaded trajectory of shape:")
    print(data.shape)
    print("Performing PCA...")
    t1 = t()
    cas = traj.get_calpha_index()
    pca_obj = iPCA(traj,atom_sel=cas)
    t2 = t()
    print(f"PCA took {round(t2-t1,3)} seconds.")
    
