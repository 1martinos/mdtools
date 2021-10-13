import numpy as np
import mdtraj as md
import pickle
import h5py
import os 
from time import time as t
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA as incpca
from itertools import combinations
from scipy.special import comb
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr as pcc
from matplotlib import pyplot as plt
from mdtools.trajectory import Trajectory as Traj
from numpy.linalg import norm
"""
Simple methods to take a trajectory object and perform a PCA

10/12/21: For Dan's work let's make a basic PCA class that takes top & dcd{s}
and does the plots and stuff very simply!
"""
class myPCA:

    @staticmethod
    def conc_trajs(trajs,top):
        test_traj = trajs[0]
        if isinstance(test_traj,str):
            trajs = [md.load(t,top) for t in trajs]
        elif not isinstance(test_traj,md.Trajectory):
            print(f"Coords type: {type(test_traj)}"
                  f"Not understood")
        traj = md.join(trajs)
        return traj

    @staticmethod
    def strip_to_atoms(traj,selection=None):
        top = traj.top
        if selection is None:
            atoms = [a.index for a in top.atoms if 
                        a.name == "CA"]
        else:
            atoms = selection
        return traj.atom_slice(atoms)
        
    def __init__(self,
                 coords,top,out_hdf=None,atoms=None,**kwargs):
        """
        Create this class to hold the trajectories in it and perform a PCA.
        Takes one topology and either a list or single coordinate file/mdtraj
        object.
        Currently performs on just the C-α atoms.
        Feed PCA options in as kwargs.
        """
        self.coords = coords
        self.top = top
        self.pca = PCA(**kwargs)
        self.atoms = atoms
        self.data = None
        self.n_atoms = None
        self.n_frames = None
        self.transformed = None
        if out_hdf:
            self.out_path = out_hdf
        else:
            self.out_path = "./pca.h5"
        if isinstance(coords,str):
            self.n_coords = 1
            self.traj = md.load(coords,top=top) 
        elif isinstance(coords,list):
            self.n_coords = len(trajs)
            self.traj = conc_trajs(trajs,top)
        else:
            print("Feed Me a list of file paths or"
                  "a single trajectory files")
            raise TypeError
        self.preprocess()
        self._pca()
        self.save()
        self.plot_scores()
        
    def preprocess(self,atoms=None):
        """
        Before performing the PCA they need to be aligned and stripped to
        just the c-alphas!
        ( feed the atoms parameters to change the atoms used, should be
          iterable of atom indices)
        Currently this requires full RAM usage but possible to do without.
        """
        traj = self.traj
        traj = self.strip_to_atoms(traj,selection=atoms)
        traj = traj.superpose(traj,0)
        n_frames,n_atoms,_ = traj._xyz.shape
        flat_shape = (n_frames,n_atoms*3)
        self.traj = traj
        self.data = traj._xyz.reshape(flat_shape)
        self.n_atoms, self.n_frames = n_atoms, n_frames

    def _pca(self):
        data = self.data
        self.transformed = self.pca.fit_transform(data)
        self.n_components = self.transformed.shape[-1]

    def save(self):
        """
        Take a PCA object and save relevant info
        """
        f_p = self.out_path
        pca_items = self.pca.__dict__.items()
        with h5py.File(f_p,"w") as hdf:
            for k,v in pca_items:
                if v is not None:
                    # We save the components reshaped back
                    if k == "components_":
                        v = v.reshape(self.n_components,self.n_atoms,3)
                    hdf.create_dataset(k,data=v)
            hdf.create_dataset("data",data=self.data.reshape(-1,self.n_atoms,3))
            hdf.create_dataset("projections",data=self.transformed)

    def plot_scores(self,n_comps=4,sharex=True,sharey=True):
        """
        Plot scores.
        """
        colours = ["red","green","blue","yellow"]
        with h5py.File(self.out_path,"r") as hdf:
            comps = hdf["components_"]
            varis = hdf["explained_variance_"]
            comps = norm(comps,axis=-1)
            fig,axs = plt.subplots(n_comps,1,figsize=(8,12),
                                  sharex=sharex,sharey=sharey)
            for k in range(n_comps):
                if k > n_comps:
                    break
                axs[k].plot(varis[k]*comps[k],marker="o",c=colours[k],
                           label=f"PCA component {k}")
            axs[0].set_title("Score plots for PCA")
            if not os.path.isdir("./plots"):
                os.makedirs("./plots")
            plt.savefig("./plots/scores.png")
        
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
    pdb = "/mnt/hdd/work/braf-craf/actual_work/"\
          "code/braf-craf/work/trajectories/braf_mono/MyInIn.pdb"
    dcd = "/mnt/hdd/work/braf-craf/actual_work/"\
          "code/braf-craf/work/trajectories/braf_mono/MyInIn.dcd"
    test = myPCA(dcd,pdb) 

    #traj = Traj(dcd,pdb)
    #data = traj.load_all().xyz
    #print("Loaded trajectory of shape:")
    #print(data.shape)
    #print("Performing PCA...")
    #t1 = t()
    #cas = traj.get_calpha_index()
    #pca_obj = iPCA(traj,atom_sel=cas)
    #t2 = t()
    #print(f"PCA took {round(t2-t1,3)} seconds.")
    
