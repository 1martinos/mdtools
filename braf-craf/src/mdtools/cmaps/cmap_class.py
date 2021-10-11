import numpy as np
import matplotlib.pyplot as plt
class ContactMap:
    """
    Construct from contact map constructor made hdf dataset
    """
    @staticmethod
    def n_fromCr(indices):
        """
        Inverting combinatorics is hard!
        Takes the indices attribute and calculates
        the end cmap shape!
        """
        uniques = set(indices[:,0])
        new_shape = (len(uniques)+1,len(uniques)+1)
        return new_shape

    def __init__(self,group):
        self.indices = group["indices"]
        self.dataset = group["cmaps"]
        self.n_frames, self.combs = self.dataset.shape
        self.shape = (self.n_frames, 
                      *self.n_fromCr(self.indices))

    
    def gen_cmap(self,index):
        arr = np.zeros((1,*self.shape[1:]))
        frame_data = self.dataset[index]
        for k,(i,j) in enumerate(self.indices):
            arr[0,i,j] = frame_data[k]
            arr[0,j,i] = frame_data[k] # is it faster to sym?
        return arr

    def gen_cmaps(self,indices):
        n_frames = len(indices)
        arr = np.zeros((n_frames,*self.shape[1:]))
        frame_data = self.dataset[indices]
        for k,(i,j) in enumerate(self.indices):
            arr[:,i,j] = frame_data[:,k]
            arr[:,j,i] = frame_data[:,k] # is it faster to sym?
        return arr

    def all(self):
        indices = [*range(self.n_frames)]
        return self[indices]

    # Class attrs
    def __getitem__(self,item):
        if isinstance(item,int) or len(item) == 1:
            return self.gen_cmap(item)
        else:
            cmaps = self.gen_cmaps(item)
            return cmaps

    def __iter__(self):
        frame = 0
        while frame < self.n_frames:
            yield self[frame]
            frame += 1
        raise StopIteration

        


