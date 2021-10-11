from mdtools.utils import *
import mdtraj as md
def make_mono(coord,topo,index=0):
    traj = md.load(crd,top=topo)
    atoms = []
    for res in traj.top._chains[0]._residues:
        for atom in res.atoms:
            atoms.append(atom.index)
    return traj.atom_slice(atoms)
            
tops,crds = get_traj_files()
for crd,top in zip(crds,tops):
    traj = make_mono(crd,top)
    traj.save(get_name(top) + ".dcd")
    traj[0].save(get_name(top) + ".pdb")
