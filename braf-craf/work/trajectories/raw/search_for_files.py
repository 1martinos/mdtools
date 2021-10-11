import os
import mdtraj as md
from os.path import getsize

def extension_filter(file):
    wanted_extensions = ["dcd"]
    extension = file.rsplit(".",1)[-1]
    if extension in wanted_extensions:
        return True
    else:
        return False

def b2mb(size_in_bytes):
    return size_in_bytes * 10**(-6)

dir_path = "./mount_point/"
top =("/mnt/hdd/work/braf-craf/actual_work/code"
      "/braf-craf/work/trajectories/raw/mount_point"
      "/dfg_out/dfg_OUT-OUT/step3_pbcsetup.pdb")
total_size = 0
counter = 0
for root,folder,files in os.walk(dir_path):
    if files:
        filtered = [*filter(extension_filter,files)]
        if filtered:
            for x in filtered:
                full_path = os.path.join(root,x)
                size = getsize(full_path)
                print(full_path)
                print(b2mb(size))
                try:
                    md.load(full_path,top=top)
                    print("Works!")
                    total_size += b2mb(size)
                    counter += 1
                except:
                    print("Error!")
                print("\n")
print("END")
print(f"TOTAL SIZE : {total_size}")
print(f"FILES FOUND : {counter}")
