import numpy as np 
import pickle
from matplotlib import pyplot as plt
from galpy.util import bovy_coords as coords
import os
import sys
from procedure import *

save_datapath="/mnt/home/bbhattarai/resonance_sweeping/New_Sims_Analysis/"
plotpath="/mnt/home/bbhattarai/resonance_sweeping/New_Sims_Analysis/plots/"
#local_datapath="./resonance_sweeping/localdata/"

#data={}
######
######
#This part of the code captures number from the sbatch script to run things parallel

argdex = int(sys.argv[1])
start  = int(argdex*131)-131
finish = int(argdex*131)


#start=0
#finish=337

a=list()
for i in range(start,finish):
    snapshot=i
    snaparr = loadwholesnap(path,snapshot)
    idd=snaparr['idd']
    x=snaparr['x']
    y=snaparr['y']
    z=snaparr['z']
    vx=snaparr['vx']
    vy=snaparr['vy']
    vz=snaparr['vz']  
    mass=snaparr['mass']  #note mass here is in solar mass (use the factor 2.324876e9)
    
    vr=snaparr['vr']
    vphi=snaparr['vphi']
    vzz=snaparr['vzz']
    r=snaparr['r']
    phi=snaparr['phi']
    zz=snaparr['zz']
    
    
    #calculating bar_angle
    discindx=(mass<1e-7*2.324876e9)
    barsample=(r>1)*(r<3)*discindx
    counts, bins, patches=plt.hist(phi[barsample],bins=360,range=[-np.pi,np.pi])
    ff=np.fft.fft(counts-np.mean(counts))
    barangle=-np.angle(ff[2])/2.
    
    barangle_degrees=np.rad2deg(barangle)
    a.append(barangle_degrees)
    

datafilename=str(start)+"_to_"+str(finish)+"_fft_barangles_sim_B3-N.ang"
bangle=np.array(a)
with open(save_datapath+datafilename, 'wb') as output:
        pickle.dump(bangle, output)