import numpy as np 
import pickle
from matplotlib import pyplot as plt
from galpy.util import bovy_coords as coords
import os
import sys

datapath="/mnt/home/bbhattarai/B3/"
save_datapath="/mnt/home/bbhattarai/resonance_sweeping/"
plotpath="/mnt/home/bbhattarai/resonance_sweeping/plots/"


argdex = int(sys.argv[1])
start  = int(argdex*42)-42
finish = int(argdex*42)


#start=0
#finish=337

a=list()
for i in range(start,finish):
    snapshot=i
    pfile = open(datapath+'step'+str(snapshot)+'.p', 'rb')
    idd,x,y,z,vx,vy,vz,mass=pickle.load(pfile)
    #Converting to cylindrical
    vr,vphi,vzz=coords.rect_to_cyl_vec(vx,vy,vz,x,y,z)
    r,phi,zz=coords.rect_to_cyl(x,y,z)
    #converting phi to degrees
    #phi=np.rad2deg(phi)
    
    #calculating bar_angle
    discindx=(mass<1e-7) # multiply by *2.324876e9 for B3-N
    barsample=(r>1)*(r<3)*discindx
    counts, bins, patches=plt.hist(phi[barsample],bins=360,range=[-np.pi,np.pi])
    ff=np.fft.fft(counts-np.mean(counts))
    barangle=-np.angle(ff[2])/2.
    
    barangle_degrees=np.rad2deg(barangle)
    a.append(barangle_degrees)
    

datafilename=str(start)+"_to_"+str(finish)+"_fft_barangles_sim_B3.ang"
bangle=np.array(a)
with open(save_datapath+datafilename, 'wb') as output:
        pickle.dump(bangle, output)
