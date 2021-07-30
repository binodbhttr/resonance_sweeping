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
    phi=np.rad2deg(phi)
    
    #calculating bar_angle
    discindx=(mass<1e-7)
    barsample=(r>1)*(r<3)*discindx
    counts, bins, patches=plt.hist(phi[barsample],bins=360,histtype='step')
    bin_centres=bins[:-1]+(bins[1]-bins[0])/2
    max_indx=np.argmax(counts)
    barangle_degrees=bin_centres[max_indx]
    a.append(barangle_degrees)
    

datafilename=str(start)+"_to_"+str(finish)+"saved_barangles_sim_B4.ang"
bangle=np.array(a)
with open(save_datapath+datafilename, 'wb') as output:
        pickle.dump(bangle, output)
