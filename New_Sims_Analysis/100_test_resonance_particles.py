import numpy as np 
import pickle
from matplotlib import pyplot as plt
from galpy.util import bovy_coords as coords
import os
import sys
from procedure import *



datapath="/mnt/home/bbhattarai/B3-N/"
freq_datapath="/mnt/home/bbhattarai/ceph/freq_data/"
plotpath="/mnt/home/bbhattarai/resonance_sweeping/New_Sims_Analysis/plots/"


snapshot=1000

freq_datafile="DiskActions"+str(snapshot)+"_B3N.npy"


freqs= np.load(freq_datapath+freq_datafile)


Jrdisk=freqs[0]
Jphidisk=freqs[1]
Jzdisk=freqs[2]
Trdisk=freqs[3]
Tphidisk=freqs[4]
Tzdisk=freqs[5]
Ordisk=freqs[6]
Ophidisk=freqs[7]
Ozdisk=freqs[8]
idd_from_freqs=freqs[9]


snapshot=1000
snaparr = loadwholesnap(path,snapshot)
print("These are the data we have",snaparr[0].dtype)
idd=snaparr['idd']
x=snaparr['x']
y=snaparr['y']
z=snaparr['z']
vx=snaparr['vx']
vy=snaparr['vy']
vz=snaparr['vz']  
mass=snaparr['mass']  #mass is in solar mass (change old mass calculations to take account of the factor 2.324876e9)

vr=snaparr['vr']
vphi=snaparr['vphi']
vzz=snaparr['vzz']
r=snaparr['r']
phi=snaparr['phi'] #phi is in radians
zz=snaparr['zz']

#converting phi to degrees
phi_degrees=np.rad2deg(phi)


discindx=(mass<1e-7*2.324876e9)
idd_snapshots=idd[discindx]
x_select=x[discindx]
y_select=y[discindx]
z_select=z[discindx]

vx_select=vx[discindx]
vy_select=vy[discindx]
vz_select=vz[discindx]


vr_select=vr[discindx]
vphi_select=vphi[discindx]
vzz_select=vzz[discindx]


r_select=r[discindx]
phi_select=phi_degrees[discindx]


angle_datapath="/mnt/home/bbhattarai/resonance_sweeping//New_Sims_Analysis/"
datafilename="0_to_1048_B3-N_fft_barangles_combined_degrees.ang"
ang_stored = open(angle_datapath+datafilename,'rb')
all_bangles=pickle.load(ang_stored)
print(len(all_bangles))


#datafilename="saved_bar_pattern_speed_km_per_s_kpc.ang"
datafilename="B3-N_saved_bar_pattern_speed_km_per_s_kpc.ang"
save_datapath="/mnt/home/bbhattarai/resonance_sweeping//New_Sims_Analysis/"
ps_stored = open(save_datapath+datafilename,'rb')
ps=pickle.load(ps_stored)
#print(ps)
ps_located=ps[snapshot]
print(ps_located)


start=snapshot-3
end=snapshot+3
s=0
c=0
for i in range(start,end):
    c+=1
    s=s+ps[i]
print(c)
ps_adjusted=s/c

print(ps_adjusted)

#factor=3.08567758/3.15576
#ps_adjusted=ps_adjusted/factor
print(ps_adjusted)



#OLR resonance
#olr_resonance=Ophidisk-Ordisk/2 # inner Linbald resonance

olr_resonance=Ophidisk+Ordisk/2 # outer Linbald resonance
omega_diff_olr=olr_resonance-ps_adjusted
keep_olr=(omega_diff_olr<0.01)*(omega_diff_olr>-0.1)


#CR resonance
omega_diff=Ophidisk-ps_adjusted
keep_cr=(omega_diff<0.01)*(omega_diff>-0.1)



x_resonance_cr=(x_select[keep_cr])
y_resonance_cr=(y_select[keep_cr])
z_resonance_cr=(z_select[keep_cr])
vr_resonance_cr=(vr_select[keep_cr])
vphi_resonance_cr=(vphi_select[keep_cr])
vzz_resonance_cr=(vzz_select[keep_cr])

print(len(x_resonance_cr))

r_resonance_cr=r_select[keep_cr]
phi_resonance_cr=phi_select[keep_cr]


idd_resonance_cr_1000=idd_snapshots[keep_cr]

datafilename="700idd_resonance_cr.pkl"

idd_stored = open(save_datapath+datafilename,'rb')
idd_resonance_cr_700=pickle.load(idd_stored)


commons_cr=list()
for i in range(len(idd_resonance_cr_700)):
    c=np.where(idd_resonance_cr_1000==idd_resonance_cr_700[i])
    commons_cr.extend(list(idd_resonance_cr_1000[c]))
print(commons_cr)
