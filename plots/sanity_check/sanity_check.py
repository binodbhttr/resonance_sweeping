import numpy as np 
import pickle
from matplotlib import pyplot as plt
from galpy.util import bovy_coords as coords
import os
import sys

datapath="/mnt/home/bbhattarai/B3/"
save_datapath="/mnt/home/bbhattarai/resonance_sweeping/"
plotpath="/mnt/home/bbhattarai/resonance_sweeping/plots/sanity_check/"
#local_datapath="./resonance_sweeping/localdata/"
#data={}
######
######
#This part of the code captures number from the sbatch script to run things parallel
#argdex = int(sys.argv[1])
#start  = int(argdex*42)-42
#finish = int(argdex*42)

start=300
finish=301
a=list()
#total 337 snapshots starting from 0 to 336
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
    barsample=(r<3)*discindx
    counts, bins, patches=plt.hist(phi[barsample],bins=360,histtype='step')
    
    bin_centres=bins[:-1]+(bins[1]-bins[0])/2
    max_indx=np.argmax(counts)
    barangle_degrees=bin_centres[max_indx]
    a.append(barangle_degrees)
    #yfa=np.fft.fft(counts)
    #mode2=yfa[2]/yfa[0]
    #barangle=np.angle(yfa[2])
    #barangle_degrees=np.rad2deg(barangle)

    # we always want the bar to be at 25 degrees from the observer
    rotation_angle=25-barangle_degrees
    rot_angle_radians=np.deg2rad(rotation_angle)
    x_rot=x*np.cos(rot_angle_radians)-y*np.sin(rot_angle_radians) #take care of radians and degress
    y_rot=x*np.sin(rot_angle_radians)+y*np.cos(rot_angle_radians)
    R_rot=(x_rot**2+y_rot**2)**(1/2)
    phi_rot=np.arctan2(y_rot,x_rot) #this is different from the original phi
    #use arctan2 to preserve the quadrant
    phi_rot=np.rad2deg(phi_rot)

    keep_phi=(phi_rot<5)*(phi_rot>-5)

    #r_export=r[keep_phi]
    #vphi_export=vphi[keep_phi]

    fig2=plt.figure()
    ax=fig2.add_subplot(111)
    ax.text(8,300,r"Snapshot %s, Slice: -5<$\phi$<5"%(str(snapshot)))
    cb1=ax.hexbin(r[keep_phi],vphi[keep_phi],extent=((5,15,50,300)),gridsize=500,mincnt=1,vmin=1,vmax=80)
    ax.set_xlabel("R")
    ax.set_ylabel(r"v$\phi$")
    cbar_ax = fig2.add_axes([0.91, 0.12, 0.03, 0.76]) # position of the colorbar (left, bottom, width, height)
    fig2.colorbar(cb1, cax=cbar_ax)
    cbar_ax.set_ylabel(r'n$_{star}$')
    cbar_ax.yaxis.label.set_size(10)
    plotname=str(snapshot)+"_v_phi_vs_r_hexbin_rotated.jpg"
    fig2.savefig(plotpath+plotname,bbox_inches="tight")
    print("Plot generated and saved to file: ",plotname)
    #data={"snapshot":i,"r":r_export,"vphi":vphi_export}
    #filename="snapshot_"+str(i)+".pkl"
    
    '''
    #figure3: x vs y plot of the entire galaxy
    fig3=plt.figure()
    ax=fig3.add_subplot(111)
    ax.text(0,34,r"Snapshot %s "%(str(snapshot)))
    cb1=ax.hexbin(x_rot,y_rot,extent=((-35,35,-35,35)),gridsize=500,mincnt=1,bins="log")
    ax.set_xlabel("x (rotated) Kpc")
    ax.set_ylabel("y (rotated) Kpc")
    cbar_ax = fig3.add_axes([0.91, 0.12, 0.03, 0.76]) # position of the colorbar (left, bottom, width, height)
    fig3.colorbar(cb1, cax=cbar_ax)
    cbar_ax.set_ylabel(r'n$_{star}$')
    cbar_ax.yaxis.label.set_size(10)
    plotname=str(snapshot)+"x_rot_vs_y_rot_all.jpg"
    fig3.savefig(plotpath+plotname,bbox_inches="tight")
    print("Plot generated and saved to file: ",plotname)
    '''
    
    #figure4: x_rot_select=x_rot[keep_phi], y_rot_select=y_rot[keep_phi] (x and y of the slected region)
    fig4=plt.figure()
    ax4=fig4.add_subplot(111)
    ax4.text(5,3,r"Snapshot %s, Slice: -5<$\phi$<5"%(str(snapshot)))
    #print("These are the rotated x and y ",x_rot[keep_phi])
    #ax.plot(x,y)
    x_rot_select=x_rot[keep_phi]
    y_rot_select=y_rot[keep_phi]

    cb4=ax4.hexbin(x_rot_select,y_rot_select,extent=((-1,35,-3,3)),gridsize=500,mincnt=1,bins="log")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    cbar_ax4 = fig4.add_axes([0.91, 0.12, 0.03, 0.76]) # position of the colorbar (left, bottom, width, height)
    fig4.colorbar(cb4, cax=cbar_ax4)
    cbar_ax4.set_ylabel(r'n$_{star}$')
    cbar_ax4.yaxis.label.set_size(10)
    plotname=str(snapshot)+"x_rot_vs_y_rot_selected_region.jpg"
    fig4.savefig(plotpath+plotname,bbox_inches="tight")
    print("Plot generated and saved to file: ",plotname)
    
    
    
    fig5=plt.figure()
    ax5=fig5.add_subplot(111)
    keep=(x_rot<8.5)*(x_rot>7.5)*(y_rot>-0.5)*(y_rot<0.5)*(z>-0.5)*(z<0.5)
    vphi_select=vphi[keep]
    vr_select=vr[keep]
    ax5.text(8,300,r"Snapshot %s "%(str(snapshot)))
    cb5=ax5.hexbin(vr_select,vphi_select,extent=((-200,200,50,300)),gridsize=500,mincnt=1,vmin=1,vmax=30)
    ax5.set_xlabel("vR")
    ax5.set_ylabel(r"v$\phi$")
    cbar_ax5 = fig5.add_axes([0.91, 0.12, 0.03, 0.76]) # position of the colorbar (left, bottom, width, height)
    fig5.colorbar(cb5, cax=cbar_ax5)
    cbar_ax5.set_ylabel(r'n$_{star}$')
    cbar_ax5.yaxis.label.set_size(10)
    plotname=str(snapshot)+"_v_phi_vs_vr_hexbin_rotated.jpg"
    fig5.savefig(plotpath+plotname,bbox_inches="tight")
    print("Plot generated and saved to file: ",plotname)

datafilename="saved_barangles.ang"
bangle=np.array(a)
with open(save_datapath+datafilename, 'wb') as output:
        pickle.dump(bangle, output)
