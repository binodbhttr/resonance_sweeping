import numpy as np 
import pickle
from matplotlib import pyplot as plt
from galpy.util import bovy_coords as coords

print("Loading Data...")
datapath="/mnt/home/bbhattarai/B3/"
i=336

pfile = open(datapath+'step'+str(i)+'.p', 'rb')
idd,x,y,z,vx,vy,vz,mass=pickle.load(pfile)


R=(x**2+y**2)**(1/2)
vR = (x*vx + y*vy ) / R
vPHI = ( x*vy - y*vx ) / R

print(vPHI)
print(np.max(vPHI))

print(x)
#plt.hexbin(x,y,extent=((-30,30,-30,30)),gridsize=1000,bins="log",mincnt=1)
#plt.savefig("test.jpg")
#plt.show()


vr,vphi,vzz=coords.rect_to_cyl_vec(vx,vy,vz,x,y,z)
r,phi,zz=coords.rect_to_cyl(x,y,z)


keep=(x<9)*(x>7)*(y>-1)*(y<1)

fig1=plt.figure()
ax=fig1.add_subplot(111)
ax.hexbin(vr[keep],vphi[keep],extent=((-200,200,50,300)),gridsize=1000,bins="log",mincnt=1)
ax.set_xlabel("vr")
ax.set_ylabel("vphi")
fig1.savefig("v_phi_vs_v_r_selected_region.jpg",bbox_inches="tight")
#fig1.closefig()

keep_phi=(phi<5)*(phi>-5)

fig2=plt.figure()
ax=fig2.add_subplot(111)
cb1=ax.hexbin(r[keep_phi],vphi[keep_phi],extent=((5,15,50,300)),gridsize=1000,mincnt=1)
ax.set_xlabel("r")
ax.set_ylabel("vphi")
cbar_ax = fig2.add_axes([1, 0.14, 0.03, 0.78]) # position of the colorbar (left, bottom, width, height)
fig2.colorbar(cb1, cax=cbar_ax)
cbar_ax.set_ylabel('log_n_star')
cbar_ax.yaxis.label.set_size(10)
fig2.savefig("v_phi_vs_r_hexbin.jpg",bbox_inches="tight")


plotpath="./plots/"

for i in range(5,21):
    keep=(x<i+1)*(x>i)*(y>-1)*(y<1)
    fig3=plt.figure()
    ax=fig3.add_subplot(111)
    ax.hexbin(vr[keep],vphi[keep],extent=((-200,200,0,300)),gridsize=500,bins="log",mincnt=1)
    ax.set_xlabel("vr")
    ax.set_ylabel("vphi")
    plotname="new_v_phi_vs_v_r_selected_region_"+str(i)+"_to_"+str(i+1)+".jpg"
    fig3.savefig(plotpath+plotname,bbox_inches="tight")
    print("\n Saved the image to file...: ",plotname)
    plt.close(fig3)