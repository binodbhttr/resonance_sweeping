#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:35:34 2021

@author: rlmcclure/jhunt/cfilion

moving to forsims.py class
"""
#%%
path = '../jhunt/scratch/Bonsai/r3/B3-N/'
ncores = 32
start = 0
finish = None
savestr=''


infodtype = [('time','d'),('n','i'),('ndim','i'),('ng','i'),('nd','i'),('ns','i'),('on','i')]
stellardtype = [('idd', '<u8'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4'), ('vr', '<f4'), ('vphi', '<f4'), ('vzz', '<f4'), ('r', '<f4'), ('phi', '<f4'), ('zz', '<f4'), ('mass', '<f4')]
dmdtype = [('mass','f'), ('x','f'), ('y','f'), ('z','f'), ('vx','f'), ('vy','f'), ('vz','f'), ('ID','Q')]

#%%annulus params
#%%
# from forsims import *
#%%
#python 3.8
import datetime
import numpy as np
from galpy.util import coords
import matplotlib.pyplot as plt
import matplotlib
import pickle
import pandas as pd
from mpl_toolkits import mplot3d
print('start at: '+str(datetime.datetime.now()),flush=True)
#%%
def printnow():
    print('check: '+str(datetime.datetime.now()),flush=True)
#%% loaders

#% load one snapshot
def loader(filename,wdm=0,verbose=0):
    """
    loads an individual time snapshot from an individual node 
    converts to physical units

    Inputs
    ------------------
    filename (str): path to file of snapshot
    wdm (boolean): with dark matter toggle, will get stars and dark matter

    Returns
    ------------------
    cats (np.ndarray): stars [mass value for disk or bluge (float32), x position(float32),
                            y position (float32), z position (float32), vx x velocity(float32),
                            vy y velocity(float32), vz z velocity(float32), metals (float32),
                            tform (float), ID (uint64)]
    if wdm==True: 
        returns tuple with (catd,cats,info) where catd (np.ndarray): dark matter particles [same as above]
    """
    with open(filename, 'rb') as f:
        if wdm == False:
            if verbose>1:
                print(filename)
            #file info
            info= np.fromfile(f,dtype=[('time','d'),('n','i'),('ndim','i'),
                                    ('ng','i'),('nd','i'),('ns','i'),
                                    ('on','i')],count=1)
            infoBytes = f.tell()
            if verbose>2:
                print(infoBytes)
            #skip darkmatter
            #read the first dm line
            if verbose>2:
                print(f.tell())
            catd = np.fromfile(f,dtype=[('mass','f'), ('x','f'), ('y','f'), 
                                    ('z','f'), ('vx','f'), ('vy','f'), 
                                    ('vz','f'), ('ID',np.int64)],
                            count=1)   
            #get the bytes location and subtract off the bytes location after loading info to get n bytes a line for dm
            if verbose>2:
                print(f.tell())
            current = f.tell()
            dmBytes = current-infoBytes
            f.seek(dmBytes*(info['nd'][0]-1)+current)
            if verbose>2:
                print(f.tell())
            # stars setup                    
            cats= np.fromfile(f,dtype=[('mass','f'), ('x','f'), ('y','f'), 
                                    ('z','f'), ('vx','f'), ('vy','f'), 
                                    ('vz','f'), ('metals','f'), 
                                    ('tform','f'), ('ID',np.int64)],
                            count=info['ns'][0])
            if verbose>2:
                print('done')
        else:
            if verbose>1:
                print(filename)
            #file info
            info= np.fromfile(f,dtype=[('time','d'),('n','i'),('ndim','i'),
                                    ('ng','i'),('nd','i'),('ns','i'),
                                    ('on','i')],count=1)
            if verbose>2:
                print(f.tell())
            # #dark matter setup count is reading the number of ?rows? 
            catd= np.fromfile(f,dtype=[('mass','f'), ('x','f'), ('y','f'), 
                                    ('z','f'), ('vx','f'), ('vy','f'), 
                                    ('vz','f'), ('ID',np.int64)],
                            count=info['nd'][0]) 
            if verbose>2:
                print(f.tell())                   
            # stars setup                    
            cats= np.fromfile(f,dtype=[('mass','f'), ('x','f'), ('y','f'), 
                                    ('z','f'), ('vx','f'), ('vy','f'), 
                                    ('vz','f'), ('metals','f'), 
                                    ('tform','f'), ('ID',np.int64) ],
                            count=info['ns'][0])
            if verbose>2:
                print('done')
        
    
    #convert to physical units as found in README.md
    if wdm == True:
        catd['mass']*=2.324876e9
        catd['vx']*=100.
        catd['vy']*=100.
        catd['vz']*=100.
    cats['mass']*=2.324876e9
    cats['vx']*=100.
    cats['vy']*=100.
    cats['vz']*=100.
    
    if wdm == True:
        return(catd,cats,info)
    else:
        return(cats,info)

#% load one whole timestep
def loadwholesnap(path,timestepid,ncores=32,ntot=1279999360,ndark=1003921280,nstar=276078080,wdmBool=False,verbose=0):

    #load the times file
    times=np.genfromtxt(path+'times.txt',dtype='str')

    #if the number of total, dark, and stellar particles are not set then determine
    if ntot==None:
        print('Loading to find nstars, ndark, and ntot'+path+times[timestepid], flush=True)
        if wdmBool == True:
            _,cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose)
        else:
            cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose)
        ntot=info['n']
        ndark=info['nd']
        nstar=info['ns']
        for j in range(1,ncores):
            print('Loading '+path+times[timestepid][:-1]+str(j), flush=True)
            if wdmBool == True:
                _,cats,info=loader(path+times[timestepid][:-1]+str(j),wdm=wdmBool,verbose=verbose)
            else:
                cats,info=loader(path+times[timestepid][:-1]+str(j),wdm=wdmBool,verbose=verbose)
            ntot=ntot+info['n']
            ndark=ndark+info['nd']
            nstar=nstar+info['ns']
            print('in length maker',ntot,ndark,nstar,info['n'],info['nd'],info['ns'],flush=True)
    if verbose > 1:
        print('\nLoading for info '+path+times[timestepid], flush=True)
    if wdmBool == True:
        _,cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose)
    else:
        cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose)

    if verbose > 1:
        print(info['n'],info['ns'],info['nd'], flush=True)
    tnstar=int(info['ns'])

    if verbose > 2:
        print(ntot,ndark,nstar, flush=True) 
    snaparr = np.empty(nstar,dtype=[('t','d'), ('idd','Q'),('x','f'), ('y','f'), ('z','f'), 
                                   ('vx','f'), ('vy','f'), ('vz','f'), 
                                   ('vr','f'), ('vphi','f'), ('vzz','f'), 
                                   ('r','f'), ('phi','f'), ('zz','f'), ('mass','f')])        

    if verbose > 1:
        print('Loading '+path+times[timestepid], flush=True)
    if wdmBool == True:
        _,cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose)
    else:
        cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose)

    if verbose > 2:
        print(info['n'],info['ns'],info['nd'], flush=True)
    tnstar=int(info['ns'])
    # tndark=int(info['nd'])

    # timefw=info['time']*9.778145/1000.
    snaparr['x'][0:tnstar]=cats['x']
    snaparr['y'][0:tnstar]=cats['y']
    snaparr['z'][0:tnstar]=cats['z']
    snaparr['vx'][0:tnstar]=cats['vx']
    snaparr['vy'][0:tnstar]=cats['vy']
    snaparr['vz'][0:tnstar]=cats['vz']
    snaparr['mass'][0:tnstar]=cats['mass']
    snaparr['idd'][0:tnstar]=cats['ID']
    snaparr['t'][0:tnstar]=info['time']*9.778145/1000.
    
    arrayindx=tnstar

    if verbose > 2:
        print(arrayindx, flush=True)
    for j in range(1,ncores):
        if verbose > 1:
            print('Loading '+path+times[timestepid][:-1]+str(j), flush=True)
        if wdmBool == True:
            _,cats,info=loader(path+times[timestepid][:-1]+str(j),wdm=wdmBool,verbose=verbose)
        else:
            cats,info=loader(path+times[timestepid][:-1]+str(j),wdm=wdmBool,verbose=verbose)
        tnstar=int(info['ns'])
        if verbose > 2:
            print(arrayindx,tnstar,arrayindx+tnstar, flush=True)
        snaparr['x'][arrayindx:arrayindx+tnstar]=cats['x']
        snaparr['y'][arrayindx:arrayindx+tnstar]=cats['y']
        snaparr['z'][arrayindx:arrayindx+tnstar]=cats['z']
        snaparr['vx'][arrayindx:arrayindx+tnstar]=cats['vx']
        snaparr['vy'][arrayindx:arrayindx+tnstar]=cats['vy']
        snaparr['vz'][arrayindx:arrayindx+tnstar]=cats['vz']
        snaparr['idd'][arrayindx:arrayindx+tnstar]=cats['ID']
        snaparr['mass'][arrayindx:arrayindx+tnstar]=cats['mass']
        snaparr['t'][arrayindx:arrayindx+tnstar]=info['time']*9.778145/1000.

        arrayindx=arrayindx+tnstar
        #except:
        #    print(times[i]+'-'+str(j),'has no star particles', flush=True)

    snaparr['vr'],snaparr['vphi'],snaparr['vzz']=coords.rect_to_cyl_vec(snaparr['vx'],snaparr['vy'],snaparr['vz'],snaparr['x'],snaparr['y'],snaparr['z'])
    snaparr['r'],snaparr['phi'],snaparr['zz']=coords.rect_to_cyl(snaparr['x'],snaparr['y'],snaparr['z'])

    return(snaparr)

#% load whole pickle step
def loadstep(path,stepid):
    fstr = path+'step'+str(stepid)+'.p'
    
    #load the pickle
    with open(fstr,'rb') as f:
        idd,x,y,z,vx,vy,vz,mass = pickle.load(f)
    
    return (idd,x,y,z,vx,vy,vz,mass)

#% load one particle over time
def loadonesource(path,idx,ncores=32,start=0,finish=None,wdmBool=False,verbose=0):
    '''
    takes a path to the folder containing snapshots and an id
    returns an array [time,x,y,z,vx,vy,vz,mass,vr,vphi,vzz,r,phi,zz]
        dtype=[('t','f'),('x','f'), ('y','f'), ('z','f'), 
            ('vx','f'), ('vy','f'), ('vz','f'), 
            ('vr','f'), ('vphi','f'), ('vzz','f'), 
            ('r','f'), ('phi','f'), ('zz','f'), ('mass','f'), ('idd','Q')] 

    '''

    idx = np.uint64(idx)
    #load the times file
    times=np.genfromtxt(path+'times.txt',dtype='str')

    #set other params
    if finish == None:
        finish = len(times)

    # let's allocate space for the finished final array for one source
    tlen = len(range(start,finish))

    sourcearr=np.empty(tlen,dtype=[('t','d'),('x','f'), ('y','f'), ('z','f'), 
                                   ('vx','f'), ('vy','f'), ('vz','f'), 
                                   ('vr','f'), ('vphi','f'), ('vzz','f'), 
                                   ('r','f'), ('phi','f'), ('zz','f'), ('mass','f'), ('idd','Q')] )

    indvarrayindx=0
    lastcore = 0
    
    for i in range(start,finish):
        keepgoing = 1
        if keepgoing > 0:
            if verbose > 1:
                print('Loading starting from last core first '+path+times[i][:-1]+str(lastcore), flush=True)
            if wdmBool == True:
                _,cats,info=loader(path+times[i][:-1]+str(lastcore),wdm=wdmBool,verbose=verbose)
            else:
                cats,info=loader(path+times[i][:-1]+str(lastcore),wdm=wdmBool,verbose=verbose)
            if np.where(np.isin(cats['ID'],idx,assume_unique=1))[0].size > 0:
                j = lastcore
                if verbose > 0:
                    print('Source found in '+path+times[i][:-1]+str(j), flush=True)
                lastcore = j
                if verbose > 1:
                    print('Last core is ',j)
                foundid = np.argwhere(idx == cats['ID']).flatten()
                if verbose >1:
                    print('printing cats[foundid]')
                    print(cats[foundid])
                
                sourcearr['x'][indvarrayindx]=cats['x'][foundid]
                sourcearr['y'][indvarrayindx]=cats['y'][foundid]
                sourcearr['z'][indvarrayindx]=cats['z'][foundid]
                sourcearr['vx'][indvarrayindx]=cats['vx'][foundid]
                sourcearr['vy'][indvarrayindx]=cats['vy'][foundid]
                sourcearr['vz'][indvarrayindx]=cats['vz'][foundid]
                sourcearr['mass'][indvarrayindx]=cats['mass'][foundid]
                sourcearr['idd'][indvarrayindx]=cats['ID'][foundid]
                

                sourcearr['vr'][indvarrayindx],sourcearr['vphi'][indvarrayindx],sourcearr['vzz'][indvarrayindx]=coords.rect_to_cyl_vec(cats['vx'][foundid],cats['vy'][foundid],cats['vz'][foundid],cats['x'][foundid],cats['y'][foundid],cats['z'][foundid])
                sourcearr['r'][indvarrayindx],sourcearr['phi'][indvarrayindx],sourcearr['zz'][indvarrayindx]=coords.rect_to_cyl(cats['x'][foundid],cats['y'][foundid],cats['z'][foundid])
                sourcearr['t'][indvarrayindx]=info['time']*9.778145/1000.
                
                keepgoing = 0

        if keepgoing > 0:
            for j in range(0,ncores):
                if keepgoing > 0:
                    if verbose > 1:
                        print('Loading '+path+times[i][:-1]+str(j), flush=True)
                    if wdmBool == True:
                        _,cats,info=loader(path+times[i][:-1]+str(j),wdm=wdmBool,verbose=verbose)
                    else:
                        cats,info=loader(path+times[i][:-1]+str(j),wdm=wdmBool,verbose=verbose)
                    if np.where(np.isin(cats['ID'],idx,assume_unique=1))[0].size > 0:
                        if verbose > 0:
                            print('Source found in '+path+times[i][:-1]+str(j), flush=True)
                        lastcore = j
                        if verbose > 1:
                            print('Last core is ',j)
                        foundid = np.argwhere(idx == cats['ID']).flatten()
                        
                        if verbose >1:
                            print('printing cats[\'x\'][foundid]')
                            print(cats['x'][foundid])

                        if verbose >1:
                            print('printing cats[\'ID\'][foundid]')
                            print(cats['ID'][foundid])
                        sourcearr['x'][indvarrayindx]=cats['x'][foundid]
                        sourcearr['y'][indvarrayindx]=cats['y'][foundid]
                        sourcearr['z'][indvarrayindx]=cats['z'][foundid]
                        sourcearr['vx'][indvarrayindx]=cats['vx'][foundid]
                        sourcearr['vy'][indvarrayindx]=cats['vy'][foundid]
                        sourcearr['vz'][indvarrayindx]=cats['vz'][foundid]
                        sourcearr['mass'][indvarrayindx]=cats['mass'][foundid]
                        sourcearr['idd'][indvarrayindx]=cats['ID'][foundid]
                        

                        sourcearr['vr'][indvarrayindx],sourcearr['vphi'][indvarrayindx],sourcearr['vzz'][indvarrayindx]=coords.rect_to_cyl_vec(cats['vx'][foundid],cats['vy'][foundid],cats['vz'][foundid],cats['x'][foundid],cats['y'][foundid],cats['z'][foundid])
                        sourcearr['r'][indvarrayindx],sourcearr['phi'][indvarrayindx],sourcearr['zz'][indvarrayindx]=coords.rect_to_cyl(cats['x'][foundid],cats['y'][foundid],cats['z'][foundid])
                        sourcearr['t'][indvarrayindx]=info['time']*9.778145/1000.
                        
                        keepgoing = 0
        indvarrayindx+=1
        #error out if cannot find the ID
        if keepgoing > 0:
            print('ID NOT FOUND in '+path+times[i][:-1])
            break
    return(sourcearr)


#% plotting
def doplots(path=None,idstr=None,inarr=None,mode='numpy',save=0,savestr='',lims=[-30,30],tmin=0,tmax=110,mks=20,toffset=None):
    if mode == 'pandas':
        fl = pd.read_csv(path+'particle_'+str(idstr)+'.txt',comment='#',sep=',')
    if mode == 'numpy':
        fl  = np.loadtxt(path+'particle_'+str(idstr)+'.txt',dtype=[('t','f'),('x','f'), ('y','f'), ('z','f'), ('vx','f'), ('vy','f'), ('vz','f'), ('vr','f'), ('vphi','f'), ('vzz','f'), ('r','f'), ('phi','f'), ('zz','f'), ('mass','f')] )
    if mode == 'array':
        fl = inarr
    if save == 1:
        # matplotlib.use('pdf')
        matplotlib.use('agg')


    if tmax is None:
        tmax = len(fl)

    f = plt.figure(figsize=(8,6))   
    ax = f.add_subplot(111, projection='3d')
    ax.plot3D(fl['x'][tmin:tmax],fl['y'][tmin:tmax],fl['z'][tmin:tmax],c='grey',lw=1)
    p = ax.scatter3D(fl['x'][tmin:tmax],fl['y'][tmin:tmax],fl['z'][tmin:tmax],c=fl['t'][tmin:tmax],s=mks)

    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_zlabel('z [kpc]')
    ax.set_xlim(lims[0],lims[1])
    ax.set_ylim(lims[0],lims[1])
    ax.set_zlim(lims[0],lims[1])
    # ax.view_init(45, 120)
    plt.title('particle '+str(idstr))
    f.colorbar(p,label='Myr')
    if save == 1:
        plt.savefig('plots/%s%s_3d.png' % (str(idstr),savestr),bbox_inches='tight',dpi=500)
        plt.close()
    else:
        plt.show()

    f = plt.figure(figsize=(6,6))
    plt.plot(fl['y'][tmin:tmax],fl['z'][tmin:tmax],c='grey',lw=1)
    p = plt.scatter(fl['y'][tmin:tmax],fl['z'][tmin:tmax],c=fl['t'][tmin:tmax],s=mks)
    plt.xlabel('y [kpc]')
    plt.xlim(lims[0],lims[1])
    plt.ylabel('z [kpc]')
    plt.ylim(lims[0],lims[1])
    plt.title('particle '+str(idstr))
    f.colorbar(p,label='Myr')
    if save == 1:
        plt.savefig('plots/%s%s_yz.png' % (str(idstr),savestr),bbox_inches='tight',dpi=500)
        plt.close()
    else:
        plt.show()


    f = plt.figure(figsize=(6,6))
    plt.plot(fl['x'][tmin:tmax],fl['y'][tmin:tmax],c='grey',lw=1)
    p = plt.scatter(fl['x'][tmin:tmax],fl['y'][tmin:tmax],c=fl['t'][tmin:tmax],s=mks)
    plt.xlabel('x [kpc]')
    plt.xlim(lims[0],lims[1])
    plt.ylabel('y [kpc]')
    plt.ylim(lims[0],lims[1])
    plt.title('particle '+str(idstr))
    f.colorbar(p,label='Myr')
    if save == 1:
        plt.savefig('plots/%s%s_xy.png' % (str(idstr),savestr),bbox_inches='tight',dpi=500)
        plt.close()
    else:
        plt.show()

# %%
print('end at: '+str(datetime.datetime.now()),flush=True)
