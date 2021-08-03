#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:35:34 2021

@author: rlmcclure/jhunt/cfilion

moving to forsims.py class
"""
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
from matplotlib.gridspec import GridSpec
import os
import glob
#%%
path = '/mnt/home/jhunt/scratch/Bonsai/r3/B3-N/' #path to simulation snaps
pointfpath = '/mnt/sdceph/users/rmcclure/pointerfiles/' #path for pointer .npy files
# xbarpartspath = os.getcwd()+'/ceph/xbarparticles/' #path to place selected .npy xbar particles
ncores = 32
start = 0
finish = None
savestr=''
infodtype = [('time','d'),('n','i'),('ndim','i'),('ng','i'),('nd','i'),('ns','i'),('on','i')]
stellardtype = [('mass','f'), ('x','f'), ('y','f'), ('z','f'), ('vx','f'), ('vy','f'), ('vz','f'), ('metals','f'), ('tform','f'), ('ID','Q')]
dmdtype = [('mass','f'), ('x','f'), ('y','f'), ('z','f'), ('vx','f'), ('vy','f'), ('vz','f'), ('ID','Q')]
commondtype = [('t','d'),('x','f'), ('y','f'), ('z','f'), ('vx','f'), ('vy','f'), ('vz','f'), ('vr','f'), ('vphi','f'), ('vzz','f'), ('r','f'), ('phi','f'), ('zz','f'), ('mass','f'), ('idd','Q')]
oldsnapdtype = [('t','d'), ('idd','Q'),('x','f'), ('y','f'), ('z','f'), ('vx','f'), ('vy','f'), ('vz','f'), ('vr','f'), ('vphi','f'), ('vzz','f'), ('r','f'), ('phi','f'), ('zz','f'), ('mass','f')]
pointdtype = [('idd','Q'),('core','H'),('seek','I')]


#%%annulus params
#%%
# from forsims import *
#%%

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
            info= np.fromfile(f,dtype=infodtype,count=1)
            infoBytes = f.tell()
            if verbose>2:
                print(infoBytes)
            #skip darkmatter
            #read the first dm line
            if verbose>2:
                print(f.tell())
            catd = np.fromfile(f,dtype= dmdtype, count=1)   
            #get the bytes location and subtract off the bytes location after loading info to get n bytes a line for dm
            if verbose>2:
                print(f.tell())
            current = f.tell()
            dmBytes = current-infoBytes
            f.seek(dmBytes*(info['nd'][0]-1)+current)
            if verbose>2:
                print(f.tell())
            # stars setup                    
            cats= np.fromfile(f,dtype=stellardtype, count=info['ns'][0])
            if verbose>2:
                print('done')
        else:
            if verbose>1:
                print(filename)
            #file info
            info= np.fromfile(f,dtype=infodtype,count=1)
            if verbose>2:
                print(f.tell())
            # #dark matter setup count is reading the number of ?rows? 
            catd= np.fromfile(f,dmdtype, count=info['nd'][0]) 
            if verbose>2:
                print(f.tell())                   
            # stars setup                    
            cats= np.fromfile(f,dtype=stellardtype, count=info['ns'][0])
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
    snaparr = np.empty(nstar,dtype=commondtype)    

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

def makepointerf(path,pointfpath,wdmBool=0,verbose=0,dosort=1):
    #%% initial load of the first timestep to set up the file
    times = np.genfromtxt(path+'times.txt',dtype='str')
    ncores = len(glob.glob(path+'*'+times[0][:-1]+'*'))
    if verbose >0:
        print('Loading to find nstars, ndark, and ntot'+path+times[0], flush=True)
    if wdmBool == True:
        _,_,info=loader(path+times[0],wdm=wdmBool,verbose=verbose)
    else:
        _,info=loader(path+times[0],wdm=wdmBool,verbose=verbose)
    ntot=info['n']
    ndark=info['nd']
    nstar=info['ns']
    for j in range(1,ncores):
        if verbose > 1:
            print('Loading '+path+times[0][:-1]+str(j), flush=True)
        if wdmBool == True:
            _,_,info=loader(path+times[0][:-1]+str(j),wdm=wdmBool,verbose=verbose)
        else:
            _,info=loader(path+times[0][:-1]+str(j),wdm=wdmBool,verbose=verbose)
        ntot=ntot+info['n'] 
        ndark=ndark+info['nd']
        nstar=nstar+info['ns']
        if verbose > 1:
            print('in length maker',ntot,ndark,nstar,info['n'],info['nd'],info['ns'],flush=True)
    #%   
    starti = len(glob.glob(pointfpath+'*'))
    for ts,tt in enumerate(times[starti:]):
        #set file name for the timestep pointer file
        tsstr = tt.split('-')[:-1][0]
        if verbose >3:
            print(tsstr)
        tsname = 'stellar_'+path.split('/')[-2]+'_'+tsstr

        #?sort cause this could be used to make it faster later, cause they're unique we dont really need to do this
        #initalize this file's stellar array
        pointdtype = [('idd','Q'),('core','H'),('seek','I')]
        starpointarr = np.empty(nstar,dtype=pointdtype)
        
        arrayindx = 0
        for nc in range(0,ncores):
            if verbose >2:
                print('core '+str(nc), flush=True)   
            #load nth core of timestep/snapshot 
            fname = path+tsstr+'-'+str(nc)

            if verbose >1:
                print('Loading '+fname, flush=True)
            
            with open(fname, 'rb') as f:
                # get byte lengths 
                info = np.fromfile(f,dtype=infodtype,count=1)
                infoBytes = f.tell()
                if verbose>2:
                    print('Info byte length is '+str(infoBytes), flush=True)

                #read the first dm line
                catd = np.fromfile(f,dtype= dmdtype, count=1)  
                current = f.tell()
                dmBytes = current-infoBytes
                if verbose>2:
                    print('DM byte length is '+str(dmBytes), flush=True)

                #seek forward past rest of dm data
                f.seek(dmBytes*(info['nd'][0]-1)+current)
                if verbose>2:
                    print('Advanced through DM to '+str(f.tell()), flush=True)
                
                #get star byte info
                cats= np.fromfile(f,dtype=stellardtype, count=1)
                current = f.tell()
                if verbose >2:
                    print(f.tell())
                stBytes = current-dmBytes*(info['nd'][0])-infoBytes
                if verbose>2:
                    print('Stellar byte length is '+str(stBytes), flush=True)
                    print(f.tell())
                #seek back to start of stars
                startofStellar = current-stBytes
                f.seek(startofStellar) 
                if verbose>2:
                    print('Back to start of star '+str(f.tell()))
                
                # stars setup
                if verbose >1:
                    print('loading stellar')
                    printnow()          
                cats= np.fromfile(f,dtype=stellardtype, count=info['ns'][0])
                afterStellar = f.tell()
                print('after stars '+str(f.tell()))
                if info['ns'][0] > 0:
                    starpointarr['idd'][arrayindx:arrayindx+info['ns'][0]] = cats['ID']
                    starpointarr['core'][arrayindx:arrayindx+info['ns'][0]] = nc
                    starpointarr['seek'][arrayindx:arrayindx+info['ns'][0]] = np.arange(startofStellar,afterStellar,stBytes)

                else:
                    if verbose>1:
                        print('No stellar particles in '+fname, flush=True)
                if verbose >1:
                    printnow()
                print(arrayindx)
                arrayindx += info['ns'][0]
                print(arrayindx)

        if verbose >1:
            print('saving timestep '+tsname)
            printnow()     
        
        #sort by idx so that the loc is the idd
        if dosort == 1:
            sortorder = np.argsort(starpointarr['idd'])

            sortpointer = np.empty(len(starpointarr),dtype=pointdtype)

            sortpointer['idd'] = starpointarr['idd'][sortorder]
            sortpointer['core'] = starpointarr['core'][sortorder]
            sortpointer['seek'] = starpointarr['seek'][sortorder]

            #save as a pickle .npy file cause it's wicked fast
            np.save(pointfpath+tsname,sortpointer)
        else:
            #save as a pickle .npy file cause it's wicked fast
            np.save(pointfpath+tsname,starpointarr)
        
        if verbose >1:
            print('finished saving timestep '+tsname)
            printnow()     

#%% load one particle over time
def loadonesource(path,idx,npypointerpath=pointfpath,ncores=32,start=0,finish=None,wdmBool=False,verbose=0,errrange=200,issorted=1,approxhalofrac=.1):
    '''
    takes a path to the folder containing snapshots and an id
    returns an array 
        dtype=[('t','f'),('x','f'), ('y','f'), ('z','f'), 
            ('vx','f'), ('vy','f'), ('vz','f'), 
            ('vr','f'), ('vphi','f'), ('vzz','f'), 
            ('r','f'), ('phi','f'), ('zz','f'), ('mass','f'), ('idd','Q')] 

    set 'npypointerpath' to pointfpath
    or 'loadall': loadall will toggle to itterate through and load all the arrays
                                      npypointer will access .npy pointer files

    '''

    idx = np.uint64(idx)
    #load the times file
    times=np.genfromtxt(path+'times.txt',dtype='str')

    #set other params
    if finish == None:
        finish = len(times)
        if npypointerpath is not None:
            finish = len(glob.glob(npypointerpath+'*'))+start
    # let's allocate space for the finished final array for one source
    tlen = len(range(start,finish))
    sourcearr=np.empty(tlen,dtype=commondtype)
    
    if npypointerpath == None:
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
    else:
        #check paths error or prompt?? idk
        if os.path.exists(npypointerpath):
            indvarrayindx=0
            lspoint = os.listdir(npypointerpath)
            lspoint.sort(key=lambda s:s.split('_')[-1].split('.npy')[0])
            for i in range(start,finish):
                pointerpath = npypointerpath+lspoint[i]
                #get pointer file
                if verbose >2:
                    print('load pointer path: ',pointerpath)
                    printnow()
                pointer = np.load(pointerpath,mmap_mode='r',allow_pickle=1) 
                
                if issorted == 0:
                    bf = datetime.datetime.now()
                    found = pointer[pointer['idd']==idx]
                    if verbose > 2:
                        print('time for starpointarr[idx] is ',datetime.datetime.now()-bf)
                else:
                    #offset application
                    if idx >10000000000000:
                        if verbose > 2:
                            bf = datetime.datetime.now()
                        try:
                            #grab sub array of region with halo particles 
                            subr = pointer[-round(len(pointer)*approxhalofrac+errrange):]
                            
                            found = subr[subr['idd']==idx]
                            if verbose >2:
                                print('time for subr[subr[\'idd\']==idx] is ',datetime.datetime.now()-bf)
                        except:
                            print('\n\nISSUE WITH FILE: ',pointerpath)
                            print('pointer[0]: ',pointer[0])
                            print('id is ', idx)
                            if pointer['idd'][0]==pointer['seek'][0]:
                                print('all zeros')
                            
                    
                    else:
                        if verbose > 2:
                            bf = datetime.datetime.now()
                        found=pointer[[idx]]

                        #catch when there's this weird offset of ~200 skipped IDs
                        if pointer[[idx]]['idd'][0]!=idx:
                            if verbose >1:
                                print('using error range to find local source')
                            #try except for end of array 
                            try:
                                subarr = pointer[int(idx-errrange):int(idx+errrange)]
                            except:
                                subarr = pointer[int(idx-errrange):]
                            found = subarr[subarr['idd']==idx]
                            if len(found) <1:
                                if verbose >1:
                                    print('expanding error range to find local source')
                                errrange *=2
                                try:
                                    subarr = pointer[int(idx-errrange):int(idx+errrange)]
                                except:
                                    subarr = pointer[int(idx-errrange):]
                                found = subarr[subarr['idd']==idx]
                                if len(found) <1:
                                    print('\n\nthere\'s something wrong, error range expansion isn\'t enough to solve it')
                                    
                        if verbose > 2:
                            print('time for pointer[idx] is ',datetime.datetime.now()-bf)

                #load the direct file
                try:
                    if verbose > 0:
                        print('Loading '+pointerpath, flush=True)
                        if verbose >2:
                            printnow()
                    filename = path+times[i][:-1]+str(found['core'][0])
                    if verbose>3:
                        print('core is ',found['core'][0])
                    with open(filename, 'rb') as f:
                        info= np.fromfile(f,dtype=infodtype,count=1)
                        if verbose>3:
                            print('seek to ',found['seek'][0])
                        f.seek(found['seek'][0])
                        cats= np.fromfile(f,dtype=stellardtype, count=1)

                        if verbose >2:
                            print('insert into indvarray')
                            printnow()
                        if verbose >1:
                            bf = datetime.datetime.now()
                        sourcearr['x'][indvarrayindx]=cats['x']
                        sourcearr['y'][indvarrayindx]=cats['y']
                        sourcearr['z'][indvarrayindx]=cats['z']
                        sourcearr['vx'][indvarrayindx]=cats['vx']
                        sourcearr['vy'][indvarrayindx]=cats['vy']
                        sourcearr['vz'][indvarrayindx]=cats['vz']
                        sourcearr['mass'][indvarrayindx]=cats['mass']
                        sourcearr['idd'][indvarrayindx]=cats['ID']
                        if verbose >1:
                            print('time for plugging into array is ',datetime.datetime.now()-bf)
                        if verbose >2:
                            print('coords transf')
                            printnow()
                        if verbose >1:
                            bf = datetime.datetime.now()
                        sourcearr['vr'][indvarrayindx],sourcearr['vphi'][indvarrayindx],sourcearr['vzz'][indvarrayindx]=coords.rect_to_cyl_vec(cats['vx'],cats['vy'],cats['vz'],cats['x'],cats['y'],cats['z'])
                        sourcearr['r'][indvarrayindx],sourcearr['phi'][indvarrayindx],sourcearr['zz'][indvarrayindx]=coords.rect_to_cyl(cats['x'],cats['y'],cats['z'])
                        sourcearr['t'][indvarrayindx]=info['time']*9.778145/1000.
                        if verbose >1:
                            print('time for computing and plugging new vals into array is ',datetime.datetime.now()-bf)
                        if verbose >2:
                            print('onto the next loop')
                            printnow()
                except:
                    print('\n\nISSUE WITH FILE: ',pointerpath,flush=True)
                    print('found[0]: ',found[0],flush=True)
                indvarrayindx+=1
    return(sourcearr)

#%% this function cant be more efficient than doing the above individually till indexing is fixed
# def loadsources(path,idxarr,npypointerpath=pointfpath,ncores=32,start=0,finish=None,wdmBool=False,verbose=0,errrange=200,issorted=1,approxhalofrac=.1):
#     '''
#     takes a path to the folder containing snapshots and an id
#     returns an array 
#         dtype=[('t','f'),('x','f'), ('y','f'), ('z','f'), 
#             ('vx','f'), ('vy','f'), ('vz','f'), 
#             ('vr','f'), ('vphi','f'), ('vzz','f'), 
#             ('r','f'), ('phi','f'), ('zz','f'), ('mass','f'), ('idd','Q')] 

#     set 'npypointerpath' to pointfpath
#     or 'loadall': loadall will toggle to itterate through and load all the arrays
#                                       npypointer will access .npy pointer files

#     '''

#     idxarr = np.uint64(idxarr)
#     #load the times file
#     times=np.genfromtxt(path+'times.txt',dtype='str')

#     #set other params
#     if finish == None:
#         finish = len(times)
#         if npypointerpath is not None:
#             finish = len(glob.glob(npypointerpath+'*'))+start
#     # let's allocate space for the finished final array for one source
#     tlen = len(range(start,finish))
#     sourcearr=np.empty((tlen,len(idxarr)),dtype=commondtype)
    
    
#     #check paths error or prompt?? idk
#     if os.path.exists(npypointerpath):
#         indvarrayindx=0
#         lspoint = os.listdir(npypointerpath)
#         lspoint.sort(key=lambda s:s.split('_')[-1].split('.npy')[0])
#         for i in range(start,finish):
#             pointerpath = npypointerpath+lspoint[i]
#             #get pointer file
#             if verbose >2:
#                 print('load pointer path: ',pointerpath)
#                 printnow()
#             pointer = np.load(pointerpath,mmap_mode='r+',allow_pickle=1) 
            
#             if issorted == 0:
#                 bf = datetime.datetime.now()
#                 found = pointer[pointer['idd']==idxarr]
#                 if verbose > 2:
#                     print('time for starpointarr[idx] is ',datetime.datetime.now()-bf)
#             else:
#                 #offset application
#                 if idxarr >10000000000000:
#                     if verbose > 2:
#                         bf = datetime.datetime.now()
#                     try:
#                         #grab sub array of region with halo particles 
#                         subr = pointer[-round(len(pointer)*approxhalofrac+errrange):]
                        
#                         found = subr[subr['idd']==idxarr]
#                         if verbose >2:
#                             print('time for subr[subr[\'idd\']==idx] is ',datetime.datetime.now()-bf)
#                     except:
#                         print('\n\nISSUE WITH FILE: ',pointerpath)
#                         print('pointer[0]: ',pointer[0])
#                         print('id is ', idxarr)
#                         if pointer['idd'][0]==pointer['seek'][0]:
#                             print('all zeros')
                        
                
#                 else:
#                     if verbose > 2:
#                         bf = datetime.datetime.now()
#                     found=pointer[[idx]]

#                     #catch when there's this weird offset of ~200 skipped IDs
#                     if pointer[[idx]]['idd'][0]!=idx:
#                         if verbose >1:
#                             print('using error range to find local source')
#                         #try except for end of array 
#                         try:
#                             subarr = pointer[int(idx-errrange):int(idx+errrange)]
#                         except:
#                             subarr = pointer[int(idx-errrange):]
#                         found = subarr[subarr['idd']==idx]
#                         if len(found) <1:
#                             if verbose >1:
#                                 print('expanding error range to find local source')
#                             errrange *=2
#                             try:
#                                 subarr = pointer[int(idx-errrange):int(idx+errrange)]
#                             except:
#                                 subarr = pointer[int(idx-errrange):]
#                             found = subarr[subarr['idd']==idx]
#                             if len(found) <1:
#                                 print('\n\nthere\'s something wrong, error range expansion isn\'t enough to solve it')
                                
#                     if verbose > 2:
#                         print('time for pointer[idx] is ',datetime.datetime.now()-bf)

#             #load the direct file
#             try:
#                 if verbose > 0:
#                     print('Loading '+pointerpath, flush=True)
#                     if verbose >2:
#                         printnow()
#                 filename = path+times[i][:-1]+str(found['core'][0])
#                 if verbose>3:
#                     print('core is ',found['core'][0])
#                 with open(filename, 'rb') as f:
#                     info= np.fromfile(f,dtype=infodtype,count=1)
#                     if verbose>3:
#                         print('seek to ',found['seek'][0])
#                     f.seek(found['seek'][0])
#                     cats= np.fromfile(f,dtype=stellardtype, count=1)

#                     if verbose >2:
#                         print('insert into indvarray')
#                         printnow()
#                     if verbose >1:
#                         bf = datetime.datetime.now()
#                     sourcearr['x'][indvarrayindx]=cats['x']
#                     sourcearr['y'][indvarrayindx]=cats['y']
#                     sourcearr['z'][indvarrayindx]=cats['z']
#                     sourcearr['vx'][indvarrayindx]=cats['vx']
#                     sourcearr['vy'][indvarrayindx]=cats['vy']
#                     sourcearr['vz'][indvarrayindx]=cats['vz']
#                     sourcearr['mass'][indvarrayindx]=cats['mass']
#                     sourcearr['idd'][indvarrayindx]=cats['ID']
#                     if verbose >1:
#                         print('time for plugging into array is ',datetime.datetime.now()-bf)
#                     if verbose >2:
#                         print('coords transf')
#                         printnow()
#                     if verbose >1:
#                         bf = datetime.datetime.now()
#                     sourcearr['vr'][indvarrayindx],sourcearr['vphi'][indvarrayindx],sourcearr['vzz'][indvarrayindx]=coords.rect_to_cyl_vec(cats['vx'],cats['vy'],cats['vz'],cats['x'],cats['y'],cats['z'])
#                     sourcearr['r'][indvarrayindx],sourcearr['phi'][indvarrayindx],sourcearr['zz'][indvarrayindx]=coords.rect_to_cyl(cats['x'],cats['y'],cats['z'])
#                     sourcearr['t'][indvarrayindx]=info['time']*9.778145/1000.
#                     if verbose >1:
#                         print('time for computing and plugging new vals into array is ',datetime.datetime.now()-bf)
#                     if verbose >2:
#                         print('onto the next loop')
#                         printnow()
#             except:
#                 print('\n\nISSUE WITH FILE: ',pointerpath,flush=True)
#                 print('found[0]: ',found[0],flush=True)
#             indvarrayindx+=1
#     else:
#         print('PATH IS WRONG')
# #     return(sourcearr)
#%%
# printnow()
# test = loadonesource(path,118301268,start=0,finish=10,npypointerpath=pointfpath,verbose=1)
# printnow()
#%% plotting
def doplots(path=None,idstr=None,inarr=None,mode='load',singleplot=1,save=0,savestr='',lims=[-30,30],tmin=0,tmax=110,mks=20,npypointerpath=pointfpath):
    if mode == 'pandas':
        fl = pd.read_csv(path+'particle_'+str(idstr)+'.txt',comment='#',sep=',')
    if mode == 'numpy':
        fl  = np.loadtxt(path+'particle_'+str(idstr)+'.txt',dtype=commondtype)
    if mode == 'array':
        fl = inarr
    if mode == 'load':
        fl = loadonesource(path,idstr,npypointerpath=pointfpath,verbose=0,start=tmin,finish=tmax,issorted=1)
    if save == 1:
        # matplotlib.use('pdf')
        matplotlib.use('agg')


    if tmax is None:
        tmax = len(fl)

    if singleplot == 0:
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
        f.colorbar(p,label='Gyr')
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
        f.colorbar(p,label='Gyr')
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
        f.colorbar(p,label='Gyr')
        if save == 1:
            plt.savefig('plots/%s%s_xy.png' % (str(idstr),savestr),bbox_inches='tight',dpi=500)
            plt.close()
        else:
            plt.show()
    else:
        f = plt.figure(figsize=(8,6))   
        plt.suptitle('particle '+str(fl['idd'][0]))
        ax = f.add_subplot(221, projection='3d')
        ax.plot3D(fl['x'][tmin:tmax],fl['y'][tmin:tmax],fl['z'][tmin:tmax],c='grey',lw=1)
        p = ax.scatter3D(fl['x'][tmin:tmax],fl['y'][tmin:tmax],fl['z'][tmin:tmax],c=fl['t'][tmin:tmax],s=mks)

        ax.set_xlabel('x [kpc]')
        ax.set_ylabel('y [kpc]')
        ax.set_zlabel('z [kpc]')
        ax.set_xlim(lims[0],lims[1])
        ax.set_ylim(lims[0],lims[1])
        ax.set_zlim(lims[0],lims[1])
        ax.view_init(azim=45)

        ax = f.add_subplot(222)
        ax.plot(fl['y'][tmin:tmax],fl['z'][tmin:tmax],c='grey',lw=1)
        p = ax.scatter(fl['y'][tmin:tmax],fl['z'][tmin:tmax],c=fl['t'][tmin:tmax],s=mks)
        ax.set_xlabel('y [kpc]')
        ax.set_xlim(lims[0],lims[1])
        ax.set_ylabel('z [kpc]')
        ax.set_ylim(lims[0],lims[1])
        f.colorbar(p,label='Gyr')

        ax = f.add_subplot(223)
        ax.plot(fl['x'][tmin:tmax],fl['y'][tmin:tmax],c='grey',lw=1)
        p = ax.scatter(fl['x'][tmin:tmax],fl['y'][tmin:tmax],c=fl['t'][tmin:tmax],s=mks)
        ax.set_xlabel('x [kpc]')
        ax.set_xlim(lims[0],lims[1])
        ax.set_ylabel('y [kpc]')
        ax.set_ylim(lims[0],lims[1])

        ax = f.add_subplot(224)
        ax.plot(fl['r'][tmin:tmax],fl['vr'][tmin:tmax],c='grey',lw=1)
        p = ax.scatter(fl['r'][tmin:tmax],fl['vr'][tmin:tmax],c=fl['t'][tmin:tmax],s=mks)
        ax.set_xlabel('r [kpc]')
        ax.set_xlim(lims[0],lims[1])
        ax.set_ylabel('vr [km/s]')
        ax.set_ylim(lims[0],lims[1])

        if save == 1:
            plt.savefig('plots/%s%s_all.png' % (str(idstr),savestr),bbox_inches='tight',dpi=500)
            plt.close()
        else:
            plt.show()


def dynamicplots(path=None,idstr=None,params=[('x','y'),('y','z'),('r','vr')],mode='load',save=0,savestr='',lims=[-30,30],tmin=0,tmax=None,mks=20,lws=1,inarr=None,npypointerpath=pointfpath,fig=None):
    '''
    makes a 2x2 series of plots with a 3d plot in the top left and then 3 plots indexed clockwise of input params (x,y)
    '''
    if mode == 'array':
        fl = inarr
    if mode == 'load':
        fl = loadonesource(path,idstr,npypointerpath=npypointerpath,verbose=0,start=tmin,finish=tmax,issorted=1)
    if save == 1:
        matplotlib.use('agg')

    
    p2x,p2y=params[0]
    p3x,p3y=params[1]
    p4x,p4y=params[2]

    if tmax is None:
        tmax = len(fl)

    if fig is None:
        f = plt.figure(figsize=(9,6))   
        plt.suptitle('particle '+str(fl['idd'][0]))
    else:
        f = fig
        plt.suptitle('many particles')
    ax = f.add_subplot(221, projection='3d')
    ax.plot3D(fl['x'],fl['y'],fl['z'],c='grey',lw=lws)
    p = ax.scatter3D(fl['x'],fl['y'],fl['z'],c=fl['t'],s=mks)

    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_zlabel('z [kpc]')
    ax.set_xlim(lims[0],lims[1])
    ax.set_ylim(lims[0],lims[1])
    ax.set_zlim(lims[0],lims[1])
    ax.view_init(azim=45)

    ax = f.add_subplot(222)
    ax.plot(fl[p2x],fl[p2y],c='grey',lw=lws)
    p = ax.scatter(fl[p2x],fl[p2y],c=fl['t'],s=mks)
    ax.set_xlabel('%s'%p2x)
    ax.set_xlim(lims[0],lims[1])
    ax.set_ylabel('%s'%p2y)
    ax.set_ylim(lims[0],lims[1])
    f.colorbar(p,label='Gyr')

    ax = f.add_subplot(223)
    ax.plot(fl[p3x],fl[p3y],c='grey',lw=lws)
    p = ax.scatter(fl[p3x],fl[p3y],c=fl['t'],s=mks)
    ax.set_xlabel('%s'%p3x)
    ax.set_xlim(lims[0],lims[1])
    ax.set_ylabel('%s'%p3y)
    ax.set_ylim(lims[0],lims[1])

    ax = f.add_subplot(224)
    ax.plot(fl[p4x],fl[p4y],c='grey',lw=lws)
    p = ax.scatter(fl[p4x],fl[p4y],c=fl['t'],s=mks)
    ax.set_xlabel('%s'%p4x)
    ax.set_xlim(lims[0],lims[1])
    ax.set_ylabel('%s'%p4y)
    ax.set_ylim(lims[0],lims[1])

    if fig is not None:
        return(f)
    elif save == 1:
        plt.savefig('plots/%s%s_all.png' % (str(idstr),savestr),bbox_inches='tight',dpi=500)
        plt.close()
    else:
        plt.show()

# %%
print('end at: '+str(datetime.datetime.now()),flush=True)
