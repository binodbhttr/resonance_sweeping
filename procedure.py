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
from astropy.units import equivalencies
import numpy as np
try:
    from galpy.util import coords
except:
    from galpy.util import bovy_coords as coords
import gala.coordinates as gcoords
try: 
    from superfreq import SuperFreq
except:
    print('PLEASE IMPORT superfreq: pip install --user git+https://github.com/adrn/superfreq', flush=True)
import astropy.units as u
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
# %%
# %%
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

times = np.genfromtxt(path+'times.txt',dtype='str')
pattern_speeds = np.load(pointfpath+'patternspeed.npy')


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
def loader(filename,wdm=0,verbose=0,kmpers=1):
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
        if kmpers == 1:
            catd['vx']*=100.
            catd['vy']*=100.
            catd['vz']*=100.
    cats['mass']*=2.324876e9
    if kmpers == 1:
            cats['vx']*=100.
            cats['vy']*=100.
            cats['vz']*=100.
    
    if wdm == True:
        return(catd,cats,info)
    else:
        return(cats,info)

#% load one whole timestep
def loadwholesnap(path,timestepid,ncores=32,ntot=1279999360,ndark=1003921280,nstar=276078080,wdmBool=False,verbose=0,barframe=0,kmpers=1):

    #load the times file
    times=np.genfromtxt(path+'times.txt',dtype='str')
    if kmpers != 1:
        print('ATTN: Enabled 100 km/s units instead of km/s for velocities.')

    #if the number of total, dark, and stellar particles are not set then determine
    if ntot==None:
        print('Loading to find nstars, ndark, and ntot'+path+times[timestepid], flush=True)
        if wdmBool == True:
            _,cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose,kmpers=kmpers)
        else:
            cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose,kmpers=kmpers)
        ntot=info['n']
        ndark=info['nd']
        nstar=info['ns']
        for j in range(1,ncores):
            print('Loading '+path+times[timestepid][:-1]+str(j), flush=True)
            if wdmBool == True:
                _,cats,info=loader(path+times[timestepid][:-1]+str(j),wdm=wdmBool,verbose=verbose,kmpers=kmpers)
            else:
                cats,info=loader(path+times[timestepid][:-1]+str(j),wdm=wdmBool,verbose=verbose,kmpers=kmpers)
            ntot=ntot+info['n']
            ndark=ndark+info['nd']
            nstar=nstar+info['ns']
            print('in length maker',ntot,ndark,nstar,info['n'],info['nd'],info['ns'],flush=True)
    if verbose > 1:
        print('\nLoading for info '+path+times[timestepid], flush=True)
    if wdmBool == True:
        _,cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose,kmpers=kmpers)
    else:
        cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose,kmpers=kmpers)

    if verbose > 1:
        print(info['n'],info['ns'],info['nd'], flush=True)
    tnstar=int(info['ns'])

    if verbose > 2:
        print(ntot,ndark,nstar, flush=True) 
    snaparr = np.empty(nstar,dtype=commondtype)    

    if verbose > 1:
        print('Loading '+path+times[timestepid], flush=True)
    if wdmBool == True:
        _,cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose,kmpers=kmpers)
    else:
        cats,info=loader(path+times[timestepid],wdm=wdmBool,verbose=verbose,kmpers=kmpers)

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
            _,cats,info=loader(path+times[timestepid][:-1]+str(j),wdm=wdmBool,verbose=verbose,kmpers=kmpers)
        else:
            cats,info=loader(path+times[timestepid][:-1]+str(j),wdm=wdmBool,verbose=verbose,kmpers=kmpers)
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

    #move into bar frame if true
    if barframe == True:
        #check for path
        if os.path.exists(pointfpath+'patternspeeds.npy'):
            patternspeeds = np.load(pointfpath+'patternspeeds.npy',allow_pickle=1,mmap_mode='r')
            pspeed = patternspeeds[['tind']==timestepid]['patternspeed_radperGyr'] 
        else:
            pspeed = getpatternspeed(path,timestepid,inputsnaparr=snaparr,verbose=verbose) #also in radperGyr
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
def loadonesource(path,idx,npypointerpath=pointfpath,ncores=32,start=0,finish=None,wdmBool=False,verbose=0,errrange=200,issorted=1,approxhalofrac=.1,kmpers=1):
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
    if kmpers != 1:
        print('ATTN: Enabled 100 km/s units instead of km/s for velocities.')
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
                    _,cats,info=loader(path+times[i][:-1]+str(lastcore),wdm=wdmBool,verbose=verbose,kmpers=kmpers)
                else:
                    cats,info=loader(path+times[i][:-1]+str(lastcore),wdm=wdmBool,verbose=verbose,kmpers=kmpers)
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
                            _,cats,info=loader(path+times[i][:-1]+str(j),wdm=wdmBool,verbose=verbose,kmpers=kmpers)
                        else:
                            cats,info=loader(path+times[i][:-1]+str(j),wdm=wdmBool,verbose=verbose,kmpers=kmpers)
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
                        if kmpers == 1:
                                cats['vx']*=100.
                                cats['vy']*=100.
                                cats['vz']*=100.
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
#%% plotting
def doplots(path=None,idstr=None,inarr=None,mode='load',singleplot=1,save=0,savestr='',lims=[-30,30],tmin=0,tmax=110,mks=20,npypointerpath=pointfpath,kmpers=0):
    if mode == 'pandas':
        fl = pd.read_csv(path+'particle_'+str(idstr)+'.txt',comment='#',sep=',')
    if mode == 'numpy':
        fl  = np.loadtxt(path+'particle_'+str(idstr)+'.txt',dtype=commondtype)
    if mode == 'array':
        fl = inarr
    if mode == 'load':
        fl = loadonesource(path,idstr,npypointerpath=pointfpath,verbose=0,start=tmin,finish=tmax,issorted=1,kmpers=kmpers)
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
        if kmpers == 1:
            ax.set_ylabel('vr [km/s]')
        else:
            ax.set_ylabel('vr [100 km/s]')
        ax.set_ylim(lims[0],lims[1])

        if save == 1:
            plt.savefig('plots/%s%s_all.png' % (str(idstr),savestr),bbox_inches='tight',dpi=500)
            plt.close()
        else:
            plt.show()


def dynamicplots(path=None,idstr=None,params=[('x','y'),('y','z'),('r','vr')],mode='load',save=0,savestr='',lims=[-30,30],tmin=0,tmax=None,mks=20,lws=1,inarr=None,npypointerpath=pointfpath,fig=None,kmpers=0):
    '''
    makes a 2x2 series of plots with a 3d plot in the top left and then 3 plots indexed clockwise of input params (x,y)
    '''
    if mode == 'array':
        fl = inarr
    if mode == 'load':
        fl = loadonesource(path,idstr,npypointerpath=npypointerpath,verbose=0,start=tmin,finish=tmax,issorted=1,kmpers=kmpers)
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
        plt.savefig('plots/%s%s_all.png' % (savestr,str(idstr)),bbox_inches='tight',dpi=500)
        plt.close()
    else:
        plt.show()


#%% working with the data
# def() frequency decomposition
# # cordsys = 'cartesian'
# cordsys = 'cylindrical'
# tstart = 0
# tstep = 100 #load 100 for 0.50ish Gyr per
# ntsteps = 10

# # set array for selected ids and freqs over time
# # earlyarr = np.empty((len(xbarsample_early),ntsteps),dtype=[('ti','d'),('tf','d'),('freqi','f'), ('freqj','f'), ('freqk','f'), ('idd','Q')])
# # latearr = np.empty((len(xbarsample_late),ntsteps),dtype=[('ti','d'),('tf','d'),('freqi','f'), ('freqj','f'), ('freqk','f'), ('idd','Q')])
# buildarr = np.empty((len(xbarsample_late),ntsteps),dtype=[('ti','d'),('tf','d'),('freqi','f'), ('freqj','f'), ('freqk','f'), ('idd','Q')])

# for selectedids,sstr in [(xbarsample_early,'early'), (xbarsample_late,'late')]:
#     for ind,iii in enumerate(selectedids):
#         startt = tstart
#         for ttt in range(ntsteps):
#             try :
#                 ii=loadonesource(path,iii,verbose=0,start=startt,finish=startt+tstep,issorted=1)
                
#                 #Create a SuperFreq object by passing in the array of times to the initializer:
#                 sf = SuperFreq(ii['t'])
                
#                 #Set array
#                 if cordsys == 'cylindrical': #does this even work whatre the units of phi from -pi,pi
#                     w = np.array([ii['r']*u.kpc, ii['phi']*u.kpc, ii['zz']*u.kpc, ii['vr']*(u.km/u.s).to(u.kpc/u.Gyr), ii['vphi']*(u.km/u.s).to(u.kpc/u.Gyr), ii['vzz']*(u.km/u.s).to(u.kpc/u.Gyr)],dtype='d')
#                 else:# implied cordsys == 'cartesian':
#                     w = np.array([ii['x']*u.kpc, ii['y']*u.kpc, ii['z']*u.kpc, ii['vx']*(u.km/u.s).to(u.kpc/u.Gyr), ii['vy']*(u.km/u.s).to(u.kpc/u.Gyr), ii['vz']*(u.km/u.s).to(u.kpc/u.Gyr)],dtype='d')

#                 #Define the complex time series from which we would like to derive the fundamental frequencies as three complex arrays from the orbit data
#                 ndim = len(w)
#                 fs = [(w[i,:] * 1j*w[i+ndim//2,:]) for i in range(ndim//2)]

#                 #Run freq solver
#                 buildarr[ind,ttt]['idd'] = iii
#                 buildarr[ind,ttt]['ti'] = times[startt].split('-')[:-1][0].split('_')[-1]
#                 buildarr[ind,ttt]['tf'] = times[startt+tstep].split('-')[:-1][0].split('_')[-1]
#                 buildarr[ind,ttt]['freqi'] = np.abs(sf.find_fundamental_frequencies(fs).fund_freqs)[0]
#                 buildarr[ind,ttt]['freqj'] = np.abs(sf.find_fundamental_frequencies(fs).fund_freqs)[1]
#                 buildarr[ind,ttt]['freqk'] = np.abs(sf.find_fundamental_frequencies(fs).fund_freqs)[2]

#                 # tbl = sf.find_fundamental_frequencies(fs).freq_mode_table
#                 # ix = sf.find_fundamental_frequencies(fs).fund_freqs_idx
#             except:
#                 print('ISSUE WITH PARTICLE: ',iii,' over times ',startt,times[startt], times[startt+tstep],startt+tstep)
#             # printnow()
#             # print(tbl)
#             startt += tstep
#     np.savetxt('particle_logs/xbarparts_%s.txt'%sstr,buildarr,comments='#dtype=%s'%str([('ti','d'),('tf','d'),('freqi','f'), ('freqj','f'), ('freqk','f'), ('idd','Q')]))
#     buildarr = np.empty((len(xbarsample_late),ntsteps),dtype=[('ti','d'),('tf','d'),('freqi','f'), ('freqj','f'), ('freqk','f'), ('idd','Q')])

# move into the bar frame
def find_angle(snaparr, degBool=False, rmin=1.5, rmax=2.5, m=2):
    '''
    Computes the angle of the m[=2] bar to the 0 phi axis

    Inputs
    ---------------
    snaparr (array) [dtype: commondtype]: must be loaded with loadwholesnap()
    degBool (bool): default is False to return angle in radians
    rmin,rmax (floats) = radial bounds for selecting the region of the disk to compute the fft on
    m=2 mode as the bar mode

    Returns
    ---------------
    barangle (float): returns a float value in radians of the bar angle compared to snaparr['phi']=zero
    '''
    barsample = (snaparr['r']<rmax)&(snaparr['r']>rmin)&(snaparr['mass']<400)
    
    #binning from -pi to pi in 360 steps, i.e. degree at a time
    counts, _ = np.histogram(snaparr['phi'][barsample], bins = np.linspace(-np.pi, np.pi, 360))
    
    #fourier transform the counts removing the offset
    ff=np.fft.fft(counts-np.mean(counts)) 

    barangle = -np.angle(ff[m],deg=degBool)/m
    return barangle
###
### we're figuring out how to move into the bar frame when loading a single snap or a single source
###
def compute_pattern_speed(path, tmin=0, tmax=None, rmin=1.5, rmax=2.5, verbose=0):
    '''
    Create a csv file with the pattern speed in rad/Gyr and km/s over the time bounds provided. 
    rmin,rmax set the distance to compute the speed at and then their average value is the location for km/s pattern speed tangental
    '''
    times = np.genfromtxt(path+'times.txt',dtype='str')

    if tmax is None:
        tmax = len(times)
        
    outarr = np.empty((tmax-tmin,1),dtype=[('t','d'),('tind','H'),('barangle','f'),('patternspeed_radperGyr','f'),('patternspeed','f')])

    #compute bar angle for step ahead if not first timestep
    if tmin == 0:
        old_bar_angle = 0
    else:
        if verbose >0:
            print('Loading ',path)
        snaparr = loadwholesnap(path, tmin-1)
        old_bar_angle = find_angle(snaparr,rmin=rmin,rmax=rmax)

    for tt in range(tmin,tmax):
        if verbose >0:
            print('Loading ',path)
        snaparr = loadwholesnap(path, tt)
        #if first loop then set physical params
        if tt == tmin:
            if verbose >1:
                print('First loop in time range, determining timestep with particle ',str(snaparr['idd'][0]), 'between ',str(times[tmin+1]),str(times[tmin+2]))
            #grab the next timestep to measure the time resolution
            timestep = loadonesource(path,snaparr['idd'][0],start=tmin+1,finish=tmin+2)['t'][0]-snaparr['t'][0]
            
        bar_angle = find_angle(snaparr)
        pattern_speed = (old_bar_angle-bar_angle)/(timestep)

        #bar_angle > old_bar_angle, if difference is near 180, flip it
        if abs(old_bar_angle-bar_angle)>=(np.pi*3/4):
            pattern_speed = (old_bar_angle - (bar_angle + np.pi))/(timestep)
        
        if verbose >1:
            print('Bar angle is ', bar_angle, 'radians', flush=True)
            print('Pattern speed is ', pattern_speed, 'radians per Gyr', flush=True)

        
        outarr[tt]['t'] = snaparr['t'][0]
        outarr[tt]['tind'] = tt
        outarr[tt]['barangle'] = bar_angle
        outarr[tt]['patternspeed_radperGyr'] = pattern_speed
        outarr[tt]['patternspeed'] = pattern_speed*(u.radian/u.Gyr*u.kpc).to(u.km/u.s,equivalencies=u.dimensionless_angles()) #conversion factor to go from rad/Gyr to km/s/kpc 
        
        printnow()
        print('now at ', outarr[tt]['t'],flush=True)
        old_bar_angle = bar_angle
        np.save(pointfpath+'patternspeed',outarr)
    
    #save as a .npy in the pointer file path
    np.save(pointfpath+'patternspeed',outarr)

    # #put into a dataframe to save
    # bar_info = pd.DataFrame(np.array([outarr['t'],outarr['barangle'], outarr['patternspeed_radperGyr'], outarr['patternspeed']]).T, 
    #                     columns=['time','bar_angle','pattern_speed_radperGyr', 'pattern_speed_kmpersperkpc'])
    # bar_info.to_csv('bar_info.csv')
    
    return outarr

def getpatternspeed(path,tidx,patternspeedfile = None,withbarangle=True,prior_bar_angle=None,timestep=None,inputsnaparr=None,rmin=1.5,rmax=2.5,verbose=0):
    '''
    Function to compute the pattern speed [in Rad/Gyr] at any given snap. Will always return the patternspeed and 
    can opt to include bar angles of target snap to facilitate computation over time without reloading redundantly.

    If itterating over in series, first loop do not set prior_bar_angle but do return withbarangle,
        then go into loop with prior_bar_angles from previous step.
    
    If computing for an array that is already loaded, must still provide the time index in tidx (it checks to see they match),
        but then it can skip loadwholesnap of the targetarray though it will still have to load the prior one.
    '''
    times = np.genfromtxt(path+'times.txt',dtype='str')
    #if there is a pattern speed file for the simulation, grab from it
    if patternspeedfile is not None:
        print('load from file not configured as file DNE')
        patternspeeds = np.load(pointfpath+'patternspeed.npy',mmap_mode='r+',allow_pickle=1)
        patterninfo = patternspeeds[['tind']==tidx]
        bar_angle = patterninfo['barangle']
        timestep = patternspeeds[['tind']==tidx-1]-patterninfo['t']
        pattern_speed = patterninfo['patternspeed_radperGyr']
    
    #assume that you will load the desired timestep targeted whole snap and the snap prior unless they're provided
    else:
        #check for input array
        if inputsnaparr is not None:
            #check that the times match
            if round(inputsnaparr['t'][0],3) == round(float(times[tidx].split('-')[0].split('_')[2])*9.778145/1000.,3):
                bar_angle = find_angle(inputsnaparr,rmin=rmin,rmax=rmax)
            else:
                print('ATTN: time index is ', tidx, 'for times[tidx]', times[tidx],'while input timearr is at t=',str(inputsnaparr['t'][0]))
                print('Running loadwholesnap() at provided tidx.')
                targetsnaparr = loadwholesnap(path, tidx)
                bar_angle = find_angle(targetsnaparr,rmin=rmin,rmax=rmax)

        else:
            #load target info
            targetsnaparr = loadwholesnap(path, tidx)
            bar_angle = find_angle(targetsnaparr,rmin=rmin,rmax=rmax)
        
        #check for input prior bar angle
        if prior_bar_angle is not None:
            old_bar_angle = prior_bar_angle
            timestep = timestep
        else:
            priorsnaparr = loadwholesnap(path, tidx-1)
            old_bar_angle = find_angle(priorsnaparr,rmin=rmin,rmax=rmax)
            #compute timestep
            timestep = targetsnaparr['t'][0]-priorsnaparr['t'][0]

        #bar_angle > old_bar_angle, if difference is near 180, flip it
        pattern_speed = (old_bar_angle-bar_angle)/(timestep)
        if abs(old_bar_angle-bar_angle)>=(np.pi*3/4):
            pattern_speed = (old_bar_angle - (bar_angle + np.pi))/(timestep)
        
        if verbose >1:
            print('Bar angle is ', bar_angle, 'radians', flush=True)
            print('Pattern speed is ', pattern_speed, 'radians per Gyr',flush=True)

    if withbarangle == True:
        return (bar_angle,timestep,pattern_speed)
    else:
        return pattern_speed
'''
def intobarframe(pattern_speed, time, snaparr=None,sourcearr=None):
    \'''
    Takes pattern_speed at a given time, the time (or array of times if using sourcearr mode) and removes this rotation to move into bar frame

    Inputs
    -----------------------
    pattern_speed [RAD/Gyr] (float)
    time [Gyr] (float)
    
    either:
        snaparr with commondtype of a single snap in time
    or:
        sourcearr with commondtype of one particle over a time series
    \'''
    if sourcearr is not None:
        #get time range
        ti = sourcearr['t'][0]
        tf = sourcearr['t'][-1]
        #check that to see if this is a single source at a single time
         ti==tf:
            time = sourcearr['t']
            x = sourcearr['x'][0]
            vx = sourcearr['vx'][0]
            y = sourcearr['y'][0]
            vy = sourcearr['vy'][0]
            phi = sourcearr['phi'][0]
            vphi = sourcearr['vphi'][0]

            rotation = pattern_speed*sourcearr['t']
            x_rot = x * np.cos(rotation) - y * np.sin(rotation)
            y_rot = x * np.sin(rotation) + y * np.cos(rotation)
            vx_rot = vx * np.cos(rotation) - vy * np.sin(rotation)
            vy_rot = vx * np.sin(rotation) + vy * np.cos(rotation)
            phi_rot = phi - rotation
            vphi_rot = vphi - pattern_speed



        else:
            



    rotation = pattern_speed*time
    x_rot = x * np.cos(rotation) - y * np.sin(rotation)
    y_rot = x * np.sin(rotation) + y * np.cos(rotation)
    vx_rot = vx * np.cos(rotation) - vy * np.sin(rotation)
    vy_rot = vx * np.sin(rotation) + vy * np.cos(rotation)
    phi_rot = phi - rotation
    vphi_rot = vphi = pattern_speed


    return updatedarr
'''
# %%
print('end at: '+str(datetime.datetime.now()),flush=True)