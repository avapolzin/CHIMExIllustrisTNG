import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c
import h5py
import matplotlib as mpl
from astropy.cosmology import FlatLambdaCDM
import os

########## For IllustrisTNG API (from https://www.tng-project.org/data/docs/api/) ###########
import requests
def get(path, params=None):
    # make HTTP GET request to path
    headers = {"api-key":"################################"} #redacted :)
    r = requests.get(path, params=params, headers=headers)
    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()
    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string
    return r

############### Molecular and atomic hydrogen (HI+H2) galaxy contents catalog ###############
## https://www.tng-project.org/data/docs/specifications/#sec5i
z1_h = h5py.File('hih2_galaxy_z=1.hdf5', 'r')
z2_h = h5py.File('hih2_galaxy_z=2.hdf5', 'r')

#################################### Group catalogs #########################################
## https://www.tng-project.org/data/docs/specifications/#sec2
basePath = '../../TNG300-1/output'
halos_z2 = il.groupcat.loadHalos(basePath,33, 
                                 fields = ['GroupNsubs', 'GroupPos', 'GroupFirstSub', 'Group_M_Crit200', 'Group_R_Crit200'])
subhalos_z2 = il.groupcat.loadSubhalos(basePath,33, fields = ['SubhaloGrNr', 'SubhaloPos', 'SubhaloStellarPhotometricsMassInRad'])
halos_z1 = il.groupcat.loadHalos(basePath,50, 
                                 fields = ['GroupNsubs', 'GroupPos', 'GroupFirstSub', 'Group_M_Crit200', 'Group_R_Crit200'])
subhalos_z1 = il.groupcat.loadSubhalos(basePath,50, fields = ['SubhaloGrNr', 'SubhaloPos', 'SubhaloStellarPhotometricsMassInRad'])
halos_z0 = il.groupcat.loadHalos(basePath,99, fields = ['GroupFirstSub', 'Group_M_Crit200'])
subhalos_z0 = il.groupcat.loadSubhalos(basePath,99, fields = ['SubhaloGrNr', 'SubhaloPos', 'SubhaloStellarPhotometricsMassInRad'])

#############################################################################################
h = 0.6774

def check_M200(verbose = True, plot = False):
    """
    Returns the Group IDs and positions of mass-selected galaxy clusters at z = 1, 2.
    
    This may take a while depending upon network/server speed.

    Optional print statements and plotting for debugging.
    (Would only advise plot = True in a notebook, since this will generate 280 figures.)
    """
    
    m200 = halos_z0['Group_M_Crit200']*1e10/h
    prims = halos_z0['GroupFirstSub'][m200 >= 1e14]
    
    z1_id = [] #mass-selected cluster Group IDs at z = 1
    z2_id = [] # "             "             "  at z = 2
    
    z1_iscen = [] #track whether central subhalo at z = 0 is central subhalo at z = 1
    z2_iscen = [] # "                         "                            " at z = 2
    
    z1_pos = [] #mass-selected cluster Group positions at z = 1
    z2_pos = [] # "                "                "  at z = 2
    
    for i in prims:
        # if verbose:
        #     print(i, end = '\r')
        url = 'http://www.illustris-project.org/api/TNG300-1/snapshots/99/subhalos/' + str(i) + '/sublink/mpb.hdf5'
        mpb_ = get(url)
        mpb = h5py.File(mpb_)
    
        ## z = 1
        sindz1 = np.where(mpb['SnapNum'][:] == 50)
        if len(mpb['SnapNum'][sindz1]) == 0: #checks that subhalo progenitor is tracked to z = 1
            z1_id.append(np.nan)
            z1_pos.append(np.nan)
            z1_iscen.append(np.nan)
        
        else:
            subz1 = mpb['SubfindID'][sindz1][0]
            if subz1 == -1: #confirms that subhalo identified at z = 1
                z1_id.append(np.nan)
                z1_pos.append(np.nan)
                z1_iscen.append(np.nan)
            else:
                indz1 = subhalos_z1['SubhaloGrNr'][subz1] #GroupID for subhalo
                if abs(mpb['SubfindID'][sindz1] - mpb['GroupFirstSub'][sindz1]) > 0: #checks if subhalo central at z = 1
                    iscen1 = 0
                else:
                    iscen1 = 1
                z1_id.append(indz1)
                z1_pos.append(halos_z1['GroupPos'][indz1])
                z1_iscen.append(iscen1)
                if verbose:
                    print(i, "at z = 1", subz1, indz1, iscen1)
        
        ## z = 2
        sindz2 = np.where(mpb['SnapNum'][:] == 33)
        if len(mpb['SnapNum'][sindz2]) == 0: #checks that subhalo progenitor is tracked to z = 2
            z2_id.append(np.nan)
            z2_pos.append(np.nan)
            z2_iscen.append(np.nan)
        
        else:
            subz2 = mpb['SubfindID'][sindz2][0]
            if subz2 == -1: #confirms that subhalo identified at z = 2
                z2_id.append(np.nan)
                z2_pos.append(np.nan)
                z2_iscen.append(np.nan)
            else:
                indz2 = subhalos_z2['SubhaloGrNr'][subz2] #GroupID for subhalo
                if abs(mpb['SubfindID'][sindz2] - mpb['GroupFirstSub'][sindz2]) > 0: #checks if subhalo central at z = 2
                    iscen2 = 0
                else:
                    iscen2 = 1
                z2_id.append(indz2)
                z2_pos.append(halos_z2['GroupPos'][indz2])
                z2_iscen.append(iscen2)
                if verbose:
                    print(i, "at z = 2", subz2, indz2, iscen2)
            
        if plot: #plot Group_M_Crit200/Subfind ID/if central for all snapshots to check
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 5))
            ax1.plot(mpb['SnapNum'][:], (1e10/h)*mpb['Group_M_Crit200'][:], color = 'k')
            ax1.set_yscale('log')
            ax1.axvline(33, ls = ':', color = 'gray')
            ax1.axvline(50, ls = ':', color = 'gray')
            ax1.axhline(1e14, ls = '--', color = 'gray')
            ax1.set_xlabel('SnapNum')
            ax1.set_ylabel('Group_M_Crit200')
            ax1.set_title('SubhaloID (z = 0): %i'%i)
            
            ax2.plot(mpb['SnapNum'][:], mpb['SubfindID'][:], color = 'k', 
                     label = 'Progenitor SubfindID')
            ax2.plot(mpb['SnapNum'][:], mpb['GroupFirstSub'][:], color = 'mediumvioletred', ls = '--', 
                     label = 'Central SubfindID')
            ax2.set_yscale('symlog')
            ax2.legend(loc = 'best')
            ax2.axvline(33, ls = ':', color = 'gray')
            ax2.axvline(50, ls = ':', color = 'gray')
            ax2.set_xlabel('SnapNum')
            ax2.set_ylabel('SubfindID')
            
            ax3.plot(mpb['SnapNum'][:], mpb['SubfindID'][:] - mpb['GroupFirstSub'][:], color = 'k')
            ax3.axvline(33, ls = ':', color = 'gray')
            ax3.axvline(50, ls = ':', color = 'gray')
            ax3.axhline(0, ls = '--', color = 'gray')
            ax3.set_xlabel('SnapNum')
            ax3.set_ylabel('SubfindID - GroupFirstSub')
            
            plt.show()
            
    
        os.remove(mpb_) #delete the mpb file once it's been used to save room
    
    if verbose:
        print(z1_id)
        print(z2_id)
    
    z1 = {'GroupID':z1_id, 'iscen':z1_iscen, 'GroupPos':z1_pos, 'GroupM200':m200[m200 >= 1e14]}
    z2 = {'GroupID':z2_id, 'iscen':z2_iscen, 'GroupPos':z2_pos, 'GroupM200':m200[m200 >= 1e14]}
    
    return z1, z2


def build_HI_cats(verbose = True):
    """
    Make a cluster and galaxy catalog at each of z = 1, 2 with HI masses, stellar masses, ...
    
    The cluster catalog has a yes/no flag for each of mass and radius selection.
    
    The galaxy catalog has a cluster member/non-cluster flag.
    
    As a caution: takes a *long* time to run (on the order of hours).
    """
    
    ## iterating though HI supplementary catalog for easier debugging
    z1_sub_id = [] #subhalo ids for galaxy catalog
    z2_sub_id = []
    
    z1_sub_posx = [] #subhalo positions for galaxy catalog
    z1_sub_posy = [] #separating x,y,z for ease later
    z1_sub_posz = []
    z2_sub_posx = []
    z2_sub_posy = []
    z2_sub_posz = []
    
    z1_sub_mhi = [] #subhalo HI mass for galaxy catalog -- uses GK11 (can be changed)
    z2_sub_mhi = []
    
    z1_sub_mstar = [] #subhalo (observational) stellar mass for galaxy catalog
    z2_sub_mstar = []
    
    z1_incluster = [] #cluster member/non-cluster flag for galaxy catalog
    z2_incluster = []
    
    z1_sub_massselect = [] #for cluster members, whether mass- or radius-selected or both
    z1_sub_radiusselect = []
    z2_sub_massselect = []
    z2_sub_radiusselect = []
    
    z1_halo_id = [] #halo ids for cluster catalog
    z2_halo_id = []
    
    z1_halo_id = [] #halo ids for cluster catalog
    z2_halo_id = []
    
    z1_halo_posx = [] #halo positions for cluster catalog
    z1_halo_posy = [] #separating x,y,z for ease later
    z1_halo_posz = []
    z2_halo_posx = []
    z2_halo_posy = []
    z2_halo_posz = []
    
    z1_halo_mhi = [] #total halo HI mass for cluster catalog -- uses GK11 (can be changed)
    z2_halo_mhi = []
    
    z1_massselect = [] #mass-selected cluster flag
    z2_massselect = []
    
    z1_radiusselect = [] #radius-selected cluster flag
    z2_radiusselect = []
    
    
    # z = 1
    for i in np.unique(z1_h['id_group']):
        
        if verbose:
            print('z = 1: ', i, end = '\r')
        
        if i in cluster_groups_z1:
            i_ = int(i)
            
            z1_halo_id.append(i_)
            z1_halo_posx.append(halos_z1['GroupPos'][i_][0])
            z1_halo_posy.append(halos_z1['GroupPos'][i_][1])
            z1_halo_posz.append(halos_z1['GroupPos'][i_][2])
            
            if (i in list(msz1['GroupID'])) and (i in list(rsz1['GroupID'])):
                massselect1 = 1
                radiusselect1 = 1
            elif i in list(msz1['GroupID']):
                massselect1 = 1
                radiusselect1 = 0
            elif i in list(rsz1['GroupID']):
                massselect1 = 0
                radiusselect1 = 1
            else:
                massselect1 = 0
                radiusselect1 = 0
            
            mhi = []
            for j in np.where(z1_h['id_group'] == i)[0]:
                mhi.append(z1_h['m_hi_GK11_vol'][j])
                
                id_ = int(z1_h['id_subhalo'][j])
                
                z1_sub_id.append(id_)
                z1_sub_posx.append(subhalos_z1['SubhaloPos'][id_][0])
                z1_sub_posy.append(subhalos_z1['SubhaloPos'][id_][1])
                z1_sub_posz.append(subhalos_z1['SubhaloPos'][id_][2])
                z1_sub_mhi.append(z1_h['m_hi_GK11_vol'][j])
                z1_sub_mstar.append((1e10/h)*subhalos_z1['SubhaloStellarPhotometricsMassInRad'][id_])
                z1_incluster.append(1)
                z1_sub_massselect.append(massselect1)
                z1_sub_radiusselect.append(radiusselect1)
            
            z1_halo_mhi.append(np.sum(mhi))
            
            z1_massselect.append(massselect1)
            z1_radiusselect.append(radiusselect1)
                
        else:
            for j in np.where(z1_h['id_group'] == i)[0]:
                id_ = int(z1_h['id_subhalo'][j])
                
                z1_sub_id.append(id_)
                z1_sub_posx.append(subhalos_z1['SubhaloPos'][id_][0])
                z1_sub_posy.append(subhalos_z1['SubhaloPos'][id_][1])
                z1_sub_posz.append(subhalos_z1['SubhaloPos'][id_][2])
                z1_sub_mhi.append(z1_h['m_hi_GK11_vol'][j])
                z1_sub_mstar.append((1e10/h)*subhalos_z1['SubhaloStellarPhotometricsMassInRad'][id_])
                z1_incluster.append(0)
                z1_sub_massselect.append(0)
                z1_sub_radiusselect.append(0)
        
    # z = 2
    for i in np.unique(z2_h['id_group']): 
        if verbose:
            print('z = 2: ', i, end = '\r')
        
        if i in cluster_groups_z2:
            i_ = int(i)
            
            z2_halo_id.append(i_)
            z2_halo_posx.append(halos_z2['GroupPos'][i_][0])
            z2_halo_posy.append(halos_z2['GroupPos'][i_][1])
            z2_halo_posz.append(halos_z2['GroupPos'][i_][2])
            
            if (i in list(msz2['GroupID'])) and (i in list(rsz2['GroupID'])):
                massselect2 = 1
                radiusselect2 = 1
            elif i in list(msz2['GroupID']):
                massselect2 = 1
                radiusselect2 = 0
            elif i in list(rsz2['GroupID']):
                massselect2 = 0
                radiusselect2 = 1
            else:
                massselect2 = 0
                radiusselect2 = 0
            
            mhi = []
            for j in np.where(z2_h['id_group'] == i)[0]:
                mhi.append(z2_h['m_hi_GK11_vol'][j])
                
                id_ = int(z2_h['id_subhalo'][j])
                
                z2_sub_id.append(id_)
                z2_sub_posx.append(subhalos_z2['SubhaloPos'][id_][0])
                z2_sub_posy.append(subhalos_z2['SubhaloPos'][id_][1])
                z2_sub_posz.append(subhalos_z2['SubhaloPos'][id_][2])
                z2_sub_mhi.append(z2_h['m_hi_GK11_vol'][j])
                z2_sub_mstar.append((1e10/h)*subhalos_z2['SubhaloStellarPhotometricsMassInRad'][id_])
                z2_incluster.append(1)
                z2_sub_massselect.append(massselect2)
                z2_sub_radiusselect.append(radiusselect2)
            
            z2_halo_mhi.append(np.sum(mhi))
            
            z2_massselect.append(massselect2)
            z2_radiusselect.append(radiusselect2)
                
        else:
            for j in np.where(z2_h['id_group'] == i)[0]:
                id_ = int(z2_h['id_subhalo'][j])
                
                z2_sub_id.append(id_)
                z2_sub_posx.append(subhalos_z2['SubhaloPos'][id_][0])
                z2_sub_posy.append(subhalos_z2['SubhaloPos'][id_][1])
                z2_sub_posz.append(subhalos_z2['SubhaloPos'][id_][2])
                z2_sub_mhi.append(z2_h['m_hi_GK11_vol'][j])
                z2_sub_mstar.append((1e10/h)*subhalos_z2['SubhaloStellarPhotometricsMassInRad'][id_])
                z2_incluster.append(0)
                z2_sub_massselect.append(0)
                z2_sub_radiusselect.append(0)
    
    gal_z1 = {'SubhaloID':z1_sub_id, 
              'x':z1_sub_posx,
              'y':z1_sub_posy, 
              'z':z1_sub_posz, 
              'sub_mHI':z1_sub_mhi, 
              'sub_mstar':z1_sub_mstar, 
              'incluster':z1_incluster, 
              'mass_select':z1_sub_massselect, 
              'radius_select':z1_sub_radiusselect}
    
    cluster_z1 = {'GroupID':z1_halo_id, 
                  'x':z1_halo_posx, 
                  'y':z1_halo_posy, 
                  'z':z1_halo_posz, 
                  'group_mHI':z1_halo_mhi, 
                  'mass_select':z1_massselect, 
                  'radius_select':z1_radiusselect}
    
    gal_z2 = {'SubhaloID':z2_sub_id, 
              'x':z2_sub_posx,
              'y':z2_sub_posy, 
              'z':z2_sub_posz, 
              'sub_mHI':z2_sub_mhi, 
              'sub_mstar':z2_sub_mstar, 
              'incluster':z2_incluster, 
              'mass_select':z2_sub_massselect, 
              'radius_select':z2_sub_radiusselect}
    
    cluster_z2 = {'GroupID':z2_halo_id, 
                  'x':z2_halo_posx, 
                  'y':z2_halo_posy, 
                  'z':z2_halo_posz, 
                  'group_mHI':z2_halo_mhi, 
                  'mass_select':z2_massselect, 
                  'radius_select':z2_radiusselect}
    
    return gal_z1, cluster_z1, gal_z2, cluster_z2


################################ Mock catalog construction ##################################

msz1, msz2 = check_M200() #creates initial mass-selected cluster catalogs

msz1_ = pd.DataFrame.from_dict(msz1) #writes to csv file (and avoids re-running code that accesses TNG API)
msz1_.to_csv('msz1.txt', sep = '\t', index = False)
msz2_ = pd.DataFrame.from_dict(msz2)
msz2_.to_csv('msz2.txt', sep = '\t', index = False)

rz1 = np.where(halos_z1['Group_R_Crit200'] >= 750)[0] #creates initial radius-selected cluster catalogs
rsz1 = {'GroupID': rs1, 'GroupPos': [np.array(i) for i in halos_z1['GroupPos'][rs1]], 'GroupR200':halos_z1['Group_R_Crit200'][rs1]}
rz2 = np.where(halos_z2['Group_R_Crit200'] >= 750)[0]
rsz2 = {'GroupID': rs2, 'GroupPos': [np.array(i) for i in halos_z2['GroupPos'][rs2]], 'GroupR200':halos_z2['Group_R_Crit200'][rs2]}

rsz1_ = pd.DataFrame.from_dict(rsz1) #reads out to csv file (probably not strictly necessary)
rsz1_.to_csv('rsz1.txt', sep = '\t', index = False)
rsz2_ = pd.DataFrame.from_dict(rsz2)
rsz2_.to_csv('rsz2.txt', sep = '\t', index = False)

cluster_groups_z1 = list(np.unique(np.concatenate((msz1['GroupID'], rsz1['GroupID'])))) #list of group ids associated with a cluster
cluster_groups_z2 = list(np.unique(np.concatenate((msz2['GroupID'], rsz2['GroupID']))))

#define group ids that are not mass- or radius-selected clusters
field_groups_z1 = list(np.unique(z1_h['id_group'])[~np.isin(np.unique(z1_h['id_group']), cluster_groups_z1)]) 
field_groups_z2 = list(np.unique(z2_h['id_group'])[~np.isin(np.unique(z2_h['id_group']), cluster_groups_z2)])


#generate catalogs -- including hydrogen and stellar mass -- for permutation + analysis
gal_z1, cluster_z1, gal_z2, cluster_z2 = build_HI_cats()

gal_z1_ = pd.DataFrame.from_dict(gal_z1) #writes to csv file (to avoid rerunning build_HI_cats())
gal_z1_.to_csv('gal_z1.txt', sep = '\t', index = False)
gal_z2_ = pd.DataFrame.from_dict(gal_z2)
gal_z2_.to_csv('gal_z2.txt', sep = '\t', index = False)

cluster_z1_ = pd.DataFrame.from_dict(cluster_z1)
cluster_z1_.to_csv('cluster_z1.txt', sep = '\t', index = False)
cluster_z2_ = pd.DataFrame.from_dict(cluster_z2)
cluster_z2_.to_csv('cluster_z2.txt', sep = '\t', index = False)
