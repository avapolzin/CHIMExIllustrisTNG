import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM
from scipy.signal import convolve
from photutils.aperture import EllipticalAperture, aperture_photometry

################################## constants and utility functions ##################################

h = 0.6774
nu1 = 714 #MHz
nu2 = 476 #MHz

cosmo = FlatLambdaCDM(H0 = 67.74, Om0 = 0.3089)

def reverse(arr):
	return 205000 - arr #in ckpc/h


def hi_flux(arr, mHIarr, redshift):
	if redshift == 1:
		dLbase = 6.8e3 #Mpc
		freq = nu1 #MHz

	if redshift == 2:
		dLbase = 1.59e4 #Mpc
		freq = nu2 #MHz

	a = 1/(1 + redshift)

	los = a*(arr)*u.kpc.to(u.Mpc)/0.6774 #ckpc/h to physical Mpc

	dL = dLbase + los

	return (2.022e-8 * mHIarr * dL**-2)*freq #output in Jy


def stack_to_mass(stack, redshift):
	if redshift == 1:
		dLbase = 6.8e3 #Mpc
		freq = nu1 #MHz
		a = 2.75 #pix, based on FWHM
		b = 2.25

	if redshift == 2:
		dLbase = 1.59e4 #Mpc
		freq = nu2 #MHz
		a = 3.75
		b = 3

	aper = EllipticalAperture((65/2 - 0.5, 69/2 - 0.5), a = a, b = b, theta = np.deg2rad(90))
	flux = aperture_photometry(stack, aper)['aperture_sum']/aper.area #summed flux in Jy

	return (flux * dLbase**2)/(freq * 2.022e-8) #mass in Msun

def create_condition_cat(cat, field):
	"""
	Create tile-length array of location-independent fields 
		for cuts on cluster membership/stellar mass/...

	Takes in the catalog of interest and the column name (str).

	Returns array that can be used to index/filter the larger catalogs.
	"""

	subtile_out = list(cat[field]) * 48
	tile_out = subtile_out * 8

	return np.array(tile_out)


def make_bins(redshift):
	"""
	Convert absolute ckpc/h positions to CHIME-like RA and ZA bins.

	Bins are fed to plt.hist2d() at tiling stage. 
	"""

	RA = (205000/h) * 24
	ZA = (205000/h) * 16

	RA_bins = np.arange(RA/(cosmo.kpc_comoving_per_arcmin(redshift).value*5.3))
	ZA_bins = np.arange(ZA/(cosmo.kpc_comoving_per_arcmin(redshift).value*5.7))

	return RA_bins, ZA_bins

def make_subtile(xarr, yarr, zarr, mHI, redshift):
	"""
	Makes subtiles based on subhalos TNG300 volume.

	Output values are location of object in pix, FHI.

	(Comments are just to keep track of different permutations.)
	"""

	EW = []
	NS = []
	FHI = []


	## -x-y(-z)
	EW.append(reverse(xarr))
	NS.append(reverse(yarr))
	FHI.append(hi_flux(reverse(zarr), mHI, redshift))

	## x-y(-z)
	EW.append(xarr)
	NS.append(205000 + reverse(yarr))
	FHI.append(hi_flux(reverse(zarr), mHI, redshift))

	## -xy(-z)
	EW.append(reverse(xarr))
	NS.append(2*205000 + yarr)
	FHI.append(hi_flux(reverse(zarr), mHI, redshift))

	## -x-y(z)
	EW.append(reverse(xarr))
	NS.append(3*205000 + reverse(yarr))
	FHI.append(hi_flux(zarr, mHI, redshift))

	## xy(-z)
	EW.append(xarr)
	NS.append(4*205000 + yarr)
	FHI.append(hi_flux(reverse(zarr), mHI, redshift))

	## x-y(z)
	EW.append(xarr)
	NS.append(5*205000 + reverse(yarr))
	FHI.append(hi_flux(zarr, mHI, redshift))

	## -xy(z)
	EW.append(reverse(xarr))
	NS.append(6*205000 + yarr)
	FHI.append(hi_flux(zarr, mHI, redshift))

	## xy(z)
	EW.append(xarr)
	NS.append(7*205000 + yarr)
	FHI.append(hi_flux(zarr, mHI, redshift))


	## -y-x(-z)
	EW.append(205000 + reverse(yarr))
	NS.append(reverse(xarr))
	FHI.append(hi_flux(reverse(zarr), mHI, redshift))

	## -yx(-z)
	EW.append(205000 + reverse(yarr))
	NS.append(205000 + xarr)
	FHI.append(hi_flux(reverse(zarr), mHI, redshift))

	## y-x(-z)
	EW.append(205000 + yarr)
	NS.append(2*205000 + reverse(xarr))
	FHI.append(hi_flux(reverse(zarr), mHI, redshift))

	## -y-x(z)
	EW.append(205000 + reverse(yarr))
	NS.append(3*205000 + reverse(xarr))
	FHI.append(hi_flux(zarr, mHI, redshift))

	## yx(-z)
	EW.append(205000 + yarr)
	NS.append(4*205000 + xarr)
	FHI.append(hi_flux(reverse(zarr), mHI, redshift))

	## -yx(z)
	EW.append(205000 + reverse(yarr))
	NS.append(5*205000 + xarr)
	FHI.append(hi_flux(zarr, mHI, redshift))

	## y-x(z)
	EW.append(205000 + yarr)
	NS.append(6*205000 + reverse(xarr))
	FHI.append(hi_flux(zarr, mHI, redshift))

	## yx(z)
	EW.append(205000 + yarr)
	NS.append(7*205000 + xarr)
	FHI.append(hi_flux(zarr, mHI, redshift))


	## -x-z(-y)
	EW.append(2*205000 + reverse(xarr))
	NS.append(reverse(zarr))
	FHI.append(hi_flux(reverse(yarr), mHI, redshift))

	## x-z(-y)
	EW.append(2*205000 + xarr)
	NS.append(205000 + reverse(zarr))
	FHI.append(hi_flux(reverse(yarr), mHI, redshift))

	## -x-z(y)
	EW.append(2*205000 + reverse(xarr))
	NS.append(2*205000 + reverse(zarr))
	FHI.append(hi_flux(yarr, mHI, redshift))

	## -xz(-y)
	EW.append(2*205000 + reverse(xarr))
	NS.append(3*205000 + zarr)
	FHI.append(hi_flux(reverse(yarr), mHI, redshift))

	## x-z(y)
	EW.append(2*205000 + xarr)
	NS.append(4*205000 + reverse(zarr))
	FHI.append(hi_flux(yarr, mHI, redshift))

	## xz(-y)
	EW.append(2*205000 + xarr)
	NS.append(5*205000 + zarr)
	FHI.append(hi_flux(reverse(yarr), mHI, redshift))

	## -xz(y)
	EW.append(2*205000 + reverse(xarr))
	NS.append(6*205000 + zarr)
	FHI.append(hi_flux(yarr, mHI, redshift))

	## xz(y)
	EW.append(2*205000 + xarr)
	NS.append(7*205000 + zarr)
	FHI.append(hi_flux(yarr, mHI, redshift))


	## -z-x(-y)
	EW.append(3*205000 + reverse(zarr))
	NS.append(reverse(xarr))
	FHI.append(hi_flux(reverse(yarr), mHI, redshift))

	## -zx(-y)
	EW.append(3*205000 + reverse(zarr))
	NS.append(205000 + xarr)
	FHI.append(hi_flux(reverse(yarr), mHI, redshift))

	## -z-x(y)
	EW.append(3*205000 + reverse(zarr))
	NS.append(2*205000 + reverse(xarr))
	FHI.append(hi_flux(yarr, mHI, redshift))

	## z-x(-y)
	EW.append(3*205000 + zarr)
	NS.append(3*205000 + reverse(xarr))
	FHI.append(hi_flux(reverse(yarr), mHI, redshift))

	## -zx(y)
	EW.append(3*205000 + reverse(zarr))
	NS.append(4*205000 + xarr)
	FHI.append(hi_flux(yarr, mHI, redshift))

	## zx(-y)
	EW.append(3*205000 + zarr)
	NS.append(5*205000 + xarr)
	FHI.append(hi_flux(reverse(yarr), mHI, redshift))

	## z-x(y)
	EW.append(3*205000 + zarr)
	NS.append(6*205000 + reverse(xarr))
	FHI.append(hi_flux(yarr, mHI, redshift))

	## zx(y)
	EW.append(3*205000 + zarr)
	NS.append(7*205000 + xarr)
	FHI.append(hi_flux(yarr, mHI, redshift))


	## -y-z(-x)
	EW.append(4*205000 + reverse(yarr))
	NS.append(reverse(zarr))
	FHI.append(hi_flux(reverse(xarr), mHI, redshift))

	## -y-z(x)
	EW.append(4*205000 + reverse(yarr))
	NS.append(205000 + reverse(zarr))
	FHI.append(hi_flux(xarr, mHI, redshift))

	## y-z(-x)
	EW.append(4*205000 + yarr)
	NS.append(2*205000 + reverse(zarr))
	FHI.append(hi_flux(reverse(xarr), mHI, redshift))

	## -yz(-x)
	EW.append(4*205000 + reverse(yarr))
	NS.append(3*205000 + zarr)
	FHI.append(hi_flux(reverse(xarr), mHI, redshift))

	## y-z(x)
	EW.append(4*205000 + yarr)
	NS.append(4*205000 + reverse(zarr))
	FHI.append(hi_flux(xarr, mHI, redshift))

	## -yz(x)
	EW.append(4*205000 + reverse(yarr))
	NS.append(5*205000 + zarr)
	FHI.append(hi_flux(xarr, mHI, redshift))

	## yz(-x)
	EW.append(4*205000 + yarr)
	NS.append(6*205000 + zarr)
	FHI.append(hi_flux(reverse(xarr), mHI, redshift))

	## yzx
	EW.append(4*205000 + yarr)
	NS.append(7*205000 + zarr)
	FHI.append(hi_flux(xarr, mHI, redshift))


	## -z-y(-x)
	EW.append(5*205000 + reverse(zarr))
	NS.append(reverse(yarr))
	FHI.append(hi_flux(reverse(xarr), mHI, redshift))

	## -z-y(x)
	EW.append(5*205000 + reverse(zarr))
	NS.append(205000 + reverse(yarr))
	FHI.append(hi_flux(xarr, mHI, redshift))

	## -zy(-x)
	EW.append(5*205000 + reverse(zarr))
	NS.append(2*205000 + yarr)
	FHI.append(hi_flux(reverse(xarr), mHI, redshift))

	## z-y(-x)
	EW.append(5*205000 + zarr)
	NS.append(3*205000 + reverse(yarr))
	FHI.append(hi_flux(reverse(xarr), mHI, redshift))

	## -zy(x)
	EW.append(5*205000 + reverse(zarr))
	NS.append(4*205000 + yarr)
	FHI.append(hi_flux(xarr, mHI, redshift))

	## z-y(x)
	EW.append(5*205000 + zarr)
	NS.append(5*205000 + reverse(yarr))
	FHI.append(hi_flux(xarr, mHI, redshift))

	## zy(-x)
	EW.append(5*205000 + zarr)
	NS.append(6*205000 + yarr)
	FHI.append(hi_flux(reverse(xarr), mHI, redshift))

	## zy(x)
	EW.append(5*205000 + zarr)
	NS.append(7*205000 + yarr)
	FHI.append(hi_flux(xarr, mHI, redshift))

	subtile = {'EW': np.array(EW).ravel(), 'NS': np.array(NS).ravel(), 'F_HI':np.array(FHI).ravel()}

	return subtile


def tile(subtiles, redshift):
	"""
	Takes redshift and list of subtiles generated by make_subtile()

	Returns tiled and binned intensity map, ready for convolution, and catalog of coordinates.

	Tiles the following way:
	st1	st2	st3	st4
	st5	st6	st7	st8
	"""

	ud = 8*205000 #in ckpc/h
	lr = 6*205000 #in ckpc/h

	EW_out = np.concatenate((st1['EW'], lr+st2['EW'], 2*lr+st3['EW'], 3*lr+st4['EW'], 
							 st5['EW'], lr+st6['EW'], 2*lr+st7['EW'], 3*lr+st8['EW']))
	NS_out = np.concatenate((ud+st1['NS'], ud+st2['NS'], ud+st3['NS'], ud+st4['NS'], 
							 st5['NS'], st6['NS'], st7['NS'], st8['NS']))
	FHI_out = np.concatenate((st1['F_HI'], st2['F_HI'], st3['F_HI'], st4['F_HI'], 
							  st5['F_HI'], st6['F_HI'], st7['F_HI'], st8['F_HI']))

	EW = EW_out/(h*cosmo.kpc_comoving_per_arcmin(redshift).value*5.3)
	NS = NS_out/(h*cosmo.kpc_comoving_per_arcmin(redshift).value*5.7)
	RA_bins, ZA_bins = make_bins(redshift)
	tile_img, x, y = np.histogram2d(EW, NS, bins = [RA_bins, ZA_bins], weights = FHI_out)

	tile_cat = {'EW':EW, 'NS':NS}

	return tile_img, tile_cat


def convolve_beam(img, beam):

	convolved_img = convolve(img, beam, mode = 'same')

	return convolved_img


def make_stack_map(img, verbose = False):
	"""
	Pad beam-convolved intensity map to be able to stack on objects within 3 deg of the map edges.

	Will transform object locations accordingly in stacking step.

	Takes in intensity map (optionally, if verbose = True, prints values used to determine padding.)
	"""

	if verbose:
		ra_pad = np.deg2rad(3)/np.deg2rad(5.3/60)
		za_pad = np.deg2rad(3)/np.deg2rad(5.7/60)

		print('Adding 3 deg buffer 34 and 32 pix respectively to accomodate %.2f pix in RA and %.2f pix in ZA.'%(ra_pad, za_pad))

	#pad with 0 rather than NaN to make subsequent stacking easier
	return np.pad(img, pad_width = ((34, 34), (32, 32)), mode = 'constant', constant_values = 0)


def stack(img, cat, verbose = True):
	"""
	Takes in mock CHIME map and catalog of objects on which to stack.

	Returns mean stack (6 deg x 6 deg).

	(Strongly recommend verbose = False.)
	"""

	EW = cat['EW'] + 34 #adding 34 and 32 pix pad explicitly (though we could skip this step
	NS = cat['NS'] + 32 # and incorporate it later)

	tot = len(cat['EW'])

	stack_arr = np.zeros((69, 65))
	count = 0
	for i in range(tot):
		center = (int(EW[i]), int(NS[i]))
		lower_ra = center[0] - 34
		upper_ra = center[0] + 34 + 1 #add 1 explicitly for indexing
		lower_za = center[1] - 32
		upper_za = center[1] + 32 + 1 #add 1 explicitly for indexing
        
		obj = img[lower_ra:upper_ra, lower_za:upper_za]
		if (upper_ra > img.shape[0]) or (upper_za > img.shape[1]):
			continue #since image has been cropped to CHIME-like pixel scale, but catalog has not
		stack_arr += obj
		count += 1

		if verbose:
			print('completed: %i/%i'%(i+1, tot), end = '\r')
			
	return stack_arr/count

def mean_stack_mass(img, cat, mHIarr):

	EW = cat['EW'] + 34 #adding 34 and 32 pix pad explicitly (though we could skip this step
	NS = cat['NS'] + 32 # and incorporate it later)

	tot = len(cat['EW'])

	stack_mass = 0
	count = 0
	for i in range(tot):
		center = (int(EW[i]), int(NS[i]))
		lower_ra = center[0] - 34
		upper_ra = center[0] + 34 + 1 #add 1 explicitly for indexing
		lower_za = center[1] - 32
		upper_za = center[1] + 32 + 1 #add 1 explicitly for indexing
        
		obj = mHIarr[i]
		if (upper_ra > img.shape[0]) or (upper_za > img.shape[1]):
			continue #since image has been cropped to CHIME-like pixel scale, but catalog has not
		stack_mass += obj
		count += 1

		if verbose:
			print('completed: %i/%i'%(i+1, tot), end = '\r')
			
	return stack_mass/count


######################################################################################################
## The following code is all fairly quick to run.

#load beam for convolution
beam1 = np.load('synthbeam_z1.npy') #not included in repository
beam2 = np.load('synthbeam_z2.npy') #not included in repository

#load catalogs created in chime_mock_catalog.py
cluster_z1 = pd.read_csv('../catalogs/prepermutation/cluster_z1.txt', sep = '\t', header = 0)
cluster_z2 = pd.read_csv('../catalogs/prepermutation/cluster_z2.txt', sep = '\t', header = 0)
gal_z1 = pd.read_csv('../catalogs/prepermutation/gal_z1.txt', sep = '\t', header = 0)
gal_z2 = pd.read_csv('../catalogs/prepermutation/gal_z2.txt', sep = '\t', header = 0)


#make full map, z = 1 -- this step is necessary bc output catalog is GBs
st1 = make_subtile(np.array(gal_z1['x']), np.array(gal_z1['y']), np.array(gal_z1['z']), np.array(gal_z1['sub_mHI']), 1)
st2 = make_subtile(reverse(np.array(gal_z1['x'])), np.array(gal_z1['y']), np.array(gal_z1['z']), np.array(gal_z1['sub_mHI']), 1)
st3 = make_subtile(np.array(gal_z1['x']), reverse(np.array(gal_z1['y'])), np.array(gal_z1['z']), np.array(gal_z1['sub_mHI']), 1)
st4 = make_subtile(np.array(gal_z1['x']), np.array(gal_z1['y']), reverse(np.array(gal_z1['z'])), np.array(gal_z1['sub_mHI']), 1)
st5 = make_subtile(reverse(np.array(gal_z1['x'])), reverse(np.array(gal_z1['y'])), np.array(gal_z1['z']), np.array(gal_z1['sub_mHI']), 1)
st6 = make_subtile(reverse(np.array(gal_z1['x'])), np.array(gal_z1['y']), reverse(np.array(gal_z1['z'])), np.array(gal_z1['sub_mHI']), 1)
st7 = make_subtile(np.array(gal_z1['x']), reverse(np.array(gal_z1['y'])), reverse(np.array(gal_z1['z'])), np.array(gal_z1['sub_mHI']), 1)
st8 = make_subtile(reverse(np.array(gal_z1['x'])), reverse(np.array(gal_z1['y'])), reverse(np.array(gal_z1['z'])), np.array(gal_z1['sub_mHI']), 1)
tileimg1, tilecat1 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 1)
np.save('full_map_z1.npy', tileimg1)
full_conv1 = convolve_beam(tileimg1, beam1) #convolve tiled image with beam
full_conv_pad1 = make_stack_map(full_conv1) #pad beam-convolved map to allow stacking to edges

#stack full map on all galaxies with Mstar >= 1e10 Msun
mstar = create_condition_cat(gal_z1, 'sub_mstar')
crop_cat = {'EW':tilecat1['EW'][mstar >= 1e10], 'NS':tilecat1['NS'][mstar >= 1e10]}
full_stack_z1 = stack(full_conv_pad1, crop_cat, verbose = False)
np.save('full_stack_z1.npy', full_stack_z1)

#stack full map on all non-cluster galaxies with Mstar >= 1e10 Msun
incluster = create_condition_cat(gal_z1, 'incluster')
crop_cat = {'EW':tilecat1['EW'][(mstar >= 1e10) & (incluster == 0)], 'NS':tilecat1['NS'][(mstar >= 1e10) & (incluster == 0)]}
nc_stack_z1 = stack(full_conv_pad1, crop_cat, verbose = False)
np.save('fullnc_stack_z1.npy', nc_stack_z1)

#stack full map on all cluster galaxies with Mstar >= 1e10 Msun
crop_cat = {'EW':tilecat1['EW'][(mstar >= 1e10) & (incluster == 1)], 'NS':tilecat1['NS'][(mstar >= 1e10) & (incluster == 1)]}
c_stack_z1 = stack(full_conv_pad1, crop_cat, verbose = False)
np.save('fullc_stack_z1.npy', c_stack_z1)


#make full map, z = 2
st1 = make_subtile(np.array(gal_z2['x']), np.array(gal_z2['y']), np.array(gal_z2['z']), np.array(gal_z2['sub_mHI']), 2)
st2 = make_subtile(reverse(np.array(gal_z2['x'])), np.array(gal_z2['y']), np.array(gal_z2['z']), np.array(gal_z2['sub_mHI']), 2)
st3 = make_subtile(np.array(gal_z2['x']), reverse(np.array(gal_z2['y'])), np.array(gal_z2['z']), np.array(gal_z2['sub_mHI']), 2)
st4 = make_subtile(np.array(gal_z2['x']), np.array(gal_z2['y']), reverse(np.array(gal_z2['z'])), np.array(gal_z2['sub_mHI']), 2)
st5 = make_subtile(reverse(np.array(gal_z2['x'])), reverse(np.array(gal_z2['y'])), np.array(gal_z2['z']), np.array(gal_z2['sub_mHI']), 2)
st6 = make_subtile(reverse(np.array(gal_z2['x'])), np.array(gal_z2['y']), reverse(np.array(gal_z2['z'])), np.array(gal_z2['sub_mHI']), 2)
st7 = make_subtile(np.array(gal_z2['x']), reverse(np.array(gal_z2['y'])), reverse(np.array(gal_z2['z'])), np.array(gal_z2['sub_mHI']), 2)
st8 = make_subtile(reverse(np.array(gal_z2['x'])), reverse(np.array(gal_z2['y'])), reverse(np.array(gal_z2['z'])), np.array(gal_z2['sub_mHI']), 2)
tileimg2, tilecat2 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 2)
np.save('full_map_z2.npy', tileimg2)
full_conv2 = convolve_beam(tileimg2, beam2) #convolve tiled image with beam
full_conv_pad2 = make_stack_map(full_conv2) #pad beam-convolved map to allow stacking to edges

#stack full map on all galaxies with Mstar >= 1e10 Msun
mstar = create_condition_cat(gal_z2, 'sub_mstar')
crop_cat = {'EW':tilecat2['EW'][mstar >= 1e10], 'NS':tilecat2['NS'][mstar >= 1e10]}
full_stack_z2 = stack(full_conv_pad2, crop_cat, verbose = False)
np.save('full_stack_z2.npy', full_stack_z2)

#stack full map on all non-cluster galaxies with Mstar >= 1e10 Msun
incluster = create_condition_cat(gal_z2, 'incluster')
crop_cat = {'EW':tilecat2['EW'][(mstar >= 1e10) & (incluster == 0)], 'NS':tilecat2['NS'][(mstar >= 1e10) & (incluster == 0)]}
nc_stack_z2 = stack(full_conv_pad2, crop_cat, verbose = False)
np.save('fullnc_stack_z2.npy', nc_stack_z2)

#stack full map on all cluster galaxies with Mstar >= 1e10 Msun
crop_cat = {'EW':tilecat2['EW'][(mstar >= 1e10) & (incluster == 1)], 'NS':tilecat2['NS'][(mstar >= 1e10) & (incluster == 1)]}
c_stack_z2 = stack(full_conv_pad2, crop_cat, verbose = False)
np.save('fullc_stack_z2.npy', c_stack_z2)


#make all cluster catalog, z = 1
st1 = make_subtile(np.array(cluster_z1['x']), np.array(cluster_z1['y']), np.array(cluster_z1['z']), np.array(cluster_z1['group_mHI']), 1)
st2 = make_subtile(reverse(np.array(cluster_z1['x'])), np.array(cluster_z1['y']), np.array(cluster_z1['z']), np.array(cluster_z1['group_mHI']), 1)
st3 = make_subtile(np.array(cluster_z1['x']), reverse(np.array(cluster_z1['y'])), np.array(cluster_z1['z']), np.array(cluster_z1['group_mHI']), 1)
st4 = make_subtile(np.array(cluster_z1['x']), np.array(cluster_z1['y']), reverse(np.array(cluster_z1['z'])), np.array(cluster_z1['group_mHI']), 1)
st5 = make_subtile(reverse(np.array(cluster_z1['x'])), reverse(np.array(cluster_z1['y'])), np.array(cluster_z1['z']), np.array(cluster_z1['group_mHI']), 1)
st6 = make_subtile(reverse(np.array(cluster_z1['x'])), np.array(cluster_z1['y']), reverse(np.array(cluster_z1['z'])), np.array(cluster_z1['group_mHI']), 1)
st7 = make_subtile(np.array(cluster_z1['x']), reverse(np.array(cluster_z1['y'])), reverse(np.array(cluster_z1['z'])), np.array(cluster_z1['group_mHI']), 1)
st8 = make_subtile(reverse(np.array(cluster_z1['x'])), reverse(np.array(cluster_z1['y'])), reverse(np.array(cluster_z1['z'])), np.array(cluster_z1['group_mHI']), 1)
tileimg1_clusters, clustercat1 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 1)

#stack full map on all clusters, z = 1
fullcluster_stack_z1 = stack(full_conv_pad1, clustercat1, verbose = False)
np.save('fullcluster_stack_z1.npy', fullcluster_stack_z1)


#make all cluster catalog, z = 2
st1 = make_subtile(np.array(cluster_z2['x']), np.array(cluster_z2['y']), np.array(cluster_z2['z']), np.array(cluster_z2['group_mHI']), 2)
st2 = make_subtile(reverse(np.array(cluster_z2['x'])), np.array(cluster_z2['y']), np.array(cluster_z2['z']), np.array(cluster_z2['group_mHI']), 2)
st3 = make_subtile(np.array(cluster_z2['x']), reverse(np.array(cluster_z2['y'])), np.array(cluster_z2['z']), np.array(cluster_z2['group_mHI']), 2)
st4 = make_subtile(np.array(cluster_z2['x']), np.array(cluster_z2['y']), reverse(np.array(cluster_z2['z'])), np.array(cluster_z2['group_mHI']), 2)
st5 = make_subtile(reverse(np.array(cluster_z2['x'])), reverse(np.array(cluster_z2['y'])), np.array(cluster_z2['z']), np.array(cluster_z2['group_mHI']), 2)
st6 = make_subtile(reverse(np.array(cluster_z2['x'])), np.array(cluster_z2['y']), reverse(np.array(cluster_z2['z'])), np.array(cluster_z2['group_mHI']), 2)
st7 = make_subtile(np.array(cluster_z2['x']), reverse(np.array(cluster_z2['y'])), reverse(np.array(cluster_z2['z'])), np.array(cluster_z2['group_mHI']), 2)
st8 = make_subtile(reverse(np.array(cluster_z2['x'])), reverse(np.array(cluster_z2['y'])), reverse(np.array(cluster_z2['z'])), np.array(cluster_z2['group_mHI']), 2)
tileimg2_clusters, clustercat2 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 2)

#stack full map on all clusters, z = 2
fullcluster_stack_z2 = stack(full_conv_pad2, clustercat2, verbose = False)
np.save('fullcluster_stack_z2.npy', fullcluster_stack_z2)


#make mass-selected cluster catalog, z = 1
mscluster1 = {'x':np.array(cluster_z1['x'])[np.array(cluster_z1['mass_select']) == 1], 
             'y':np.array(cluster_z1['y'])[np.array(cluster_z1['mass_select']) == 1], 
             'z':np.array(cluster_z1['z'])[np.array(cluster_z1['mass_select']) == 1], 
             'group_mHI': np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['mass_select']) == 1]}

st1 = make_subtile(mscluster1['x'], mscluster1['y'], mscluster1['z'], mscluster1['group_mHI'], 1)
st2 = make_subtile(reverse(mscluster1['x']), mscluster1['y'], mscluster1['z'], mscluster1['group_mHI'], 1)
st3 = make_subtile(mscluster1['x'], reverse(mscluster1['y']), mscluster1['z'], mscluster1['group_mHI'], 1)
st4 = make_subtile(mscluster1['x'], mscluster1['y'], reverse(mscluster1['z']), mscluster1['group_mHI'], 1)
st5 = make_subtile(reverse(mscluster1['x']), reverse(mscluster1['y']), mscluster1['z'], mscluster1['group_mHI'], 1)
st6 = make_subtile(reverse(mscluster1['x']), mscluster1['y'], reverse(mscluster1['z']), mscluster1['group_mHI'], 1)
st7 = make_subtile(mscluster1['x'], reverse(mscluster1['y']), reverse(mscluster1['z']), mscluster1['group_mHI'], 1)
st8 = make_subtile(reverse(mscluster1['x']), reverse(mscluster1['y']), reverse(mscluster1['z']), mscluster1['group_mHI'], 1)
tileimg1_msclusters, msclustercat1 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 1)

#stack full map on mass-selected galaxy clusters, z = 1
fullms_stack_z1 = stack(full_conv_pad1, msclustercat1, verbose = False)
np.save('fullms_stack_z1.npy', fullms_stack_z1)


#make mass-selected cluster catalog, z = 2
mscluster2 = {'x':np.array(cluster_z2['x'])[np.array(cluster_z2['mass_select']) == 1], 
             'y':np.array(cluster_z2['y'])[np.array(cluster_z2['mass_select']) == 1], 
             'z':np.array(cluster_z2['z'])[np.array(cluster_z2['mass_select']) == 1], 
             'group_mHI': np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['mass_select']) == 1]}

st1 = make_subtile(mscluster2['x'], mscluster2['y'], mscluster2['z'], mscluster2['group_mHI'], 2)
st2 = make_subtile(reverse(mscluster2['x']), mscluster2['y'], mscluster2['z'], mscluster2['group_mHI'], 2)
st3 = make_subtile(mscluster2['x'], reverse(mscluster2['y']), mscluster2['z'], mscluster2['group_mHI'], 2)
st4 = make_subtile(mscluster2['x'], mscluster2['y'], reverse(mscluster2['z']), mscluster2['group_mHI'], 2)
st5 = make_subtile(reverse(mscluster2['x']), reverse(mscluster2['y']), mscluster2['z'], mscluster2['group_mHI'], 2)
st6 = make_subtile(reverse(mscluster2['x']), mscluster2['y'], reverse(mscluster2['z']), mscluster2['group_mHI'], 2)
st7 = make_subtile(mscluster2['x'], reverse(mscluster2['y']), reverse(mscluster2['z']), mscluster2['group_mHI'], 2)
st8 = make_subtile(reverse(mscluster2['x']), reverse(mscluster2['y']), reverse(mscluster2['z']), mscluster2['group_mHI'], 2)
tileimg2_msclusters, msclustercat2 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 2)

#stack full map on mass-selected galaxy clusters, z = 2
fullms_stack_z2 = stack(full_conv_pad2, msclustercat2, verbose = False)
np.save('fullms_stack_z2.npy', fullms_stack_z2)


#make radius-selected cluster catalog, z = 1
rscluster1 = {'x':np.array(cluster_z1['x'])[np.array(cluster_z1['radius_select']) == 1], 
             'y':np.array(cluster_z1['y'])[np.array(cluster_z1['radius_select']) == 1], 
             'z':np.array(cluster_z1['z'])[np.array(cluster_z1['radius_select']) == 1], 
             'group_mHI': np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['radius_select']) == 1]}

st1 = make_subtile(rscluster1['x'], rscluster1['y'], rscluster1['z'], rscluster1['group_mHI'], 1)
st2 = make_subtile(reverse(rscluster1['x']), rscluster1['y'], rscluster1['z'], rscluster1['group_mHI'], 1)
st3 = make_subtile(rscluster1['x'], reverse(rscluster1['y']), rscluster1['z'], rscluster1['group_mHI'], 1)
st4 = make_subtile(rscluster1['x'], rscluster1['y'], reverse(rscluster1['z']), rscluster1['group_mHI'], 1)
st5 = make_subtile(reverse(rscluster1['x']), reverse(rscluster1['y']), rscluster1['z'], rscluster1['group_mHI'], 1)
st6 = make_subtile(reverse(rscluster1['x']), rscluster1['y'], reverse(rscluster1['z']), rscluster1['group_mHI'], 1)
st7 = make_subtile(rscluster1['x'], reverse(rscluster1['y']), reverse(rscluster1['z']), rscluster1['group_mHI'], 1)
st8 = make_subtile(reverse(rscluster1['x']), reverse(rscluster1['y']), reverse(rscluster1['z']), rscluster1['group_mHI'], 1)
tileimg1_rsclusters, rsclustercat1 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 1)

#stack full map on radius-selected galaxy clusters, z = 1
fullrs_stack_z1 = stack(full_conv_pad1, rsclustercat1, verbose = False)
np.save('fullrs_stack_z1.npy', fullrs_stack_z1)


#make radius-selected cluster catalog, z = 2
rscluster2 = {'x':np.array(cluster_z2['x'])[np.array(cluster_z2['radius_select']) == 1], 
             'y':np.array(cluster_z2['y'])[np.array(cluster_z2['radius_select']) == 1], 
             'z':np.array(cluster_z2['z'])[np.array(cluster_z2['radius_select']) == 1], 
             'group_mHI': np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['radius_select']) == 1]}

st1 = make_subtile(rscluster2['x'], rscluster2['y'], rscluster2['z'], rscluster2['group_mHI'], 2)
st2 = make_subtile(reverse(rscluster2['x']), rscluster2['y'], rscluster2['z'], rscluster2['group_mHI'], 2)
st3 = make_subtile(rscluster2['x'], reverse(rscluster2['y']), rscluster2['z'], rscluster2['group_mHI'], 2)
st4 = make_subtile(rscluster2['x'], rscluster2['y'], reverse(rscluster2['z']), rscluster2['group_mHI'], 2)
st5 = make_subtile(reverse(rscluster2['x']), reverse(rscluster2['y']), rscluster2['z'], rscluster2['group_mHI'], 2)
st6 = make_subtile(reverse(rscluster2['x']), rscluster2['y'], reverse(rscluster2['z']), rscluster2['group_mHI'], 2)
st7 = make_subtile(rscluster2['x'], reverse(rscluster2['y']), reverse(rscluster2['z']), rscluster2['group_mHI'], 2)
st8 = make_subtile(reverse(rscluster2['x']), reverse(rscluster2['y']), reverse(rscluster2['z']), rscluster2['group_mHI'], 2)
tileimg2_rsclusters, rsclustercat2 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 2)

#stack full map on radius-selected galaxy clusters, z = 2
fullrs_stack_z2 = stack(full_conv_pad2, rsclustercat2, verbose = False)
np.save('fullrs_stack_z2.npy', fullrs_stack_z2)


#make mass-selected cluster map, z = 1
mscluster_subs1 = {'x':np.array(gal_z1['x'])[np.array(gal_z1['mass_select']) == 1], 
             'y':np.array(gal_z1['y'])[np.array(gal_z1['mass_select']) == 1], 
             'z':np.array(gal_z1['z'])[np.array(gal_z1['mass_select']) == 1], 
             'group_mHI': np.array(gal_z1['sub_mHI'])[np.array(gal_z1['mass_select']) == 1]}

st1 = make_subtile(mscluster_subs1['x'], mscluster_subs1['y'], mscluster_subs1['z'], mscluster_subs1['sub_mHI'], 1)
st2 = make_subtile(reverse(mscluster_subs1['x']), mscluster_subs1['y'], mscluster_subs1['z'], mscluster_subs1['sub_mHI'], 1)
st3 = make_subtile(mscluster_subs1['x'], reverse(mscluster_subs1['y']), mscluster_subs1['z'], mscluster_subs1['sub_mHI'], 1)
st4 = make_subtile(mscluster_subs1['x'], mscluster_subs1['y'], reverse(mscluster_subs1['z']), mscluster_subs1['sub_mHI'], 1)
st5 = make_subtile(reverse(mscluster_subs1['x']), reverse(mscluster_subs1['y']), mscluster_subs1['z'], mscluster_subs1['sub_mHI'], 1)
st6 = make_subtile(reverse(mscluster_subs1['x']), mscluster_subs1['y'], reverse(mscluster_subs1['z']), mscluster_subs1['sub_mHI'], 1)
st7 = make_subtile(mscluster_subs1['x'], reverse(mscluster_subs1['y']), reverse(mscluster_subs1['z']), mscluster_subs1['sub_mHI'], 1)
st8 = make_subtile(reverse(mscluster_subs1['x']), reverse(mscluster_subs1['y']), reverse(mscluster_subs1['z']), mscluster_subs1['sub_mHI'], 1)
tileimg1_ms, mscat1 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 1)
np.save('mscluster_map_z1.npy', tileimg1_ms)
ms_conv1 = convolve_beam(tileimg1_ms, beam1) #convolve tiled image with beam
ms_conv_pad1 = make_stack_map(ms_conv1) #pad beam-convolved map to allow stacking to edges

#stack mass-selected cluster map on mass-selected clusters, z = 1
ms_stack_z1 = stack(ms_conv_pad1, msclustercat1, verbose = False)
np.save('ms_stack_z1.npy', ms_stack_z1)


#make radius-selected cluster map, z = 1
rscluster_subs1 = {'x':np.array(gal_z1['x'])[np.array(gal_z1['radius_select']) == 1], 
             'y':np.array(gal_z1['y'])[np.array(gal_z1['radius_select']) == 1], 
             'z':np.array(gal_z1['z'])[np.array(gal_z1['radius_select']) == 1], 
             'sub_mHI': np.array(gal_z1['sub_mHI'])[np.array(gal_z1['radius_select']) == 1]}

st1 = make_subtile(rscluster_subs1['x'], rscluster_subs1['y'], rscluster_subs1['z'], rscluster_subs1['sub_mHI'], 1)
st2 = make_subtile(reverse(rscluster_subs1['x']), rscluster_subs1['y'], rscluster_subs1['z'], rscluster_subs1['sub_mHI'], 1)
st3 = make_subtile(rscluster_subs1['x'], reverse(rscluster_subs1['y']), rscluster_subs1['z'], rscluster_subs1['sub_mHI'], 1)
st4 = make_subtile(rscluster_subs1['x'], rscluster_subs1['y'], reverse(rscluster_subs1['z']), rscluster_subs1['sub_mHI'], 1)
st5 = make_subtile(reverse(rscluster_subs1['x']), reverse(rscluster_subs1['y']), rscluster_subs1['z'], rscluster_subs1['sub_mHI'], 1)
st6 = make_subtile(reverse(rscluster_subs1['x']), rscluster_subs1['y'], reverse(rscluster_subs1['z']), rscluster_subs1['sub_mHI'], 1)
st7 = make_subtile(rscluster_subs1['x'], reverse(rscluster_subs1['y']), reverse(rscluster_subs1['z']), rscluster_subs1['sub_mHI'], 1)
st8 = make_subtile(reverse(rscluster_subs1['x']), reverse(rscluster_subs1['y']), reverse(rscluster_subs1['z']), rscluster_subs1['sub_mHI'], 1)
tileimg1_rs, rscat1 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 1)
np.save('rscluster_map_z1.npy', tileimg1_rs)
rs_conv1 = convolve_beam(tileimg1_rs, beam1) #convolve tiled image with beam
rs_conv_pad1 = make_stack_map(rs_conv1) #pad beam-convolved map to allow stacking to edges

#stack radius-selected cluster map on radius-selected clusters, z = 1
rs_stack_z1 = stack(rs_conv_pad1, rsclustercat1, verbose = False)
np.save('rs_stack_z1.npy', rs_stack_z1)


#make mass-selected cluster map, z = 2
mscluster_subs2 = {'x':np.array(gal_z2['x'])[np.array(gal_z2['mass_select']) == 1], 
             'y':np.array(gal_z2['y'])[np.array(gal_z2['mass_select']) == 1], 
             'z':np.array(gal_z2['z'])[np.array(gal_z2['mass_select']) == 1], 
             'sub_mHI': np.array(gal_z2['sub_mHI'])[np.array(gal_z2['mass_select']) == 1]}

st1 = make_subtile(mscluster_subs2['x'], mscluster_subs2['y'], mscluster_subs2['z'], mscluster_subs2['sub_mHI'], 2)
st2 = make_subtile(reverse(mscluster_subs2['x']), mscluster_subs2['y'], mscluster_subs2['z'], mscluster_subs2['sub_mHI'], 2)
st3 = make_subtile(mscluster_subs2['x'], reverse(mscluster_subs2['y']), mscluster_subs2['z'], mscluster_subs2['sub_mHI'], 2)
st4 = make_subtile(mscluster_subs2['x'], mscluster_subs2['y'], reverse(mscluster_subs2['z']), mscluster_subs2['sub_mHI'], 2)
st5 = make_subtile(reverse(mscluster_subs2['x']), reverse(mscluster_subs2['y']), mscluster_subs2['z'], mscluster_subs2['sub_mHI'], 2)
st6 = make_subtile(reverse(mscluster_subs2['x']), mscluster_subs2['y'], reverse(mscluster_subs2['z']), mscluster_subs2['sub_mHI'], 2)
st7 = make_subtile(mscluster_subs2['x'], reverse(mscluster_subs2['y']), reverse(mscluster_subs2['z']), mscluster_subs2['sub_mHI'], 2)
st8 = make_subtile(reverse(mscluster_subs2['x']), reverse(mscluster_subs2['y']), reverse(mscluster_subs2['z']), mscluster_subs2['sub_mHI'], 2)
tileimg2_ms, mscat2 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 2)
np.save('mscluster_map_z2.npy', tileimg2_ms)
ms_conv2 = convolve_beam(tileimg2_ms, beam2) #convolve tiled image with beam
ms_conv_pad2 = make_stack_map(ms_conv2) #pad beam-convolved map to allow stacking to edges

#stack mass-selected cluster map on mass-selected clusters, z = 2
ms_stack_z2 = stack(ms_conv_pad2, msclustercat2, verbose = False)
np.save('ms_stack_z2.npy', ms_stack_z2)


#make radius-selected cluster map, z = 2
rscluster_subs2 = {'x':np.array(gal_z2['x'])[np.array(gal_z2['radius_select']) == 1], 
             'y':np.array(gal_z2['y'])[np.array(gal_z2['radius_select']) == 1], 
             'z':np.array(gal_z2['z'])[np.array(gal_z2['radius_select']) == 1], 
             'sub_mHI': np.array(gal_z2['sub_mHI'])[np.array(gal_z2['radius_select']) == 1]}

st1 = make_subtile(rscluster_subs2['x'], rscluster_subs2['y'], rscluster_subs2['z'], rscluster_subs2['sub_mHI'], 2)
st2 = make_subtile(reverse(rscluster_subs2['x']), rscluster_subs2['y'], rscluster_subs2['z'], rscluster_subs2['sub_mHI'], 2)
st3 = make_subtile(rscluster_subs2['x'], reverse(rscluster_subs2['y']), rscluster_subs2['z'], rscluster_subs2['sub_mHI'], 2)
st4 = make_subtile(rscluster_subs2['x'], rscluster_subs2['y'], reverse(rscluster_subs2['z']), rscluster_subs2['sub_mHI'], 2)
st5 = make_subtile(reverse(rscluster_subs2['x']), reverse(rscluster_subs2['y']), rscluster_subs2['z'], rscluster_subs2['sub_mHI'], 2)
st6 = make_subtile(reverse(rscluster_subs2['x']), rscluster_subs2['y'], reverse(rscluster_subs2['z']), rscluster_subs2['sub_mHI'], 2)
st7 = make_subtile(rscluster_subs2['x'], reverse(rscluster_subs2['y']), reverse(rscluster_subs2['z']), rscluster_subs2['sub_mHI'], 2)
st8 = make_subtile(reverse(rscluster_subs2['x']), reverse(rscluster_subs2['y']), reverse(rscluster_subs2['z']), rscluster_subs2['sub_mHI'], 2)
tileimg2_rs, rscat2 = tile([st1, st2, st3, st4, st5, st6, st7, st8], 2)
np.save('rscluster_map_z2.npy', tileimg2_rs)
rs_conv2 = convolve_beam(tileimg2_rs, beam2) #convolve tiled image with beam
rs_conv_pad2 = make_stack_map(rs_conv2) #pad beam-convolved map to allow stacking to edges

#stack radius-selected cluster map on radius-selected clusters, z = 2
rs_stack_z2 = stack(rs_conv_pad2, rsclustercat2, verbose = False)
np.save('rs_stack_z2.npy', rs_stack_z2)


#try to return the mean HI mass associated with each cluster stack
print(np.log10(stack_to_mass(ms_stack_z2, 2)), np.log10(stack_to_mass(ms_stack_z1, 1)))
print(np.log10(stack_to_mass(ms_stack_z2 - np.median(ms_stack_z2), 2)), np.log10(stack_to_mass(ms_stack_z1 - np.median(ms_stack_z1), 1)))
print(np.log10(stack_to_mass(fullms_stack_z2, full_stack_z2, 2)), np.log10(stack_to_mass(fullms_stack_z1 - full_stack_z1, 1)))
print(np.log10(stack_to_mass(rs_stack_z2, 2)), np.log10(stack_to_mass(rs_stack_z1, 1)))
print(np.log10(stack_to_mass(rs_stack_z2 - np.median(rs_stack_z2), 2)), np.log10(stack_to_mass(rs_stack_z1 - np.median(rs_stack_z1), 1)))
print(np.log10(stack_to_mass(fullms_stack_z2, full_stack_z2, 2)), np.log10(stack_to_mass(fullms_stack_z1 - full_stack_z1, 1)))
print(stack_to_mass(rs_stack_z2 - np.median(rs_stack_z2), 2)/stack_to_mass(ms_stack_z2 - np.median(ms_stack_z2), 2), 
	stack_to_mass(rs_stack_z1 - np.median(rs_stack_z1), 1)/stack_to_mass(ms_stack_z1 - np.median(ms_stack_z1), 1))






