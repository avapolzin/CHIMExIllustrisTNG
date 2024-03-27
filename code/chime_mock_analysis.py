import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from scipy.signal import convolve
from photutils.aperture import EllipticalAperture, aperture_photometry

################################## constants and utility functions ##################################

h = 0.6774
hi = 1420.405761768 #MHz
freq_width = 0.390 #MHz
fnt = 6.4 #fraction of flux from "non-target" sources
fff = 0.2 #fraction of flux that survives foreground filtering

cosmo = FlatLambdaCDM(H0 = h*100, Om0 = 0.3089)

def get_bandwidth(redshift):
	# ## assumes TNG300 volume size by default
	# ## returned in integer multiples of CHIME channel width
	# a = 1/(1 + redshift)
	# dLbase = cosmo.luminosity_distance(redshift).to(u.Mpc)
	# z = z_at_value(cosmo.luminosity_distance, dLbase+(a*205/h)*u.Mpc)
	# f = hi/(1 + redshift)
	# fnew = hi/(1 + z)
	# bw = abs(fnew - f)
	# return int(bw/freq_width)
	return 1 #uncomment rest of function for outer limit on CHIME bandwidth -- does not include foreground filtering effects

def reverse(arr):
	return 205000 - arr #in ckpc/h


def hi_flux(arr, mHIarr, redshift):

	freq = hi/(1 + redshift)
	dLbase = cosmo.luminosity_distance(redshift).to(u.Mpc).value
	bandwidth = get_bandwidth(redshift)*freq_width

	a = 1/(1 + redshift)

	los = a*(arr)*u.kpc.to(u.Mpc)/h #ckpc/h to physical Mpc

	dL = dLbase + los

	flux_dens = 2.022e-8 * mHIarr * dL**-2 #output in Jy MHz

	return flux_dens/bandwidth #Jy


def single_gauss_0(x, n):
    if z == 1:
        sigma = 1.62
    if z == 2:
        sigma = 2.41
    return n*np.exp(-0.5*((x - 32)/sigma)**2)

def single_gauss_1(x, n):
    if z == 1:
        sigma = 2.73
    if z == 2:
        sigma = 3.08
    return n*np.exp(-0.5*((x - 34)/sigma)**2)


def return_mass(out_stack, img, cat, redshift, verbose = False, nrand = 'cat'):
	#for nrand, give an integer or 'cat', 'cat' uses length of catalog

	global z
	z = redshift  
    
	if redshift == 1:
		normz = 1.29
	if redshift == 2:
		normz = 2.93

	dl = cosmo.luminosity_distance(redshift).to(u.Mpc).value
	bandwidth = get_bandwidth(redshift)*freq_width

	norm = fff*fnt*normz

	if nrand == 'cat':
		nrand = len(cat['EW'])

	rand_cat = {'EW':np.random.uniform(np.min(cat['EW']), np.max(cat['EW']), nrand), 
			'NS':np.random.uniform(np.min(cat['NS']), np.max(cat['NS']), nrand)}

	rand_stack = stack(img, rand_cat, verbose = False)

	norm_stack = out_stack - rand_stack

	n_0, pcov = curve_fit(single_gauss_0, np.arange(norm_stack.shape[1]), norm_stack[34, :])    
	n_1, pcov = curve_fit(single_gauss_1, np.arange(norm_stack.shape[0]), norm_stack[:, 32])
	fpk = np.mean([n_0, n_1]) #all done very explicitly so it can be simplified easily 
    
	flux = fpk*bandwidth

	mass = 4.945e7 * flux * dl**2/norm #mass in Msun

	if not verbose:
		return mass

	if verbose:
		print(fpk, mass)
		fig = plt.figure()
		plt.plot(np.arange(norm_stack.shape[1]), norm_stack[34, :], 
				color = '#579eb2', alpha = 0.5)
		plt.plot(np.arange(norm_stack.shape[1]), single_gauss_0(np.arange(norm_stack.shape[1]), n_0), 
				color = '#579eb2', ls = '--', alpha = 0.5)
		plt.plot(np.arange(norm_stack.shape[0]), norm_stack[:, 32], 
				color = '#e7a834', alpha = 0.5)
		plt.plot(np.arange(norm_stack.shape[0]), single_gauss_1(np.arange(norm_stack.shape[0]), n_1), 
				color = '#e7a834', ls = '--', alpha = 0.5)
		plt.axhline(fpk, color = '#8f8c7b', ls = '--', alpha = 0.8) 
		plt.show()
		plt.close()
		return mass, norm_stack


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


def tile(subtiles, redshift, return_fhi = False):
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

	if not return_fhi:
	tile_cat = {'EW':EW, 'NS':NS}
	if return_fhi:
	tile_cat = {'EW':EW, 'NS':NS, 'F_HI':FHI_out}

	return tile_img, tile_cat


def convolve_beam(img, beam, add_noise = True, sig = 0.6e-3, verbose = True):
	"""
	Convolve map with beam and, if add_noise, add Gaussian noise to image.

	sig: Standard deviation of noise in Jy/pixel; 
			0.6e-3 is measured from CHIME maps (CHIME Collaboration 2023)
	"""

	convolved_img = convolve(img, beam, mode = 'same')

	if add_noise:
		if verbose:
			print('adding noise')
		convolved_img = fff*convolved_img + np.random.normal(loc = 0, scale = sig, size = convolved_img.shape)
		# fff is the approximation of attenuation due to foreground filtering

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

def nsamp_stack(img, cat, n, verbose = True, return_ind = False):
	"""
	Takes in mock CHIME map and catalog of objects on which to stack.

	Returns mean stack on only N samples(6 deg x 6 deg).

	(Strongly recommend verbose = False.)
	"""

	ind = np.random.choice(a = np.arange(len(cat['EW'])), size = n, replace = False) #generate random N indices for 

	new_cat = {'EW':cat['EW'][ind], 'NS':cat['NS'][ind]}

	EW = new_cat['EW'] + 34 #adding 34 and 32 pix pad explicitly (though we could skip this step
	NS = new_cat['NS'] + 32 # and incorporate it later)

	tot = len(new_cat['EW'])

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

	if not return_ind:
		return stack_arr/count
	
	if return_ind:
		return stack_arr/count, ind


######################################################################################################
## The following code is all fairly quick to run.

#load beam for convolution
beam1 = np.load('synthbeam_z1.npy')
beam2 = np.load('synthbeam_z2.npy')

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
np.save('full_map_z1_conv.npy', full_conv1)
np.save('full_map_z1_convpad.npy', full_conv_pad1)


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
np.save('full_map_z2_conv.npy', full_conv2)
np.save('full_map_z2_convpad.npy', full_conv_pad2)

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
			 'sub_mHI': np.array(gal_z1['sub_mHI'])[np.array(gal_z1['mass_select']) == 1]}

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
np.save('mscluster_map_z1_conv.npy', ms_conv1)
np.save('mscluster_map_z1_convpad.npy', ms_conv_pad1)

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
np.save('rscluster_map_z1_conv.npy', rs_conv1)
np.save('rscluster_map_z1_convpad.npy', rs_conv_pad1)

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
np.save('mscluster_map_z2_conv.npy', ms_conv2)
np.save('mscluster_map_z2_convpad.npy', ms_conv_pad2)

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
np.save('rscluster_map_z2_conv.npy', rs_conv2)
np.save('rscluster_map_z2_convpad.npy', rs_conv_pad2)

#stack radius-selected cluster map on radius-selected clusters, z = 2
rs_stack_z2 = stack(rs_conv_pad2, rsclustercat2, verbose = False)
np.save('rs_stack_z2.npy', rs_stack_z2)


#return the accuracy of the inferred mass from the various cluster stacks
masses_out = []
for i in range(1000): #1000 iterations for effects from random background
	ratio = (return_mass(fullcluster_stack_z2, full_conv_pad2, clustercat2, 2, nrand = 'cat'))/np.mean(np.array(cluster_z2['group_mHI']))
	if not np.isfinite(ratio):
		ratio = (return_mass(fullcluster_stack_z2, full_conv_pad2, clustercat2, 2, nrand = 'cat'))/np.mean(np.array(cluster_z2['group_mHI']))
	masses_out.append(ratio)

print(np.mean(masses_out), np.std(masses_out))


masses_out = []
for i in range(1000): #1000 iterations for effects from random background
	ratio = (return_mass(fullcluster_stack_z1, full_conv_pad1, clustercat1, 1, nrand = 'cat'))/np.mean(np.array(cluster_z1['group_mHI']))
	if not np.isfinite(ratio):
		ratio = (return_mass(fullcluster_stack_z1, full_conv_pad1, clustercat1, 1, nrand = 'cat'))/np.mean(np.array(cluster_z1['group_mHI']))
	masses_out.append(ratio)

print(np.mean(masses_out), np.std(masses_out))


masses_out = []
for i in range(1000): #1000 iterations for effects from random background
	ratio = (return_mass(fullms_stack_z2, full_conv_pad2, msclustercat2, 2, nrand = 'cat'))/np.mean(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['mass_select']) == 1])
	if not np.isfinite(ratio):
		ratio = (return_mass(fullms_stack_z2, full_conv_pad2, msclustercat2, 2, nrand = 'cat'))/np.mean(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['mass_select']) == 1])
	masses_out.append(ratio)

print(np.mean(masses_out), np.std(masses_out))


masses_out = []
for i in range(1000): #1000 iterations for effects from random background
	ratio = (return_mass(fullms_stack_z1, full_conv_pad1, msclustercat1, 1, nrand = 'cat'))/np.mean(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['mass_select']) == 1])
	if not np.isfinite(ratio):
		ratio = (return_mass(fullms_stack_z1, full_conv_pad1, msclustercat1, 1, nrand = 'cat'))/np.mean(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['mass_select']) == 1])
	masses_out.append(ratio)

print(np.mean(masses_out), np.std(masses_out))


masses_out = []
for i in range(1000): #1000 iterations for effects from random background
	ratio = (return_mass(fullrs_stack_z2, full_conv_pad2, rsclustercat2, 2, nrand = 'cat'))/np.mean(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['radius_select']) == 1])
	if not np.isfinite(ratio):
		ratio = (return_mass(fullrs_stack_z2, full_conv_pad2, rsclustercat2, 2, nrand = 'cat'))/np.mean(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['radius_select']) == 1])
	masses_out.append(ratio)

print(np.mean(masses_out), np.std(masses_out))


masses_out = []
for i in range(1000): #1000 iterations for effects from random background
	ratio = (return_mass(fullrs_stack_z1, full_conv_pad1, rsclustercat1, 1, nrand = 'cat'))/np.mean(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['radius_select']) == 1])
	if not np.isfinite(ratio):
		ratio = (return_mass(fullrs_stack_z1, full_conv_pad1, rsclustercat1, 1, nrand = 'cat'))/np.mean(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['radius_select']) == 1])
	masses_out.append(ratio)

print(np.mean(masses_out), np.std(masses_out))


#return the accuracy of the inferred mass from various stacks with N samples
nsamps_all = [3000, 1000, 300, 100, 30]

for nsamps in nsamps_all:
	mass_out = []
	for i in range(1000): #1000 iterations for effects from random background AND random catalog
		samp_stack, ind = nsamp_stack(full_conv_pad2, clustercat2, n = nsamps, verbose = False, return_ind = True)
		cond = create_condition_cat(cluster_z2, 'group_mHI')
		
		ratio = (return_mass(samp_stack, full_conv_pad2, clustercat2, 2, nrand = nsamps))/np.mean(np.array(cond)[ind])
		if not np.isfinite(ratio):
			ratio = (return_mass(samp_stack, full_conv_pad2, clustercat2, 2, nrand = nsamps))/np.mean(np.array(cond)[ind])
		mass_out.append(ratio)

	print(np.mean(mass_out), np.std(mass_out))


for nsamps in nsamps_all:
	mass_out = []
	for i in range(1000): #1000 iterations for effects from random background AND random catalog
		samp_stack, ind = nsamp_stack(full_conv_pad1, clustercat1, n = nsamps, verbose = False, return_ind = True)
		cond = create_condition_cat(cluster_z1, 'group_mHI')
		
		ratio = (return_mass(samp_stack, full_conv_pad1, clustercat1, 1, nrand = nsamps))/np.mean(np.array(cond)[ind])
		if not np.isfinite(ratio):
			ratio = (return_mass(samp_stack, full_conv_pad1, clustercat1, 1, nrand = nsamps))/np.mean(np.array(cond)[ind])
		mass_out.append(ratio)

	print(np.mean(mass_out), np.std(mass_out))


for nsamps in nsamps_all:
	mass_out = []
	for i in range(1000): #1000 iterations for effects from random background AND random catalog
		samp_stack, ind = nsamp_stack(full_conv_pad2, msclustercat2, n = nsamps, verbose = False, return_ind = True)
		cond = create_condition_cat(cluster_z2, 'group_mHI')
		cond2 = create_condition_cat(cluster_z2, 'mass_select')
		cond = np.array(cond)[cond2 == 1]
		
		ratio = (return_mass(samp_stack, full_conv_pad2, msclustercat2, 2, nrand = nsamps))/np.mean(np.array(cond)[ind])
		if not np.isfinite(ratio):
			ratio = (return_mass(samp_stack, full_conv_pad2, msclustercat2, 2, nrand = nsamps))/np.mean(np.array(cond)[ind])
		mass_out.append(ratio)

	print(np.mean(mass_out), np.std(mass_out))


for nsamps in nsamps_all:
	mass_out = []
	for i in range(1000): #1000 iterations for effects from random background AND random catalog
		samp_stack, ind = nsamp_stack(full_conv_pad1, msclustercat1, n = nsamps, verbose = False, return_ind = True)
		cond = create_condition_cat(cluster_z1, 'group_mHI')
		cond2 = create_condition_cat(cluster_z1, 'mass_select')
		cond = np.array(cond)[cond2 == 1]
		
		ratio = (return_mass(samp_stack, full_conv_pad1, msclustercat1, 1, nrand = nsamps))/np.mean(np.array(cond)[ind])
		if not np.isfinite(ratio):
			ratio = (return_mass(samp_stack, full_conv_pad1, msclustercat1, 1, nrand = nsamps))/np.mean(np.array(cond)[ind])
		mass_out.append(ratio)

	print(np.mean(mass_out), np.std(mass_out))


for nsamps in nsamps_all:
	mass_out = []
	for i in range(1000): #1000 iterations for effects from random background AND random catalog
		samp_stack, ind = nsamp_stack(full_conv_pad2, rsclustercat2, n = nsamps, verbose = False, return_ind = True)
		cond = create_condition_cat(cluster_z2, 'group_mHI')
		cond2 = create_condition_cat(cluster_z2, 'radius_select')
		cond = np.array(cond)[cond2 == 1]
		
		ratio = (return_mass(samp_stack, full_conv_pad2, rsclustercat2, 2, nrand = nsamps))/np.mean(np.array(cond)[ind])
		if not np.isfinite(ratio):
			ratio = (return_mass(samp_stack, full_conv_pad2, rsclustercat2, 2, nrand = nsamps))/np.mean(np.array(cond)[ind])
		mass_out.append(ratio)

	print(np.mean(mass_out), np.std(mass_out))


for nsamps in nsamps_all:
	mass_out = []
	for i in range(1000): #1000 iterations for effects from random background AND random catalog
		samp_stack, ind = nsamp_stack(full_conv_pad1, rsclustercat1, n = nsamps, verbose = False, return_ind = True)
		cond = create_condition_cat(cluster_z1, 'group_mHI')
		cond2 = create_condition_cat(cluster_z1, 'radius_select')
		cond = np.array(cond)[cond2 == 1]
		
		ratio = (return_mass(samp_stack, full_conv_pad1, rsclustercat1, 1, nrand = nsamps))/np.mean(np.array(cond)[ind])
		if not np.isfinite(ratio):
			ratio = (return_mass(samp_stack, full_conv_pad1, rsclustercat1, 1, nrand = nsamps))/np.mean(np.array(cond)[ind])
		mass_out.append(ratio)

	print(np.mean(mass_out), np.std(mass_out))






######################################################################################################
## ancillary code -- used to enable other analysis, includes 
## 	(1) measuring new beam FWHM and (2) computing fnt from noiseless stacks
## only re-run if required for alternative cluster-selection or beam

#compute beam FWHM (redshift-dependent, and, depending on beam, direction-dependent)
def single_gauss_0_fwhm(x, sigma, n):
    return n*np.exp(-0.5*((x - 32)/sigma)**2)

def single_gauss_1_fwhm(x, sigma, n):
    return n*np.exp(-0.5*((x - 34)/sigma)**2)


def return_fwhm(out_stack, img, cat, redshift, verbose = False, nrand = 'cat'):
	#for nrand, give an integer or 'cat', 'cat' uses length of catalog

	if nrand == 'cat':
		nrand = len(cat['EW'])

	rand_cat = {'EW':np.random.uniform(np.min(cat['EW']), np.max(cat['EW']), nrand), 
			'NS':np.random.uniform(np.min(cat['NS']), np.max(cat['NS']), nrand)}

	rand_stack = stack(img, rand_cat, verbose = False)

	norm_stack = out_stack - rand_stack

	sig_0, n_0, pcov = curve_fit(single_gauss_0_fwhm, np.arange(norm_stack.shape[1]), norm_stack[34, :])    
	sig_1, n_1, pcov = curve_fit(single_gauss_1_fwhm, np.arange(norm_stack.shape[0]), norm_stack[:, 32])

	if not verbose:
		return sig_0, sig_1

	if verbose:
		print(fpk, mass)
		fig = plt.figure()
		plt.plot(np.arange(norm_stack.shape[1]), norm_stack[34, :], 
				color = '#579eb2', alpha = 0.5)
		plt.plot(np.arange(norm_stack.shape[1]), single_gauss_0(np.arange(norm_stack.shape[1]), n_0), 
				color = '#579eb2', ls = '--', alpha = 0.5)
		plt.plot(np.arange(norm_stack.shape[0]), norm_stack[:, 32], 
				color = '#e7a834', alpha = 0.5)
		plt.plot(np.arange(norm_stack.shape[0]), single_gauss_1(np.arange(norm_stack.shape[0]), n_1), 
				color = '#e7a834', ls = '--', alpha = 0.5)
		plt.axhline(fpk, color = '#8f8c7b', ls = '--', alpha = 0.8) 
		plt.show()
		plt.close()
		return sig_0, sig_1

#redshift 2 FWHM
sig_0_, sig_1_ = [], []
for i in range(1000):
	print(i, end = '\r')
	sig_0, sig_1 = return_fwhm(fullcluster_stack_z2, full_conv_pad2, clustercat2, 2, nrand = 'cat', verbose = False)
	sig_0_.append(sig_0)
	sig_1_.append(sig_1)
print(np.mean(sig_0_), np.std(sig_0_)) 
print(np.mean(sig_1_), np.std(sig_1_)) 

#redshift 1 FWHM
sig_0_, sig_1_ = [], []
for i in range(1000):
	print(i, end = '\r')
	sig_0, sig_1 = return_fwhm(fullcluster_stack_z1, full_conv_pad1, clustercat1, 1, nrand = 'cat', verbose = False)
	sig_0_.append(sig_0)
	sig_1_.append(sig_1)
print(np.mean(sig_0_), np.std(sig_0_)) 
print(np.mean(sig_1_), np.std(sig_1_))


######################################


#generate noisless maps + stacks
full_conv1_noiseless = convolve_beam(tileimg1, beam1, add_noise = False) #convolve tiled image with beam
full_conv_pad1_noiseless = make_stack_map(full_conv1_noiseless) #pad beam-convolved map to allow stacking to edges

full_conv2_noiseless = convolve_beam(tileimg2, beam2, add_noise = False) #convolve tiled image with beam
full_conv_pad2_noiseless = make_stack_map(full_conv2_noiseless) #pad beam-convolved map to allow stacking to edges 

ms_conv1_noiseless = convolve_beam(tileimg1_ms, beam1, add_noise = False) #convolve tiled image with beam
ms_conv_pad1_noiseless = make_stack_map(ms_conv1_noiseless) #pad beam-convolved map to allow stacking to edges
rs_conv1_noiseless = convolve_beam(tileimg1_rs, beam1, add_noise = False) #convolve tiled image with beam
rs_conv_pad1_noiseless = make_stack_map(rs_conv1_noiseless) #pad beam-convolved map to allow stacking to edges

ms_conv2_noiseless = convolve_beam(tileimg2_ms, beam2, add_noise = False) #convolve tiled image with beam
ms_conv_pad2_noiseless = make_stack_map(ms_conv2_noiseless) #pad beam-convolved map to allow stacking to edges
rs_conv2_noiseless = convolve_beam(tileimg2_rs, beam2, add_noise = False) #convolve tiled image with beam
rs_conv_pad2_noiseless = make_stack_map(rs_conv2_noiseless) #pad beam-convolved map to allow stacking to edges

fullms_stack_z1_noiseless = stack(full_conv_pad1_noiseless, msclustercat1, verbose = False)
fullrs_stack_z1_noiseless = stack(full_conv_pad1_noiseless, rsclustercat1, verbose = False)

fullms_stack_z2_noiseless = stack(full_conv_pad2_noiseless, msclustercat2, verbose = False)
fullrs_stack_z2_noiseless = stack(full_conv_pad2_noiseless, rsclustercat2, verbose = False)

ms_stack_z1_noiseless = stack(ms_conv_pad1_noiseless, msclustercat1, verbose = False)
rs_stack_z1_noiseless = stack(rs_conv_pad1_noiseless, rsclustercat1, verbose = False)

ms_stack_z2_noiseless = stack(ms_conv_pad2_noiseless, msclustercat2, verbose = False)
rs_stack_z2_noiseless = stack(rs_conv_pad2_noiseless, rsclustercat2, verbose = False)

#create background-subtracted stacks
def norm_stack(out_stack, img, cat, redshift, verbose = False, nrand = 1000):

    dl = cosmo.luminosity_distance(redshift).to(u.Mpc).value
        
    bandwidth = get_bandwidth(redshift)*freq_width
    
    if nrand == 'cat':
        nrand = len(cat) 
    
    rand_cat = {'EW':np.random.uniform(np.min(cat['EW']), np.max(cat['EW']), nrand), 
            'NS':np.random.uniform(np.min(cat['NS']), np.max(cat['NS']), nrand)}

    rand_stack = stack(img, rand_cat, verbose = False)
    
    norm_stack = out_stack - rand_stack
    
    return norm_stack

#recover ratio of flux in full map stacks and cluster-only stacks
rs1__ = []
rs2__ = []
ms1__ = []
ms2__ = []
for i in range(1000):
    print(i, end = '\r')
    rsfull_norm_z1 = norm_stack(fullrs_stack_z1_noiseless, full_conv_pad1_noiseless, rsclustercat1, 1)
    rs_norm_z1 = norm_stack(rs_stack_z1_noiseless, rs_conv_pad1_noiseless, rsclustercat1, 1)

    msfull_norm_z1 = norm_stack(fullms_stack_z1_noiseless, full_conv_pad1_noiseless, msclustercat1, 1)
    ms_norm_z1 = norm_stack(ms_stack_z1_noiseless, ms_conv_pad1_noiseless, msclustercat1, 1)
    
    rsfull_norm_z2 = norm_stack(fullrs_stack_z2_noiseless, full_conv_pad2_noiseless, rsclustercat2, 2)
    rs_norm_z2 = norm_stack(rs_stack_z2_noiseless, rs_conv_pad2_noiseless, rsclustercat2, 2)

    msfull_norm_z2 = norm_stack(fullms_stack_z2_noiseless, full_conv_pad2_noiseless, msclustercat2, 2)
    ms_norm_z2 = norm_stack(ms_stack_z2_noiseless, ms_conv_pad2_noiseless, msclustercat2, 2)
    
    rs1__.append(np.max(rsfull_norm_z1)/np.max(rs_norm_z1))
    rs2__.append(np.max(rsfull_norm_z2)/np.max(rs_norm_z2)*(1.53/3.48)) #correcting for relative beam size
    
    ms1__.append(np.max(msfull_norm_z1)/np.max(ms_norm_z1))
    ms2__.append(np.max(msfull_norm_z2)/np.max(ms_norm_z2)*(1.53/3.48)) #correcting for relative beam size
    
    
print('rs, z = 1', np.mean(rs1__), np.std(rs1__))
print('rs, z = 2', np.mean(rs2__), np.std(rs2__))

print('ms, z = 1', np.mean(ms1__), np.std(ms1__))
print('ms, z = 2', np.mean(ms2__), np.std(ms2__))

print('rs, fnt', 0.5*(np.mean(rs1__) + np.mean(rs2__)), np.sqrt((0.5*np.std(rs1__))**2 + (0.5*np.std(rs2__))**2)) #radius-selected
print('ms, fnt', 0.5*(np.mean(ms1__) + np.mean(ms2__)), np.sqrt((0.5*np.std(ms1__))**2 + (0.5*np.std(ms2__))**2)) #mass-selected
print('fnt', 0.25*(np.mean(rs1__) + np.mean(rs2__) + np.mean(ms1__) + np.mean(ms2__)), 
		np.sqrt((0.25*np.std(rs1__))**2 + (0.25*np.std(rs2__))**2 + (0.25*np.std(ms1__))**2 + (0.25*np.std(ms2__))**2)) #total



