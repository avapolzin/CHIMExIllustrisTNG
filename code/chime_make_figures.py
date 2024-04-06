import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import albumpl
from albumpl.cmap import RhumbLine, Winter05

from astropy.io import fits
import h5py

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['xtick.minor.size'] = 5
matplotlib.rcParams['xtick.minor.width'] = 1.5
matplotlib.rcParams['ytick.minor.size'] = 5
matplotlib.rcParams['ytick.minor.width'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 7
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 7
matplotlib.rcParams['ytick.major.width'] = 3

albumpl.set_default('RhumbLine')
ccycle = albumpl.palette.return_colors('RhumbLine')

#not including code for Figure 2 here, since 1D beam models are not being shared
#also not including Figure A1 which is a schematic produced with Keynote

#EACH FIGURE IS INDEPENDENT AND RELIES ON THE CODE IN chime_mock_analysis.py
#EXCEPT FIGURE 1, WHICH NEEDS THREE FILES NOT INCLUDED IN REPO (CODE AT END)

################################## Figure 3 ##################################

def figure3():

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (26, 8), gridspec_kw = {'wspace':0.04}, sharey=True)

	fhi_out = tilecat1['F_HI'][(tilecat1['EW'] < 231) & (tilecat1['NS'] < 215)]

	ax1.hist2d(tilecat1['EW'][(tilecat1['EW'] < 231) & (tilecat1['NS'] < 215)], 
	           tilecat1['NS'][(tilecat1['EW'] < 231) & (tilecat1['NS'] < 215)], 
	           weights = fhi_out/np.max(fhi_out), bins = 2000, cmap = Winter05(), vmin = 0, vmax = 0.75)
	ax1.set_title('TNG300')
	ax1.set_xticks(ra_inc, ra_arc)
	ax1.set_xlabel(r'$\Delta$RA (deg)')
	ax1.set_yticks(za_inc, za_arc)
	ax1.set_ylabel(r'$\Delta$ZA (deg)')

	ax2.imshow(tileimg1[:215, :231]/np.max(tileimg1[:215, :231]), cmap = Winter05(), vmin = 0, vmax = 1, 
	           aspect = 'auto', origin = 'lower')
	ax2.set_title('binned to CHIME resolution')
	ax2.set_xticks(ra_inc, ra_arc)
	ax2.set_xlabel(r'$\Delta$RA (deg)')

	ax3.imshow(full_conv1[:215, :231]/np.max(full_conv1[:215, :231]), cmap = RhumbLine('r'), vmin = -0.4, vmax = 0.4, 
	           aspect = 'auto', origin = 'lower')
	ax3.set_title('beam-convolved, noise added')
	ax3.set_xticks(ra_inc, ra_arc)
	ax3.set_xlabel(r'$\Delta$RA (deg)')

	left, bottom, width, height = [0.62, 0.25, 0.2, 0.2]
	ax4 = fig.add_axes([left, bottom, width, height])
	ax4.imshow(full_conv1[42:63, 46:68]/np.max(full_conv1[42:63, 46:68]), cmap = RhumbLine('r'), vmin = -0.4, vmax = 0.4, 
	           origin = 'lower', aspect = 'equal')
	ax4.add_patch(Ellipse((10.5, 21/2 - 0.5), 3, 4, facecolor = ccycle[0], edgecolor = ccycle[0]))
	ax4.set_xticks([0, 10, 20], [1, 2, 3], fontsize = 14)
	ax4.set_xlabel(r'$\Delta$RA (deg)', fontsize = 14)
	ax4.set_yticks([0, 10, 20], [1, 2, 3], fontsize = 14)
	ax4.set_ylabel(r'$\Delta$ZA (deg)', fontsize = 14)

	fig.savefig('processing_example_z1_v2.png', bbox_inches = 'tight')



################################## Figure 4 ##################################

def figure4():

	rand_cat = {'EW':np.random.uniform(np.min(tilecat1['EW']), np.max(tilecat1['EW']), 2*10**7), 
	            'NS':np.random.uniform(np.min(tilecat1['NS']), np.max(tilecat1['NS']), 2*10**7)}

	rand_stack = stack(full_conv_pad1, rand_cat, verbose = False)


	fig, ((ax1, ax2, ax3, ax4),
	     (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize = (20, 10), sharey = True, gridspec_kw = {'wspace':0.07})

	cbar = ax1.imshow(full_stack_z1*1e6, aspect = 'equal', origin = 'lower', cmap = RhumbLine('r'), vmin = -0.3/20*1500, vmax = 0.3/20*1500)
	ax1.set_ylabel('ZA (deg)')
	ax1.set_xticks([0, 65/2 - 0.5, 64], [-3, 0, 3])
	ax1.set_yticks([0, 69/2 - 0.5, 68], [-3, 0, 3])
	ax1.set_title('all galaxies\nlog N = 7.26', fontsize = 22)

	ax2.imshow(nc_stack_z1*1e6, aspect = 'equal', origin = 'lower', cmap = RhumbLine('r'), vmin = -0.3/20*1500, vmax = 0.3/20*1500)
	ax2.set_xticks([0, 65/2 - 0.5, 64], [-3, 0, 3])
	ax2.set_title('field galaxies\nlog N = 7.23', fontsize = 22)

	ax3.imshow(c_stack_z1*1e6, aspect = 'equal', origin = 'lower', cmap = RhumbLine('r'), vmin = -0.3/20*1500, vmax = 0.3/20*1500)
	ax3.set_xticks([0, 65/2 - 0.5, 64], [-3, 0, 3])
	ax3.set_title('cluster galaxies\nlog N = 6.11', fontsize = 22)

	ax4.imshow(rand_stack*1e6, aspect = 'equal', origin = 'lower', cmap = RhumbLine('r'), vmin = -0.3/20*1500, vmax = 0.3/20*1500)
	ax4.set_xticks([0, 65/2 - 0.5, 64], [-3, 0, 3])
	ax4.set_title('random positions\nlog N = 7.30', fontsize = 22)


	rand_cat = {'EW':np.random.uniform(np.min(tilecat1['EW']), np.max(tilecat1['EW']), 5*10**4), 
	            'NS':np.random.uniform(np.min(tilecat1['NS']), np.max(tilecat1['NS']), 5*10**4)}

	rand_stack = stack(full_conv_pad1, rand_cat, verbose = False)

	ax5.imshow(full_stack_z1_40k*1e6, aspect = 'equal', origin = 'lower', cmap = RhumbLine('r'), vmin = -0.3/20*1500, vmax = 0.3/20*1500)
	ax5.set_xlabel('RA (deg)')
	ax5.set_ylabel('ZA (deg)')
	ax5.set_xticks([0, 65/2 - 0.5, 64], [-3, 0, 3])
	ax5.set_yticks([0, 69/2 - 0.5, 68], [-3, 0, 3])
	ax5.set_title('log N = 4.67', fontsize = 22)

	ax6.imshow(nc_stack_z1_40k*1e6, aspect = 'equal', origin = 'lower', cmap = RhumbLine('r'), vmin = -0.3/20*1500, vmax = 0.3/20*1500)
	ax6.set_xlabel('RA (deg)')
	ax6.set_xticks([0, 65/2 - 0.5, 64], [-3, 0, 3])
	ax6.set_title('log N = 4.64', fontsize = 22)

	ax7.imshow(c_stack_z1_40k*1e6, aspect = 'equal', origin = 'lower', cmap = RhumbLine('r'), vmin = -0.3/20*1500, vmax = 0.3/20*1500)
	ax7.set_xlabel('RA (deg)')
	ax7.set_xticks([0, 65/2 - 0.5, 64], [-3, 0, 3])
	ax7.set_title('log N = 3.53', fontsize = 22)

	ax8.imshow(rand_stack*1e6, aspect = 'equal', origin = 'lower', cmap = RhumbLine('r'), vmin = -0.3/20*1500, vmax = 0.3/20*1500)
	ax8.set_xlabel('RA (deg)')
	ax8.set_xticks([0, 65/2 - 0.5, 64], [-3, 0, 3])
	ax8.set_title('log N = 4.70', fontsize = 22)

	plt.colorbar(cbar, ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], label = r'Average Flux ($\mu$Jy)', pad = 0.01) 

	fig.savefig('stack_example_z1.png', bbox_inches = 'tight')


################################## Figure 5 ##################################

def figure5():

	#stack on number of galaxies consistent with existing CHIME results
	mstar = create_condition_cat(gal_z1, 'sub_mstar')
	crop_cat = {'EW':tilecat1['EW'][mstar >= 1e10], 'NS':tilecat1['NS'][mstar >= 1e10]}
	full_stack_z1_40k = nsamp_stack(full_conv_pad1, crop_cat, 47253, verbose = False, return_ind = False)


	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (20, 14), sharey = 'row', gridspec_kw = {'wspace':0.05})

	ax1.set_title('z = 2')
	ax1.plot(full_stack_z2[int(69/2 - 0.5), :]*1e6, lw = 4, color = ccycle[0], label = 'all galaxies')
	ax1.plot(nc_stack_z2[int(69/2 - 0.5), :]*1e6, lw = 4, ls = '--', color = ccycle[1], label = 'field galaxies')
	ax1.plot(c_stack_z2[int(69/2 - 0.5), :]*1e6, lw = 4, ls = '--', color = ccycle[3], label = 'cluster galaxies')
	ax1.plot(fullcluster_stack_z2[int(69/2 - 0.5), :]*1e6, lw = 4, color = '#8f8c7b', ls = ':', label = 'all clusters')
	ax1.set_xlabel(r'$\Delta$RA (deg)')
	ax1.set_xticks([0, 64/6, 64/3, 64/2, 2*64/3, 5*64/6, 64], [-3, -2, -1, 0, 1, 2, 3])
	ax1.set_ylabel(r'Flux ($\mu$Jy)')
	ax1.legend(loc = 'upper left', frameon = False)

	ax2.set_title('z = 1')
	ax2.plot(full_stack_z1_40k[int(69/2 - 0.5), :]*1e6, lw = 2, color = ccycle[0], alpha = 0.5, label = r'all galaxies, log N = 4.67')
	ax2.plot(full_stack_z1[int(69/2 - 0.5), :]*1e6, lw = 4, color = ccycle[0])
	ax2.plot(nc_stack_z1[int(69/2 - 0.5), :]*1e6, lw = 4, ls = '--', color = ccycle[1])
	ax2.plot(c_stack_z1[int(69/2 - 0.5), :]*1e6, lw = 4, ls = '--', color = ccycle[3])
	ax2.plot(fullcluster_stack_z1[int(69/2 - 0.5), :]*1e6, lw = 4, color = '#8f8c7b', ls = ':')
	ax2.set_xlabel(r'$\Delta$RA (deg)')
	ax2.set_xticks([0, 64/6, 64/3, 64/2, 2*64/3, 5*64/6, 64], [-3, -2, -1, 0, 1, 2, 3])
	ax2.set_ylim([-18, 140])
	ax2.legend(loc = 'upper right', frameon = False)

	ax3.plot(full_stack_z2[:, int(65/2 - 0.5)]*1e6, lw = 4, color = ccycle[0], label = 'all galaxies')
	ax3.plot(nc_stack_z2[:, int(65/2 - 0.5)]*1e6, lw = 4, ls = '--', color = ccycle[1], label = 'field galaxies')
	ax3.plot(c_stack_z2[:, int(65/2 - 0.5)]*1e6, lw = 4, ls = '--', color = ccycle[3], label = 'cluster galaxies')
	ax3.plot(fullcluster_stack_z2[:, int(65/2 - 0.5)]*1e6, lw = 4, color = '#8f8c7b', ls = ':')
	ax3.set_xlabel(r'$\Delta$ZA (deg)')
	ax3.set_xticks([0, 68/6, 68/3, 68/2, 2*68/3, 5*68/6, 68], [-3, -2, -1, 0, 1, 2, 3])
	ax3.set_ylabel(r'Flux ($\mu$Jy)')

	ax4.plot(full_stack_z1_40k[:, int(65/2 - 0.5)]*1e6, lw = 2, color = ccycle[0], alpha = 0.5)
	ax4.plot(full_stack_z1[:, int(65/2 - 0.5)]*1e6, lw = 4, color = ccycle[0])
	ax4.plot(nc_stack_z1[:, int(65/2 - 0.5)]*1e6, lw = 4, ls = '--', color = ccycle[1])
	ax4.plot(c_stack_z1[:, int(65/2 - 0.5)]*1e6, lw = 4, ls = '--', color = ccycle[3])
	ax4.plot(fullcluster_stack_z1[:, int(65/2 - 0.5)]*1e6, lw = 4, color = '#8f8c7b', ls = ':')
	ax4.set_xlabel(r'$\Delta$ZA (deg)')
	ax4.set_xticks([0, 68/6, 68/3, 68/2, 2*68/3, 5*68/6, 68], [-3, -2, -1, 0, 1, 2, 3])
	ax4.set_ylim([-18, 140])

	fig.savefig('stacked_flux_galaxies_all.png', bbox_inches = 'tight', dpi = 300)


################################## Figure 6 ##################################

def figure6():

	#generate stacks for a realistic number of randomly selected clusters
	fullcluster_stack_z2_300 = nsamp_stack(full_conv_pad2, clustercat2, n = 300, verbose = False, return_ind = False)
	fullcluster_stack_z1_3000 = nsamp_stack(full_conv_pad1, clustercat1, n = 3000, verbose = False, return_ind = False)

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (20, 14), sharey = 'row', gridspec_kw = {'wspace':0.05})

	ax1.set_title('z = 2')
	ax1.plot(fullcluster_stack_z2_300[int(69/2 - 0.5), :]*1e6, lw = 2, color = ccycle[0], alpha = 0.5, label = 'all clusters,\nlog N  = 2.5')
	ax1.plot(fullcluster_stack_z2[int(69/2 - 0.5), :]*1e6, lw = 4, color = ccycle[0])
	ax1.plot(fullms_stack_z2[int(69/2 - 0.5), :]*1e6, lw = 4, ls = '--', color = ccycle[1])
	ax1.plot(fullrs_stack_z2[int(69/2 - 0.5), :]*1e6, lw = 4, ls = '--', color = ccycle[3])
	ax1.set_xlabel(r'$\Delta$RA (deg)')
	ax1.set_xticks([0, 64/6, 64/3, 64/2, 2*64/3, 5*64/6, 64], [-3, -2, -1, 0, 1, 2, 3])
	ax1.set_ylabel(r'Flux ($\mu$Jy)')
	ax1.legend(loc = 'upper left', fontsize = 17, frameon = False)

	ax2.set_title('z = 1')
	ax2.plot(fullcluster_stack_z1_3000[int(69/2 - 0.5), :]*1e6, lw = 2, color = ccycle[0], alpha = 0.5)
	ax2.plot(fullcluster_stack_z1[int(69/2 - 0.5), :]*1e6, lw = 4, color = ccycle[0], label = 'all clusters')
	ax2.plot(fullms_stack_z1[int(69/2 - 0.5), :]*1e6, lw = 4, ls = '--', color = ccycle[1], label = 'mass-selected clusters')
	ax2.plot(fullrs_stack_z1[int(69/2 - 0.5), :]*1e6, lw = 4, ls = '--', color = ccycle[3], label = 'radius-selected clusters')
	ax2.set_xticks([0, 64/6, 64/3, 64/2, 2*64/3, 5*64/6, 64], [-3, -2, -1, 0, 1, 2, 3])
	ax2.set_ylim([-18, 140])
	ax2.set_xlabel(r'$\Delta$RA (deg)')

	leg_ = mlines.Line2D([], [], color = ccycle[0], alpha = 0.5, lw = 2, label = 'all clusters,\nlog N = 3.5')
	leg2 = ax2.legend([leg_], ['all clusters,\nlog N = 3.5'], loc = 'upper left', frameon = False, fontsize = 17)
	ax2.legend(loc = 'upper right', frameon = False, fontsize = 17)
	ax2.add_artist(leg2)

	ax3.plot(fullcluster_stack_z2_300[:, int(65/2 - 0.5)]*1e6, lw = 2, color = ccycle[0], alpha = 0.5)
	ax3.plot(fullcluster_stack_z2[:, int(65/2 - 0.5)]*1e6, lw = 4, color = ccycle[0])
	ax3.plot(fullms_stack_z2[:, int(65/2 - 0.5)]*1e6, lw = 4, ls = '--', color = ccycle[1])
	ax3.plot(fullrs_stack_z2[:, int(65/2 - 0.5)]*1e6, lw = 4, ls = '--', color = ccycle[3])
	ax3.set_xlabel(r'$\Delta$ZA (deg)')
	ax3.set_xticks([0, 68/6, 68/3, 68/2, 2*68/3, 5*68/6, 68], [-3, -2, -1, 0, 1, 2, 3])
	ax3.set_ylabel(r'Flux ($\mu$Jy)')

	ax4.plot(fullcluster_stack_z1_3000[:, int(65/2 - 0.5)]*1e6, lw = 2, color = ccycle[0], alpha = 0.5)
	ax4.plot(fullcluster_stack_z1[:, int(65/2 - 0.5)]*1e6, lw = 4, color = ccycle[0])
	ax4.plot(fullms_stack_z1[:, int(65/2 - 0.5)]*1e6, lw = 4, ls = '--', color = ccycle[1])
	ax4.plot(fullrs_stack_z1[:, int(65/2 - 0.5)]*1e6, lw = 4, ls = '--', color = ccycle[3])
	ax4.set_xlabel(r'$\Delta$ZA (deg)')
	ax4.set_xticks([0, 68/6, 68/3, 68/2, 2*68/3, 5*68/6, 68], [-3, -2, -1, 0, 1, 2, 3])
	ax4.set_ylim([-18, 140])

	fig.savefig('stacked_flux_clusters_all.png', bbox_inches = 'tight', dpi = 300)


################################## Figure 7 ##################################

def figure7():

	fig, ax = plt.subplots(1, 1, figsize = (9, 7))

	plt.scatter([2, 1], [np.mean(np.array(gal_z2['sub_mHI'])[np.array(gal_z2['sub_mstar']) >= 1e10]), np.mean(np.array(gal_z1['sub_mHI'])[np.array(gal_z1['sub_mstar']) >= 1e10])], 
	            color = ccycle[0], label = 'all galaxies', s = 200, marker = '*')

	plt.scatter([2, 1], [np.mean(np.array(gal_z2['sub_mHI'])[(np.array(gal_z2['sub_mstar']) >= 1e10) & (np.array(gal_z2['incluster']) == 0)]), np.mean(np.array(gal_z1['sub_mHI'])[(np.array(gal_z1['sub_mstar']) >= 1e10) & (np.array(gal_z1['incluster']) == 0)])], 
	            color = ccycle[1], label = 'field galaxies', s = 150, alpha = 0.5)

	plt.scatter([2, 1], [np.mean(np.array(gal_z2['sub_mHI'])[(np.array(gal_z2['sub_mstar']) >= 1e10) & (np.array(gal_z2['incluster']) == 1)]), np.mean(np.array(gal_z1['sub_mHI'])[(np.array(gal_z1['sub_mstar']) >= 1e10) & (np.array(gal_z1['incluster']) == 1)])], 
	            color = ccycle[3], label = 'cluster galaxies', s = 150)


	in1 = ax.inset_axes([0, 0, 0.1, 1], zorder = 0)
	in1.hist(np.array(gal_z2['sub_mHI'])[(np.array(gal_z2['sub_mstar']) >= 1e10) & (np.array(gal_z2['incluster']) == 1)], 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_cg2]*n_cg2, bins = np.logspace(9, 11, 20),
	         color = ccycle[3], lw = 3)
	in1.set_yscale('log')
	in1.set_ylim([1e9, 1e11])
	in1.axis('off')

	in1.hist(np.array(gal_z2['sub_mHI'])[(np.array(gal_z2['sub_mstar']) >= 1e10) & (np.array(gal_z2['incluster']) == 0)], 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_fg2]*n_fg2, bins = np.logspace(9, 11, 20),
	         color = ccycle[1], lw = 3)

	in1.hist(np.array(gal_z2['sub_mHI'])[np.array(gal_z2['sub_mstar']) >= 1e10], 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_g2]*n_g2, bins = np.logspace(9, 11, 20),
	         color = ccycle[0], lw = 2, ls = '--')


	in2 = ax.inset_axes([0.9, 0, 0.1, 1], zorder = 0)
	in2.hist(np.array(gal_z1['sub_mHI'])[(np.array(gal_z1['sub_mstar']) >= 1e10) & (np.array(gal_z1['incluster']) == 1)], 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_cg1]*n_cg1, bins = np.logspace(9, 11, 20),
	         color = ccycle[3], lw = 3)
	in2.set_yscale('log')
	in2.invert_xaxis()
	in2.set_ylim([1e9, 1e11])
	in2.axis('off')

	in2.hist(np.array(gal_z1['sub_mHI'])[(np.array(gal_z1['sub_mstar']) >= 1e10) & (np.array(gal_z1['incluster']) == 0)], 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_fg1]*n_fg1, bins = np.logspace(9, 11, 20),
	         color = ccycle[1], lw = 3)

	in2.hist(np.array(gal_z1['sub_mHI'])[np.array(gal_z1['sub_mstar']) >= 1e10], 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_g1]*n_g1, bins = np.logspace(9, 11, 20),
	         color = ccycle[0], lw = 2, ls = '--')



	plt.legend(fontsize = 15, bbox_to_anchor = (0.97, 1))
	plt.yscale('log')
	plt.xlim([2.5, 0.5])
	plt.ylim([1e9, 1e11])
	plt.xticks([2, 1])

	plt.ylabel(r'M$_\mathrm{HI}$ (M$_\odot$)')
	plt.xlabel(r'z')

	fig.savefig('compare_HI_galaxies_v2.pdf', bbox_inches = 'tight', dpi = 300)


################################## Figure 8 ##################################

def figure8():

	fig, ax = plt.subplots(1, 1, figsize = (9, 7))

	plt.scatter([2, 1], [np.mean(np.array(cluster_z2['group_mHI'])), np.mean(np.array(cluster_z1['group_mHI']))], 
	            color = ccycle[0], label = 'all clusters', s = 200, marker = '*')

	plt.scatter([2, 1], [np.mean(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['mass_select']) == 1]), np.mean(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['mass_select']) == 1])], 
	            color = ccycle[1], label = 'mass-selected clusters', s = 200, alpha = 0.5)

	plt.scatter([2, 1], [np.mean(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['radius_select']) == 1]), np.mean(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['radius_select']) == 1])], 
	            color = ccycle[3], label = 'radius-selected clusters', s = 200, alpha = 1.)

	in1 = ax.inset_axes([0, 0, 0.1, 1], zorder = 0)
	in1.hist(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['radius_select']) == 1], 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_rs2]*n_rs2, bins = np.logspace(10, 12, 20),
	         color = ccycle[3], lw = 3)
	in1.set_yscale('log')
	in1.set_ylim([1e10, 1e12])
	in1.axis('off')

	in1.hist(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['mass_select']) == 1], 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_ms2]*n_ms2, bins = np.logspace(10, 12, 20),
	         color = ccycle[1], lw = 3)

	in1.hist(np.array(cluster_z2['group_mHI']), 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_c2]*n_c2, bins = np.logspace(10, 12, 20),
	         color = ccycle[0], lw = 2, ls = '--')


	in2 = ax.inset_axes([0.9, 0, 0.1, 1], zorder = 0)
	in2.hist(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['radius_select']) == 1], 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_rs1]*n_rs1, bins = np.logspace(10, 12, 20),
	         color = ccycle[3], lw = 3)
	in2.set_yscale('log')
	in2.invert_xaxis()
	in2.set_ylim([1e10, 1e12])
	in2.axis('off')

	in2.hist(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['mass_select']) == 1], 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_ms1]*n_ms1, bins = np.logspace(10, 12, 20),
	         color = ccycle[1], lw = 3)

	in2.hist(np.array(cluster_z1['group_mHI']), 
	         orientation = 'horizontal', histtype = 'step', weights = [1/n_c1]*n_c1, bins = np.logspace(10, 12, 20),
	         color = ccycle[0], lw = 2, ls = '--')

	plt.legend(fontsize = 15, bbox_to_anchor = (1, 1))
	plt.yscale('log')
	plt.ylim([1e10, 1e12])
	plt.xlim([2.5, 0.5])
	plt.xticks([2, 1])
	plt.ylabel(r'M$_\mathrm{HI}$ (M$_\odot$)')
	plt.xlabel(r'z')

	fig.savefig('compare_HI_clusters_v2.pdf', bbox_inches = 'tight', dpi = 300)



################################## Figure 9 ##################################

def figure9():

	### Generate values for plotting ###

	r_range = np.linspace(0, 2, 100)

	full_z2 = np.zeros((278, 100))
	for i, cl in enumerate(cluster_z2.iterrows()):
	    ind = str(cl[1]['GroupID'])
	    rproj = norm_r_z2[ind]
	    mnorm = norm_m_z2[ind]
	    
	    mhist = []
	    for r in r_range:
	        mhist.append(np.sum(mnorm[rproj <= r]))
	    
	    full_z2[i] = mhist
	    
	ms_z2 = np.zeros((276, 100))
	for i, cl in enumerate(cluster_z2.loc[cluster_z2['mass_select'] == 1].iterrows()):
	    ind = str(cl[1]['GroupID'])
	    rproj = norm_r_z2[ind]
	    mnorm = norm_m_z2[ind]
	    
	    mhist = []
	    for r in r_range:
	        mhist.append(np.sum(mnorm[rproj <= r]))
	    
	    ms_z2[i] = mhist
	    
	rs_z2 = np.zeros((26, 100))
	for i, cl in enumerate(cluster_z2.loc[cluster_z2['radius_select'] == 1].iterrows()):
	    ind = str(cl[1]['GroupID'])
	    rproj = norm_r_z2[ind]
	    mnorm = norm_m_z2[ind]
	    
	    mhist = []
	    for r in r_range:
	        mhist.append(np.sum(mnorm[rproj <= r]))
	    
	    rs_z2[i] = mhist


	full_z1 = np.zeros((309, 100))
	for i, cl in enumerate(cluster_z1.iterrows()):
	    ind = str(cl[1]['GroupID'])
	    rproj = norm_r_z1[ind]
	    mnorm = norm_m_z1[ind]
	    
	    mhist = []
	    for r in r_range:
	        mhist.append(np.sum(mnorm[rproj <= r]))
	    
	    full_z1[i] = mhist
	    
	ms_z1 = np.zeros((277, 100))
	for i, cl in enumerate(cluster_z1.loc[cluster_z1['mass_select'] == 1].iterrows()):
	    ind = str(cl[1]['GroupID'])
	    rproj = norm_r_z1[ind]
	    mnorm = norm_m_z1[ind]
	    
	    mhist = []
	    for r in r_range:
	        mhist.append(np.sum(mnorm[rproj <= r]))
	    
	    ms_z1[i] = mhist
	    
	rs_z1 = np.zeros((160, 100))
	for i, cl in enumerate(cluster_z1.loc[cluster_z1['radius_select'] == 1].iterrows()):
	    ind = str(cl[1]['GroupID'])
	    rproj = norm_r_z1[ind]
	    mnorm = norm_m_z1[ind]
	    
	    mhist = []
	    for r in r_range:
	        mhist.append(np.sum(mnorm[rproj <= r]))
	    
	    rs_z1[i] = mhist


	### Code to generate plot ###

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 7), sharey = True, gridspec_kw = {'wspace': 0.08})

	ax1.fill_between(r_range, np.percentile(full_z2, 16, axis = 0), np.percentile(full_z2, 84, axis = 0), 
	                 color = ccycle[0], alpha = 0.2)

	ax1.plot(r_range, np.percentile(full_z2, 50, axis = 0), color = ccycle[0], 
	         lw = 4, label = 'all clusters')

	ax1.fill_between(r_range, np.percentile(ms_z2, 16, axis = 0), np.percentile(ms_z2, 84, axis = 0), 
	                 color = ccycle[1], alpha = 0.2)

	ax1.plot(r_range, np.percentile(ms_z2, 50, axis = 0), color = ccycle[1], 
	         lw = 4, ls = '--', label = 'mass-selected clusters')

	ax1.fill_between(r_range, np.percentile(rs_z2, 16, axis = 0), np.percentile(rs_z2, 84, axis = 0), 
	                 color = ccycle[3], alpha = 0.2)

	ax1.plot(r_range, np.percentile(rs_z2, 50, axis = 0), color = ccycle[3], 
	         lw = 4, label = 'radius-selected clusters')




	ax2.fill_between(r_range, np.percentile(full_z1, 16, axis = 0), np.percentile(full_z1, 84, axis = 0), 
	                 color = ccycle[0], alpha = 0.2)

	ax2.plot(r_range, np.percentile(full_z1, 50, axis = 0), color = ccycle[0], lw = 4)

	ax2.fill_between(r_range, np.percentile(ms_z1, 16, axis = 0), np.percentile(ms_z1, 84, axis = 0), 
	                 color = ccycle[1], alpha = 0.2)

	ax2.plot(r_range, np.percentile(ms_z1, 50, axis = 0), color = ccycle[1], lw = 4, ls = '--')

	ax2.fill_between(r_range, np.percentile(rs_z1, 16, axis = 0), np.percentile(rs_z1, 84, axis = 0), 
	                 color = ccycle[3], alpha = 0.2)

	ax2.plot(r_range, np.percentile(rs_z1, 50, axis = 0), color = ccycle[3], lw = 4)


	ax1.set_ylim([0, 1])
	ax1.set_xlim([0, 2])
	ax2.set_xlim([0, 2])
	ax1.set_ylabel(r'fraction $M_\mathrm{HI,cluster}$ (r < R)')
	ax1.set_xlabel(r'$R/R_{200}$')
	ax2.set_xlabel(r'$R/R_{200}$')

	ax1.set_title(r'$z = 2$')
	ax2.set_title(r'$z = 1$')

	ax1.legend(loc = 'lower right', frameon = False, fontsize = 16)

	fig.savefig('massvr_inclusters.png', bbox_inches = 'tight')


################################## Figure A1 ##################################
## THIS FIGURE REQUIRES THE 13 GB z = 0 TNG300 GROUP CATALOG + 
## 			THE z = 0 Molecular and atomic hydrogen (HI+H2) galaxy contents catalogs + 
## 			THE Durabala+2020 ALFALFA-SDSS Galaxy Catalog
## NONE OF THESE ARE INCLUDED IN THE GITHUB REPO!

def figureA1():

	### z = 0 GK11 M_HI
	basePath = '../../TNG300-1/output'
	subhalos_z0 = il.groupcat.loadSubhalos(basePath,99, 
		fields = ['SubhaloGrNr', 'SubhaloPos', 'SubhaloStellarPhotometricsMassInRad'])

	z0_h = h5py.File('hih2_galaxy_z=0.hdf5', 'r')

	z0_mstar = []
	ids = z0_h['id_subhalo']
	z0_mhi = z0_h['m_hi_GK11_vol']
	for i in ids:
	    z0_mstar.append((1e10/h)*subhalos_z0['SubhaloStellarPhotometricsMassInRad'][int(i)])
	    
	z0_mstar = np.array(z0_mstar)

	### ALFALFA data
	tab = fits.open('durbala2020-table2.21-Sep-2020.fits')
	mstar_alf = tab[1].data['logMstarTaylor']
	mstarerr_alf = tab[1].data['logMstarTaylor_err']

	mhi_alf = tab[1].data['logMH']
	mhierr_alf = tab[1].data['logMH_err']


	### Code to plot

	fig = plt.figure(figsize = (10, 10))

	plt.scatter(z0_mstar[z0_mstar >= 1e10], z0_mhi[z0_mstar >= 1e10], color = ccycle[0])
	plt.scatter(10**mstar_alf[mstar_alf >= 10], 10**mhi_alf[mstar_alf >= 10], color = ccycle[4]) 

	plt.axhline(1e7, color = ccycle[4], ls = '--', lw = 3)
	plt.text(3.9e11, 1.3e7, 'ALFALFA sensitivity at 20 Mpc', color = ccycle[4], fontsize = 12, fontweight = 'heavy')
	plt.xscale('log')
	plt.yscale('log')

	legend_elements = [Patch(facecolor=ccycle[0],label='TNG300 (GK11)'), 
	                   Patch(facecolor=ccycle[4],label = 'ALFALFA-SDSS')]
	plt.legend(handles = legend_elements, loc = 'best', fontsize = 15)
	plt.xlabel(r'M$_\mathrm{star}$ (M$_\odot$)')
	plt.ylabel(r'M$_\mathrm{HI}$ (M$_\odot$)')
	plt.savefig('mstar_mhi_alfalfa.png', bbox_inches = 'tight', dpi = 300)


