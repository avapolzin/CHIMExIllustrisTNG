import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import albumpl
from albumpl.cmap import RhumbLine, Winter05

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
ccycle = return_colors('RhumbLine')

#not including code for Figure 1 here, since beam model is not being shared

################################## Figure A1 ##################################

def figureA1():
	fig = plt.figure(figsize = (9, 7))

	plt.scatter([2, 1], [np.mean(np.array(gal_z2['sub_mHI'])[np.array(gal_z2['sub_mstar']) >= 1e10]), 
		np.mean(np.array(gal_z1['sub_mHI'])[np.array(gal_z1['sub_mstar']) >= 1e10])], 
	            color = ccycle[0], label = 'all galaxies', s = 200, marker = '*')

	plt.scatter([2, 1], [np.mean(np.array(gal_z2['sub_mHI'])[(np.array(gal_z2['sub_mstar']) >= 1e10) & (np.array(gal_z2['incluster']) == 0)]), 
		np.mean(np.array(gal_z1['sub_mHI'])[(np.array(gal_z1['sub_mstar']) >= 1e10) & (np.array(gal_z1['incluster']) == 0)])], 
	            color = ccycle[1], label = 'non-cluster galaxies', s = 150, alpha = 0.5)

	plt.scatter([2, 1], [np.mean(np.array(gal_z2['sub_mHI'])[(np.array(gal_z2['sub_mstar']) >= 1e10) & (np.array(gal_z2['incluster']) == 1)]), 
		np.mean(np.array(gal_z1['sub_mHI'])[(np.array(gal_z1['sub_mstar']) >= 1e10) & (np.array(gal_z1['incluster']) == 1)])], 
	            color = ccycle[3], label = 'cluster member galaxies', s = 150)

	plt.plot([2.3, 2.3], 
		[3e9, 3e9 + np.std(np.array(gal_z2['sub_mHI'])[(np.array(gal_z2['sub_mstar']) >= 1e10) & (np.array(gal_z2['incluster']) == 1)])], 
		color = ccycle[3])
	plt.plot([2.35, 2.35], 
		[3e9, 3e9 + np.std(np.array(gal_z2['sub_mHI'])[(np.array(gal_z2['sub_mstar']) >= 1e10) & (np.array(gal_z2['incluster']) == 0)])], 
		color = ccycle[1])
	plt.plot([2.4, 2.4], 
		[3e9, 3e9 + np.std(np.array(gal_z2['sub_mHI'])[np.array(gal_z2['sub_mstar']) >= 1e10])], color = ccycle[0])

	plt.plot([0.6, 0.6], 
		[3e9, 3e9 + np.std(np.array(gal_z1['sub_mHI'])[(np.array(gal_z1['sub_mstar']) >= 1e10) & (np.array(gal_z1['incluster']) == 1)])], 
		color = ccycle[3])
	plt.plot([0.65, 0.65], 
		[3e9, 3e9 + np.std(np.array(gal_z1['sub_mHI'])[(np.array(gal_z1['sub_mstar']) >= 1e10) & (np.array(gal_z1['incluster']) == 0)])], 
		color = ccycle[1])
	plt.plot([0.7, 0.7], 
		[3e9, 3e9 + np.std(np.array(gal_z1['sub_mHI'])[np.array(gal_z1['sub_mstar']) >= 1e10])], 
		color = ccycle[0])


	plt.legend(loc = 'best', fontsize = 15, bbox_to_anchor = (1, 1))
	plt.yscale('log')
	plt.xlim([2.5, 0.5])
	plt.ylim([1e9, 1e11])
	plt.xticks([2, 1])

	plt.ylabel(r'M$_\mathrm{HI}$ (M$_\odot$)')
	plt.xlabel(r'z')

	fig.savefig('compare_HI_galaxies.pdf', bbox_inches = 'tight', dpi = 300)


################################## Figure A2 ##################################
def figureA2():
	fig = plt.figure(figsize = (9, 7))

	plt.scatter([2, 1], [np.mean(np.array(cluster_z2['group_mHI'])), np.mean(np.array(cluster_z1['group_mHI']))], 
	            color = ccycle[0], label = 'all clusters', s = 200, marker = '*')

	plt.scatter([2, 1], [np.mean(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['mass_select']) == 1]), 
		np.mean(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['mass_select']) == 1])], 
	            color = ccycle[1], label = 'mass-selected clusters', s = 200, alpha = 0.5)

	plt.scatter([2, 1], [np.mean(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['radius_select']) == 1]), 
		np.mean(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['radius_select']) == 1])], 
	            color = ccycle[3], label = 'radius-selected clusters', s = 200)

	plt.plot([2.3, 2.3], [5e10, 5e10 + np.std(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['radius_select']) == 1])], 
		color = ccycle[3])
	plt.plot([2.35, 2.35], [5e10, 5e10 + np.std(np.array(cluster_z2['group_mHI'])[np.array(cluster_z2['mass_select']) == 1])], 
		color = ccycle[1])
	plt.plot([2.4, 2.4], [5e10, 5e10 + np.std(np.array(cluster_z2['group_mHI']))], color = ccycle[0])


	plt.plot([0.6, 0.6], [5e10, 5e10 + np.std(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['radius_select']) == 1])], 
		color = ccycle[3])
	plt.plot([0.65, 0.65], [5e10, 5e10 + np.std(np.array(cluster_z1['group_mHI'])[np.array(cluster_z1['mass_select']) == 1])], 
		color = ccycle[1])
	plt.plot([0.7, 0.7], [5e10, 5e10 + np.std(np.array(cluster_z1['group_mHI']))], color = ccycle[0])

	plt.legend(loc = 'best', fontsize = 12)
	plt.yscale('log')
	plt.ylim([1e10, 1e12])
	plt.xlim([2.5, 0.5])
	plt.xticks([2, 1])
	plt.ylabel(r'M$_\mathrm{HI}$ (M$_\odot$)')
	plt.xlabel(r'z')

	fig.savefig('compare_HI_clusters.pdf', bbox_inches = 'tight', dpi = 300)


################################## Figure 2 ##################################
def figure2():
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (20, 14), sharey = 'row', gridspec_kw = {'wspace':0.05})

	ax1.set_title('z = 2')
	ax1.plot(full_stack_z2[int(69/2 - 0.5), :], lw = 4, color = ccycle[0], label = 'all galaxies')
	ax1.plot(nc_stack_z2[int(69/2 - 0.5), :], lw = 4, ls = '--', color = ccycle[1], label = 'non-cluster galaxies')
	ax1.plot(c_stack_z2[int(69/2 - 0.5), :], lw = 4, ls = '--', color = ccycle[3], label = 'cluster galaxies')
	ax1.set_xlabel(r'$\Delta$ZA (deg)')
	ax1.set_xticks([0, 64/6, 64/3, 64/2, 2*64/3, 5*64/6, 64], [-3, -2, -1, 0, 1, 2, 3])
	# ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax1.set_ylabel('Flux (Jy MHz)')
	ax1.legend(loc = 'best')

	ax2.set_title('z = 1')
	ax2.plot(full_stack_z1[int(69/2 - 0.5), :], lw = 4, color = ccycle[0])
	ax2.plot(nc_stack_z1[int(69/2 - 0.5), :], lw = 4, ls = '--', color = ccycle[1])
	ax2.plot(c_stack_z1[int(69/2 - 0.5), :], lw = 4, ls = '--', color = ccycle[3])
	ax2.set_xlabel(r'$\Delta$ZA (deg)')
	ax2.set_xticks([0, 64/6, 64/3, 64/2, 2*64/3, 5*64/6, 64], [-3, -2, -1, 0, 1, 2, 3])
	# ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax4.set_ylim([-0.00006, 0.00030])

	# ax3.set_title('z = 2')
	ax3.plot(full_stack_z2[:, int(65/2 - 0.5)], lw = 4, color = ccycle[0], label = 'all galaxies')
	ax3.plot(nc_stack_z2[:, int(65/2 - 0.5)], lw = 4, ls = '--', color = ccycle[1], label = 'non-cluster galaxies')
	ax3.plot(c_stack_z2[:, int(65/2 - 0.5)], lw = 4, ls = '--', color = ccycle[3], label = 'cluster galaxies')
	ax3.set_xlabel(r'$\Delta$RA (deg)')
	ax3.set_xticks([0, 68/6, 68/3, 68/2, 2*68/3, 5*68/6, 68], [-3, -2, -1, 0, 1, 2, 3])
	# ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax3.set_ylabel('Flux (Jy MHz)')
	# ax3.legend(loc = 'best')

	# ax4.set_title('z = 1')
	ax4.plot(full_stack_z1[:, int(65/2 - 0.5)], lw = 4, color = ccycle[0])
	ax4.plot(nc_stack_z1[:, int(65/2 - 0.5)], lw = 4, ls = '--', color = ccycle[1])
	ax4.plot(c_stack_z1[:, int(65/2 - 0.5)], lw = 4, ls = '--', color = ccycle[3])
	ax4.set_xlabel(r'$\Delta$RA (deg)')
	ax4.set_xticks([0, 68/6, 68/3, 68/2, 2*68/3, 5*68/6, 68], [-3, -2, -1, 0, 1, 2, 3])
	# ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax4.set_ylim([-0.00006, 0.00030])

	fig.savefig('stacked_flux_galaxies_all.png', bbox_inches = 'tight', dpi = 300)


################################## Figure 3 ##################################
def figure3():
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (20, 14), sharey = 'row', gridspec_kw = {'wspace':0.05})

	ax1.set_title('z = 2')
	ax1.plot(fullcluster_stack_z2[int(69/2 - 0.5), :], lw = 4, color = ccycle[0], label = 'full map, all clusters')
	ax1.plot(fullms_stack_z2[int(69/2 - 0.5), :], lw = 4, ls = '--', color = ccycle[1], label = 'full map, mass-selected clusters')
	ax1.plot(ms_stack_z2[int(69/2 - 0.5), :], lw = 4, ls = ':', color = ccycle[1], label = 'mass-selected clusters')
	ax1.plot(fullrs_stack_z2[int(69/2 - 0.5), :], lw = 4, ls = '--', color = ccycle[3], label = 'full map, radius-selected clusters', zorder = 10)
	ax1.plot(rs_stack_z2[int(69/2 - 0.5), :], lw = 4, ls = ':', color = ccycle[3], label = 'radius-selected clusters')
	ax1.set_xlabel(r'$\Delta$ZA (deg)')
	ax1.set_xticks([0, 64/6, 64/3, 64/2, 2*64/3, 5*64/6, 64], [-3, -2, -1, 0, 1, 2, 3])
	# ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax1.set_ylabel('Flux (Jy MHz)')
	ax1.legend(loc = 'upper left', fontsize = 15)

	ax2.set_title('z = 1')
	ax2.plot(fullcluster_stack_z1[int(69/2 - 0.5), :], lw = 4, color = ccycle[0])
	ax2.plot(fullms_stack_z1[int(69/2 - 0.5), :], lw = 4, ls = '--', color = ccycle[1])
	ax2.plot(ms_stack_z1[int(69/2 - 0.5), :], lw = 4, ls = ':', color = ccycle[1])
	ax2.plot(fullrs_stack_z1[int(69/2 - 0.5), :], lw = 4, ls = '--', color = ccycle[3])
	ax2.plot(rs_stack_z1[int(69/2 - 0.5), :], lw = 4, ls = ':', color = ccycle[3])
	ax2.set_xlabel(r'$\Delta$ZA (deg)')
	ax2.set_xticks([0, 64/6, 64/3, 64/2, 2*64/3, 5*64/6, 64], [-3, -2, -1, 0, 1, 2, 3])
	# ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax4.set_ylim([-0.00006, 0.00030])

	# ax3.set_title('z = 2')
	ax3.plot(fullcluster_stack_z2[:, int(65/2 - 0.5)], lw = 4, color = ccycle[0])
	ax3.plot(fullms_stack_z2[:, int(65/2 - 0.5)], lw = 4, ls = '--', color = ccycle[1])
	ax3.plot(ms_stack_z2[:, int(65/2 - 0.5)], lw = 4, ls = ':', color = ccycle[1])
	ax3.plot(fullrs_stack_z2[:, int(65/2 - 0.5)], lw = 4, ls = '--', color = ccycle[3])
	ax3.plot(rs_stack_z2[:, int(65/2 - 0.5)], lw = 4, ls = ':', color = ccycle[3])
	ax3.set_xlabel(r'$\Delta$RA (deg)')
	ax3.set_xticks([0, 68/6, 68/3, 68/2, 2*68/3, 5*68/6, 68], [-3, -2, -1, 0, 1, 2, 3])
	# ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax3.set_ylabel('Flux (Jy MHz)')
	# ax3.legend(loc = 'best')

	# ax4.set_title('z = 1')
	ax4.plot(fullcluster_stack_z1[:, int(65/2 - 0.5)], lw = 4, color = ccycle[0])
	ax4.plot(fullms_stack_z1[:, int(65/2 - 0.5)], lw = 4, ls = '--', color = ccycle[1])
	ax4.plot(ms_stack_z1[:, int(65/2 - 0.5)], lw = 4, ls = ':', color = ccycle[1])
	ax4.plot(fullrs_stack_z1[:, int(65/2 - 0.5)], lw = 4, ls = '--', color = ccycle[3])
	ax4.plot(rs_stack_z1[:, int(65/2 - 0.5)], lw = 4, ls = ':', color = ccycle[3])
	ax4.set_xlabel(r'$\Delta$RA (deg)')
	ax4.set_xticks([0, 68/6, 68/3, 68/2, 2*68/3, 5*68/6, 68], [-3, -2, -1, 0, 1, 2, 3])
	# ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax4.set_ylim([-0.00006, 0.00030])

	fig.savefig('stacked_flux_clusters_all.png', bbox_inches = 'tight', dpi = 300)