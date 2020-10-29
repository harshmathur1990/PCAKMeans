from mpl_toolkits.axes_grid1.inset_locator import inset_axes


plt.close('all')
plt.clf()
plt.cla()
fig, axs = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={'wspace': 0.001, 'hspace': 0.025})

im0 = axs[0][0].imshow(a, cmap='gray', origin='lower')
im1 = axs[0][1].imshow(b, cmap='gray', origin='lower')
im2 = axs[1][0].imshow(c, cmap='gray', origin='lower')
im3 = axs[1][1].imshow(d, cmap='gray', origin='lower')

# plt.axis('on')
axs[0][0].set_xticklabels([])
axs[0][0].set_yticklabels([])
# axs[0][0].set_aspect('equal')
axs[0][1].set_xticklabels([])
axs[0][1].set_yticklabels([])
# axs[0][1].set_aspect('equal')
axs[1][0].set_xticklabels([])
axs[1][0].set_yticklabels([])
# axs[1][0].set_aspect('equal')
axs[1][1].set_xticklabels([])
axs[1][1].set_yticklabels([])
# axs[1][1].set_aspect('equal')


axins0 = inset_axes(
    axs[0][0],
    width="5%",
    height="50%",
    loc='lower left',
    bbox_to_anchor=(-0.15, 0., 1, 1),
    bbox_transform=axs[0][0].transAxes,
    borderpad=0,
)

axins1 = inset_axes(
    axs[0][1],
    width="5%",
    height="50%",
    loc='lower right',
    bbox_to_anchor=(0.1, 0., 1, 1),
    bbox_transform=axs[0][1].transAxes,
    borderpad=0,
)

axins2 = inset_axes(
    axs[1][0],
    width="5%",
    height="50%",
    loc='lower left',
    bbox_to_anchor=(-0.15, 0., 1, 1),
    bbox_transform=axs[1][0].transAxes,
    borderpad=0,
)

axins3 = inset_axes(
    axs[1][1],
    width="5%",
    height="50%",
    loc='lower right',
    bbox_to_anchor=(0.1, 0., 1, 1),
    bbox_transform=axs[1][1].transAxes,
    borderpad=0,
)

cbar0 = fig.colorbar(im0, cax=axins0)
cbar1 = fig.colorbar(im1, cax=axins1)
cbar2 = fig.colorbar(im2, cax=axins2)
cbar3 = fig.colorbar(im3, cax=axins3)

cbar0.ax.tick_params(labelsize=10)
cbar1.ax.tick_params(labelsize=10)
cbar2.ax.tick_params(labelsize=10)
cbar3.ax.tick_params(labelsize=10)

plt.show()