#%%
import numpy as np
import autolens as al 
import pickle

epl = al.mp.EllPowerLaw(
    centre = (0.0, 0.0),
    elliptical_comps = (1e-9, 1e-9),
    einstein_radius = 1.404,
    slope = 2.2,
)

shear = al.mp.ExternalShear(
    elliptical_comps = (-0.0000001, 0.000001)
)
lens_galaxy = al.Galaxy(
    redshift=0.222,
    mass=epl,
    # shear=shear,
)



with open(f'./positions.pkl','rb') as f:
    positions = pickle.load(f)

imaging = al.Imaging.from_fits(
    image_path='./data/data4autolens.fits',
    image_hdu=0,
    noise_map_path='./data/data4autolens.fits',
    noise_map_hdu=1,
    psf_path='./data/data4autolens.fits',
    psf_hdu=2,
    pixel_scales=0.05,
)
mask_data = al.Mask2D.elliptical_annular(
    shape_native=imaging.shape_native,
    inner_major_axis_radius=0.7,
    inner_axis_ratio=1,
    inner_phi=175,
    outer_major_axis_radius=2.1,
    outer_axis_ratio=1,
    outer_phi=175,
    pixel_scales=0.05,
)
masked_imaging = imaging.apply_mask(mask_data)


sub_size = 4
masked_imaging = masked_imaging.apply_settings(
    settings=al.SettingsImaging(sub_size=sub_size, sub_size_inversion=sub_size)
)

try:
    with open(f'./psi2d_macro_data.pkl','rb') as f:
        psi_2d_macro = pickle.load(f)
except:
    psi_2d_macro = lens_galaxy.potential_2d_from(imaging.grid).binned.native
    with open(f'./psi2d_macro_data.pkl','wb') as f:
        pickle.dump(psi_2d_macro, f)


#%%
import pixelized_mass
import pixelized_source
ygrid =  masked_imaging.grid.unmasked_grid.binned.native[:,:,0]
xgrid =  masked_imaging.grid.unmasked_grid.binned.native[:,:,1]
pix_mass = pixelized_mass.PixelizedMass(
    xgrid, 
    ygrid, 
    psi_2d_macro, 
    masked_imaging.mask
) #a mass model defined by lensing potential on pixelized grids.


#%%
pix_src = pixelized_source.PixelizedSource(masked_imaging, pixelization_shape_2d=(50,50))

# %%
# find the regularization strength given a lens mass model
# print('find best source regularization')
# pix_src.find_best_regularization(pix_mass, log10_lam_range=[-5, 4])
# print(pix_src.best_fit_reg_info)
# print(pix_src.mp_lam)
# pix_src.source_inversion(pix_mass, lam_s=pix_src.mp_lam) #best reg strength found by the code below
pix_src.source_inversion(pix_mass, lam_s=6.32211064) 
print('evidence-my', pix_src.evidence_from_reconstruction())
from matplotlib import  pyplot as plt
from plot import pixelized_source as ps_plot
fig, axes = plt.subplots(figsize=(12,12), nrows=2, ncols=2)
ps_plot.visualize_unmasked_1d_image(masked_imaging.data, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[0,0])
axes[0, 0].plot(positions[:, 1], positions[:, 0], '*')
axes[0, 0].set_title('Data')
axes[0, 0].set_ylabel('Arcsec')
ps_plot.visualize_unmasked_1d_image(pix_src.mapped_reconstructed_image, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[0,1])
axes[0, 1].set_title('Model')
ps_plot.visualize_unmasked_1d_image(-1.0*pix_src.norm_residual_map, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[1,0])
axes[1, 0].set_title('Norm residual')
axes[1, 0].set_ylabel('Arcsec')
axes[1, 0].set_xlabel('Arcsec')
ps_plot.visualize_source(pix_src.relocated_pixelization_grid, pix_src.src_recontruct, ax=axes[1,1])
axes[1, 1].set_title('Source')
axes[1, 1].set_xlabel('Arcsec')
fig.savefig('macro_model_fit.jpg')


#%%
pixelization_shape_2d = (50, 50)
pixelization = al.pix.DelaunayMagnification(shape=pixelization_shape_2d)
source_galaxy = al.Galaxy(
    redshift=0.6,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=6.32211064),
)
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
fitter = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_w_tilde=False),
    settings_pixelization=al.SettingsPixelization(use_border=True),
)


#%%
from plot import pixelized_source as ps_plot
fig, axes = plt.subplots(figsize=(12,12), nrows=2, ncols=2)
ps_plot.visualize_unmasked_1d_image(masked_imaging.data, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[0,0])
ps_plot.visualize_unmasked_1d_image(fitter.inversion.mapped_reconstructed_data, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[0,1])
ps_plot.visualize_unmasked_1d_image(fitter.normalized_residual_map, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[1,0])
ps_plot.visualize_source(fitter.inversion.mapper_list[0].source_pixelization_grid, fitter.inversion.reconstruction, ax=axes[1,1]) 
fig.savefig('macro_model_fit_al.jpg')
print('evidence-autolens', fitter.log_evidence)


#%%
al_src_pts = fitter.inversion.mapper_list[0].source_pixelization_grid
my_src_pts = pix_src.relocated_pixelization_grid
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(al_src_pts[:, 1], al_src_pts[:, 0], s=0.5)
plt.subplot(122)
plt.scatter(my_src_pts[:, 1], my_src_pts[:, 0], s=0.5)
plt.axis('square')
plt.show()


# %%
print('evidence-autolens', pix_src.evidence_from_reconstruction())
print('evidence-autolens', fitter.log_evidence)

# %%