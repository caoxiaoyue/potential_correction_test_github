#%%
import numpy as np
import autolens as al 
import pickle
import pixelized_mass
import pixelized_source
import os 

try:
    with open(f'./masked_imaging.pkl','rb') as f:
        masked_imaging = pickle.load(f)
except:
    sub_size = 4
    annular_width = 0.3
    inner_mask_radius = 1.2*(1-annular_width)
    outter_mask_radius = 1.2*(1+annular_width)
    dataset_path = f'/home/cao/data_disk/autolens_xycao/potential_correction/simulate_data/dataset/sie_sis_sersic'
    imaging = al.Imaging.from_fits(
        image_path=os.path.join(dataset_path, "image.fits"),
        psf_path=os.path.join(dataset_path, "psf.fits"),
        noise_map_path=os.path.join(dataset_path, "noise_map.fits"),
        pixel_scales=0.05,
    )
    mask = al.Mask2D.circular_annular(
        shape_native=imaging.shape_native,
        pixel_scales=imaging.pixel_scales,
        sub_size=sub_size,
        inner_radius=inner_mask_radius,
        outer_radius=outter_mask_radius,
    )
    masked_imaging = imaging.apply_mask(mask=mask)
    masked_imaging = masked_imaging.apply_settings(
        settings=al.SettingsImaging(sub_size=sub_size, sub_size_inversion=sub_size)
    )
    with open(f'./masked_imaging.pkl','wb') as f:
        pickle.dump(masked_imaging, f)


with open(f'./positions.pkl','rb') as f:
    positions = pickle.load(f)

#the main lens parameters of mock data
lens_galaxy = al.Galaxy(
    redshift=0.2,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.2,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)


try:
    with open(f'./psi2d_macro_data.pkl','rb') as f:
        psi_2d = pickle.load(f)
except:
    psi_2d = lens_galaxy.potential_2d_from(masked_imaging.grid).binned.native
    with open(f'./psi2d_macro_data.pkl','wb') as f:
        pickle.dump(psi_2d, f)


ygrid =  masked_imaging.grid.unmasked_grid.binned.native[:,:,0]
xgrid =  masked_imaging.grid.unmasked_grid.binned.native[:,:,1]
pix_mass = pixelized_mass.PixelizedMass(
    xgrid, 
    ygrid, 
    psi_2d, 
    masked_imaging.mask
) #a mass model defined by lensing potential on pixelized grids.


#%%
pix_src = pixelized_source.PixelizedSource(masked_imaging, pixelization_shape_2d=(50,50))

# %%
# find the regularization strength given a lens mass model
print('find best source regularization')
pix_src.find_best_regularization(pix_mass, log10_lam_range=[-5, 4])
print(pix_src.best_fit_reg_info)
print(pix_src.mp_lam)


pix_src.source_inversion(pix_mass, lam_s=pix_src.mp_lam) #best reg strength found by the code below
# pix_src.source_inversion(pix_mass, lam_s=60) #best reg strength found by the code below


#%%
from matplotlib import  pyplot as plt
from plot import pixelized_source as ps_plot
fig, axes = plt.subplots(figsize=(12,12), nrows=2, ncols=2)
ps_plot.visualize_unmasked_1d_image(masked_imaging.data, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[0,0])
axes[0, 0].plot(positions[:, 1], positions[:, 0], '*')
axes[0, 0].set_title('Data')
axes[0, 0].set_ylabel('Arcsec')
ps_plot.visualize_unmasked_1d_image(pix_src.mapped_reconstructed_image, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[0,1])
axes[0, 1].set_title('Model')
ps_plot.visualize_unmasked_1d_image(pix_src.norm_residual_map, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[1,0])
axes[1, 0].set_title('Norm residual')
axes[1, 0].set_ylabel('Arcsec')
axes[1, 0].set_xlabel('Arcsec')
ps_plot.visualize_source(pix_src.relocated_pixelization_grid, pix_src.src_recontruct, ax=axes[1,1]) #source looks irregular becasue we manually fix the lens mass model to a incorrect one
axes[1, 1].set_title('Source')
axes[1, 1].set_xlabel('Arcsec')
fig.savefig('macro_model_fit.jpg')

# %%
