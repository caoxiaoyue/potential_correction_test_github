#%%
import autolens as al 

def psi_shear(x, y, shear_amp, shear_angle):
# The potential of the external shear has the form of 
# psi=-0.5*shear*r^2*cos2(phi-phi_shear)
    shear_angle = shear_angle - 90 #the position angle in autolens are defined with repect to the postive x-axis
    phig=np.deg2rad(shear_angle)
    phicoord=np.arctan2(y, x)
    rcoord=np.sqrt(x**2.+y**2.)

    psi = -0.5 * shear_amp * rcoord**2 * np.cos(2*(phicoord-phig))
    return psi


epl = al.mp.EllPowerLaw(
    centre = (0.0, 0.0),
    elliptical_comps = (-0.020, 0.026),
    einstein_radius = 1.404,
    slope = 2.2,
)

shear = al.mp.ExternalShear(
    elliptical_comps = (-0.056, 0.068)
)


lens_galaxy = al.Galaxy(
    redshift=0.222,
    mass=epl,
    shear=shear,
)

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

ygrid = imaging.grid.binned.native[:,:,0]
xgrid = imaging.grid.binned.native[:,:,1]


# %%
import pickle
try:
    with open(f'./psi2d_macro_data.pkl','rb') as f:
        psi_2d = pickle.load(f)
except:
    psi_2d = lens_galaxy.potential_2d_from(imaging.grid).binned.native
    psi_2d += psi_shear(xgrid, ygrid, shear.magnitude, shear.angle) #since autolend do not calculate the potential of shear
    with open(f'./psi2d_macro_data.pkl','wb') as f:
        pickle.dump(psi_2d, f)

    
# %%
from matplotlib import pyplot as plt
solver = al.PointSolver(
    grid=imaging.grid, use_upscaling=True, pixel_scale_precision=0.001, upscale_factor=2
)
positions = solver.solve(
    lensing_obj=lens_galaxy, source_plane_coordinate=(-0.147, -0.031)
)
with open(f'./positions.pkl','wb') as f:
    pickle.dump(positions, f)
    
# %%
import pixelized_mass
import numpy as np

alpha_2d = lens_galaxy.deflections_yx_2d_from(imaging.grid).binned.native
psi_2d_true = np.copy(psi_2d)
alphay_2d_true = alpha_2d[:,:,0]
alphax_2d_true = alpha_2d[:,:,1]
psi_2d_true[mask_data] = 0.0
alphax_2d_true[mask_data] = 0.0
alphay_2d_true[mask_data] = 0.0 

pix_mass = pixelized_mass.PixelizedMass(xgrid, ygrid, psi_2d, mask_data)

xgrid_1d = xgrid[~mask_data]
ygrid_1d = ygrid[~mask_data]
points = np.array(list(zip(ygrid_1d, xgrid_1d))) #use autolens [(y1,x1),(y2,x2),...] order
psi_1d_model = pix_mass.eval_psi_at(points)
alphay_1d_model, alphax_1d_model = pix_mass.eval_alpha_yx_at(points)

psi_2d_model = np.zeros_like(psi_2d_true)
alphax_2d_model = np.zeros_like(alphax_2d_true)
alphay_2d_model = np.zeros_like(alphay_2d_true)
psi_2d_model[~mask_data] = psi_1d_model
alphax_2d_model[~mask_data] = alphax_1d_model
alphay_2d_model[~mask_data] = alphay_1d_model


#%%
plt.figure(figsize=(15,10))

plt.subplot(231)
plt.imshow(alphax_2d_model, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('defl-x-numeric')

plt.subplot(232)
plt.imshow(alphax_2d_true, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('defl-x-analyt')

plt.subplot(233)
plt.imshow(alphax_2d_model - alphax_2d_true, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('defl-x-diff')

plt.subplot(234)
plt.imshow(alphay_2d_model, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('defl-y-numeric')

plt.subplot(235)
plt.imshow(alphay_2d_true, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('defl-y-analyt')

plt.subplot(236)
plt.imshow(alphay_2d_model - alphay_2d_true, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('defl-y-diff')

plt.show()


# %%
