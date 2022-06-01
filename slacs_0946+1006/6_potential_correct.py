#%%
import numpy as np
import autolens as al 
import pickle

with open(f'./psi2d_macro_data.pkl','rb') as f:
    psi_2d_macro = pickle.load(f)

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
    inner_axis_ratio=0.9,
    inner_phi=175,
    outer_major_axis_radius=2.1,
    outer_axis_ratio=0.9,
    outer_phi=175,
    pixel_scales=0.05,
)
masked_imaging = imaging.apply_mask(mask_data)


sub_size = 4
masked_imaging = masked_imaging.apply_settings(
    settings=al.SettingsImaging(sub_size=sub_size, sub_size_inversion=sub_size)
)


#%%
n1, n2 = imaging.shape_native
fac = 1
import iterative_solve_suyu
potential_corrector = iterative_solve_suyu.IterativePotentialCorrect(
    masked_imaging,
    shape_2d_dpsi=(int(n1/fac), int(n2/fac)),
    shape_2d_src=(50, 50),
)
potential_corrector.initialize_iteration(
    psi_2d_start=psi_2d_macro, 
    niter=5000, 
    lam_s_start=7, 
    lam_dpsi_start=0.2*1e10,
    lam_dpsi_type='2nd',
    psi_anchor_points=np.array([(0.0,-1.0),(-1.0,1.0),(1.0,1.0)]),
    check_converge_points=positions,
    subhalo_fiducial_point=(1.040, -0.651),
)
potential_corrector.run_iter_solve()