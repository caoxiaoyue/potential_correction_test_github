import autolens as al
import numpy as np
import grid_util
import potential_correction_util as pcu
import os
import pickle
from matplotlib import pyplot as plt
import copy

#--------tracer with subhalo
lens_galaxy = al.Galaxy(
    redshift=0.2,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.2,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    subhalo=al.mp.SphIsothermal(
        centre=(1.15, 0.0),
        einstein_radius=0.01,
    )
)
source_galaxy = al.Galaxy(
    redshift=0.6,
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, angle=60.0),
        intensity=0.8,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)
tracer_with_sub = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

#--------tracer without subhalo
lens_galaxy = al.Galaxy(
    redshift=0.2,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.2,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)
tracer_no_sub = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])


#----------get dpsi sparse grid object
mask_data = al.Mask2D.circular_annular(
    shape_native=(200, 200), pixel_scales=0.05, inner_radius=1.2*(1-0.4), outer_radius=1.2*(1+0.4)
)
grid_obj = grid_util.SparseDpsiGrid(mask_data, 0.05, (100, 100))

#-------------get lens potential perturbation vectors caused by subhaloes
try:
    with open(f'./dpsi_sparse_1d.pkl','rb') as f:
        dpsi_sparse_1d = pickle.load(f)
except:
    grid_dpsi = al.Grid2D.uniform(shape_native=grid_obj.shape_2d_dpsi, pixel_scales=grid_obj.dpix_dpsi, sub_size=2)
    dpsi_sparse_2d = tracer_with_sub.potential_2d_from(grid_dpsi).binned.native - tracer_no_sub.potential_2d_from(grid_dpsi).binned.native
    dpsi_sparse_1d = dpsi_sparse_2d[~(grid_obj.mask_dpsi)] #this is just the $\delta \psi$ term in eq.7 in our document
    with open(f'./dpsi_sparse_1d.pkl','wb') as f:
        pickle.dump(dpsi_sparse_1d, f)

#-----------------get dpsi gradient matrix, see eq.7 $D_{\psi}$ term in our document
dpsi_gradient_matrix = pcu.dpsi_gradient_operator_from(grid_obj.Hx_dpsi, grid_obj.Hy_dpsi) 

#-------src gradient matrix
image_grid_vec_numpy = np.vstack([grid_obj.ygrid_data_1d, grid_obj.xgrid_data_1d]).T
image_grid_vec_al = al.Grid2DIrregular(image_grid_vec_numpy)
alpha_src_yx = tracer_no_sub.deflections_yx_2d_from(image_grid_vec_al)
source_points = image_grid_vec_al - alpha_src_yx #I suppose this represents the source pixelization grid
source_values = tracer_no_sub.galaxies[1].image_2d_from(grid=source_points) #I suppose this represents the source reconstruction
dpsi_grid_vec_numpy = np.vstack([grid_obj.ygrid_dpsi_1d, grid_obj.xgrid_dpsi_1d]).T
dpsi_grid_vec_al = al.Grid2DIrregular(dpsi_grid_vec_numpy)
alpha_dpsi_yx = tracer_no_sub.deflections_yx_2d_from(dpsi_grid_vec_al)
src_plane_dpsi_yx = dpsi_grid_vec_al - alpha_dpsi_yx #the location of dpsi grid on the source-plane, under the lens mapping relation given by macro model (without subhalo)
source_gradient = pcu.source_gradient_from(
    source_points, 
    source_values, 
    src_plane_dpsi_yx, 
    cross_size=1e-3,
)
source_gradient_matrix = pcu.source_gradient_matrix_from(source_gradient) #shape: [Np, 2Np]
#------------conformation matrix, see the C_f matrix (eq.7) in our document
Cf_matrix = np.copy(grid_obj.map_matrix)
#-------------linear response
pt_image_correction = -1.0*np.matmul(
    Cf_matrix,
    np.matmul(
        source_gradient_matrix,
        np.matmul(dpsi_gradient_matrix, dpsi_sparse_1d),
    )
)

grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
    sub_size=2,  
)
image_with_sub = tracer_with_sub.image_2d_from(grid).binned.native #suppose this is the real data
image_no_sub = tracer_no_sub.image_2d_from(grid).binned.native #suppose this is the best-fit macro model (although it isn't, since source model can absorb signals from the subhalo)
true_image_residual = image_with_sub - image_no_sub


#solve for the lensed image location
solver = al.PointSolver(
    grid=grid, use_upscaling=True, pixel_scale_precision=0.001, upscale_factor=2
)
positions = solver.solve(
    lensing_obj=tracer_with_sub, source_plane_coordinate=source_galaxy.bulge.centre
)
with open(f'./positions.pkl','wb') as f:
    pickle.dump(positions, f)


#----------------------------plot results
correction_image_residual = np.zeros_like(grid_obj.xgrid_data)
correction_image_residual[~grid_obj.mask_data] = pt_image_correction

coordinate_1d = np.arange(len(grid_obj.mask_data)) * grid_obj.dpix_data
coordinate_1d = coordinate_1d - np.mean(coordinate_1d)
xgrid, ygrid = np.meshgrid(coordinate_1d, coordinate_1d)
rgrid = np.sqrt(xgrid**2 + ygrid**2)
limit = np.max(rgrid[~grid_obj.mask_data])
cmap = copy.copy(plt.get_cmap('jet'))
cmap.set_bad(color='white')

true_image_residual = np.ma.masked_array(true_image_residual, mask=grid_obj.mask_data)
correction_image_residual = np.ma.masked_array(correction_image_residual, mask=grid_obj.mask_data)
difference = correction_image_residual.data - true_image_residual.data
difference = np.ma.masked_array(difference, mask=grid_obj.mask_data)

plt.figure(figsize=(10,5))
plt.subplot(131)
plt.imshow(true_image_residual, cmap=cmap, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)
plt.title('True Residual')

plt.subplot(132)
plt.imshow(correction_image_residual, cmap=cmap, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)
plt.title('Linear Correction')

plt.subplot(133)
plt.imshow(difference, cmap=cmap, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)
plt.title('Difference')

plt.tight_layout()
plt.savefig('1_ideal_forward_correct.jpg')


#simulate noisy data
try:
    with open(f'./imaging.pkl','rb') as f:
        imaging = pickle.load(f)
except:
    psf = al.Kernel2D.from_gaussian(
        shape_native=(11, 11), sigma=0.05, pixel_scales=grid.pixel_scales
    )
    simulator = al.SimulatorImaging(
        exposure_time=1200.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True, noise_seed=1
    )
    imaging = simulator.via_tracer_from(tracer=tracer_with_sub, grid=grid)
    with open(f'./imaging.pkl','wb') as f:
        pickle.dump(imaging, f)

#save lens potential
try:
    with open(f'./psi2d_macro_data.pkl','rb') as f:
        psi2d_macro_data = pickle.load(f)
except:
    psi2d_macro_data = tracer_no_sub.potential_2d_from(grid=grid).binned.native
    with open(f'./psi2d_macro_data.pkl','wb') as f:
        pickle.dump(psi2d_macro_data, f)


try:
    with open(f'./aux_data.pkl','rb') as f:
        aux_data = pickle.load(f)
except:
    aux_data = {
        'Cf_matrix': Cf_matrix,
        'source_gradient_matrix': source_gradient_matrix,
        'dpsi_gradient_matrix': dpsi_gradient_matrix,
        'dpsi_sparse_1d': dpsi_sparse_1d,
        'tracer_with_sub': tracer_with_sub,
        'mask_data': mask_data,
        'grid_obj': grid_obj,
        'true_image_residual': true_image_residual,
    }
    with open(f'./aux_data.pkl','wb') as f:
        pickle.dump(aux_data, f)