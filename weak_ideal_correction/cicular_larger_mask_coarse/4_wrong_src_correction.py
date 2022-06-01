import numpy as np
import autolens as al 
import pickle
import pixelized_source
import pixelized_mass
import copy
import potential_correction_util as pcu

#----------------load data
with open(f'./psi2d_macro_data.pkl','rb') as f:
    psi2d_macro_data = pickle.load(f)
with open(f'./imaging.pkl','rb') as f:
    imaging = pickle.load(f)
with open(f'./aux_data.pkl','rb') as f:
    aux_data = pickle.load(f)
grid_obj = aux_data['grid_obj']
masked_imaging = imaging.apply_mask(mask=aux_data['mask_data'])
masked_imaging = masked_imaging.apply_settings(
    settings=al.SettingsImaging(sub_size=4, sub_size_inversion=4)
)

#----------------src inversion given macro mass model
pix_src_obj = pixelized_source.PixelizedSource(
    masked_imaging, 
    pixelization_shape_2d=(50, 50),
) 
pix_mass = pixelized_mass.PixelizedMass(
    grid_obj.xgrid_data, 
    grid_obj.ygrid_data, 
    psi2d_macro_data, 
    grid_obj.mask_data,
)
# pix_src_obj.find_best_regularization(pix_mass)
# print(pix_src_obj.best_fit_reg_info) #0.7822156306376992

pix_src_obj.source_inversion(pix_mass, lam_s=10) #best reg strength, 60

from matplotlib import  pyplot as plt
from plot import pixelized_source as ps_plot
fig, axes = plt.subplots(figsize=(12,12), nrows=2, ncols=2)
ps_plot.visualize_unmasked_1d_image(masked_imaging.data, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[0,0])
ps_plot.visualize_unmasked_1d_image(pix_src_obj.mapped_reconstructed_image, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[0,1])
ps_plot.visualize_unmasked_1d_image(pix_src_obj.norm_residual_map, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[1,0])
ps_plot.visualize_source(pix_src_obj.relocated_pixelization_grid, pix_src_obj.src_recontruct, ax=axes[1,1])
fig.savefig('4_macro_model_src.jpg')
plt.close()

#----------------source gradient matrix
dpsi_grid_vec_numpy = np.vstack([grid_obj.ygrid_dpsi_1d, grid_obj.xgrid_dpsi_1d]).T
alpha_dpsi_yx = pix_mass.eval_alpha_yx_at(dpsi_grid_vec_numpy)
alpha_dpsi_yx = np.asarray(alpha_dpsi_yx).T
src_plane_dpsi_yx = dpsi_grid_vec_numpy - alpha_dpsi_yx #the location of dpsi grid on the source-plane, under the lens mapping relation given by macro model (without subhalo)
source_gradient = pcu.source_gradient_from(
    pix_src_obj.relocated_pixelization_grid, 
    pix_src_obj.src_recontruct, 
    src_plane_dpsi_yx, 
    cross_size=1e-3,
)
Ds_matrix = pcu.source_gradient_matrix_from(source_gradient) #shape: [Np, 2Np]
# Ds_matrix = aux_data['source_gradient_matrix']


#---------------info for correction
B_matrix = copy.deepcopy(pix_src_obj.psf_blur_matrix)
pix_src_obj.inverse_covariance_matrix()
Cd_inv_matrix = pix_src_obj.inv_cov_mat

Cf_matrix = aux_data['Cf_matrix']
Dpsi_matrix = aux_data['dpsi_gradient_matrix']
# HTH_dpsi = np.matmul(grid_obj.Hx_dpsi_2nd.T, grid_obj.Hx_dpsi_2nd) + \
#             np.matmul(grid_obj.Hy_dpsi_2nd.T, grid_obj.Hy_dpsi_2nd)
# HTH_dpsi = HTH_dpsi*1e3
HTH_dpsi = np.matmul(grid_obj.Hx_dpsi_4th_reg.T, grid_obj.Hx_dpsi_4th_reg) + \
            np.matmul(grid_obj.Hy_dpsi_4th_reg.T, grid_obj.Hy_dpsi_4th_reg)
HTH_dpsi = HTH_dpsi*1e8


#----------------show the residual
fig, axes = plt.subplots(figsize=(10,5), nrows=1, ncols=2)
residual_1d = -1.0*(pix_src_obj.mapped_reconstructed_image - pix_src_obj.masked_imaging.image)
ps_plot.visualize_unmasked_1d_image(residual_1d, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[0])
residual_1d_true = (aux_data['true_image_residual'].data)[~aux_data['true_image_residual'].mask]
residual_1d_true = np.matmul(pix_src_obj.psf_blur_matrix, residual_1d_true)
ps_plot.visualize_unmasked_1d_image(residual_1d_true, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[1])
fig.savefig('4_residual_compare.jpg')
plt.close()
# residual_1d = pix_src_obj.mapped_reconstructed_image - pix_src_obj.masked_imaging.image
# residual_1d = (aux_data['true_image_residual'].data)[~aux_data['true_image_residual'].mask]
# d_1d = masked_imaging.image.native[~grid_obj.mask_data] #1d unmasked image data
# n_1d = masked_imaging.noise_map.native[~aux_data['grid_obj'].mask_data] #1d unmasked noise
# residual_1d = residual_1d_true

#--------------------------solve for the potential correction
Lc_matrix = -1.0*np.matmul(
    Cf_matrix,
    np.matmul(
        Ds_matrix,
        Dpsi_matrix,
    )
)
Mc_matrix = np.matmul(B_matrix, Lc_matrix)

curve_term = np.matmul(
    Mc_matrix.T,
    np.matmul(Cd_inv_matrix, Mc_matrix),
)
curve_reg_term = curve_term + HTH_dpsi

data_vec_term = np.matmul(
    Mc_matrix.T,
    np.matmul(Cd_inv_matrix, residual_1d),
)

dpsi_correction_1d = np.linalg.solve(curve_reg_term, data_vec_term)
print(np.linalg.cond(curve_reg_term))


#---------------------------plot
coordinate_1d = np.arange(len(grid_obj.mask_data)) * grid_obj.dpix_data
coordinate_1d = coordinate_1d - np.mean(coordinate_1d)
xgrid, ygrid = np.meshgrid(coordinate_1d, coordinate_1d)
rgrid = np.sqrt(xgrid**2 + ygrid**2)
limit = np.max(rgrid[~grid_obj.mask_data])
cmap = copy.copy(plt.get_cmap('jet'))
cmap.set_bad(color='white')

true_dpsi_map = np.zeros_like(grid_obj.xgrid_data, dtype='float')
true_dpsi_map[~grid_obj.mask_data] = np.matmul(Cf_matrix, aux_data['dpsi_sparse_1d'])
true_dpsi_map = np.ma.masked_array(true_dpsi_map, mask=grid_obj.mask_data)

diff_level = np.median(aux_data['dpsi_sparse_1d']) - np.median(dpsi_correction_1d)
model_dpsi_map = np.zeros_like(grid_obj.xgrid_data, dtype='float')
model_dpsi_map[~grid_obj.mask_data] =  np.matmul(Cf_matrix, dpsi_correction_1d + diff_level)
model_dpsi_map = np.ma.masked_array(model_dpsi_map, mask=grid_obj.mask_data)
print('------------', np.matmul(Cf_matrix, dpsi_correction_1d))
print('11111111111111', data_vec_term)
print('22222222222222', curve_reg_term)
# print('33333333333333', curve_term)
print('44444444444444', dpsi_correction_1d)
info_dict = {
    'data_vec': data_vec_term,
    'curve': curve_term,
    'reg': HTH_dpsi,
    'curve_reg': curve_reg_term,
    'dpsi_corr': dpsi_correction_1d,
}
with open('./info_test.pkl', 'wb') as f:
    pickle.dump(info_dict, f)

diff_dpsi = (model_dpsi_map.data - true_dpsi_map.data) #/true_dpsi_map.data
diff_dpsi = np.ma.masked_array(diff_dpsi, mask=grid_obj.mask_data)

true_kappa_map = np.zeros_like(grid_obj.xgrid_data, dtype='float')
true_kappa_1d = np.matmul(
    grid_obj.hamiltonian_data,
    true_dpsi_map[~grid_obj.mask_data],
)
true_kappa_map[~grid_obj.mask_data] = true_kappa_1d
true_kappa_map = np.ma.masked_array(true_kappa_map, mask=grid_obj.mask_data)

model_kappa_map = np.zeros_like(grid_obj.xgrid_data, dtype='float')
model_kappa_1d = np.matmul(
    grid_obj.hamiltonian_data,
    model_dpsi_map[~grid_obj.mask_data],
)
model_kappa_map[~grid_obj.mask_data] = model_kappa_1d
model_kappa_map = np.ma.masked_array(model_kappa_map, mask=grid_obj.mask_data)

diff_kappa = (model_kappa_map.data - true_kappa_map.data) #/true_kappa_map.data
diff_kappa = np.ma.masked_array(diff_kappa, mask=grid_obj.mask_data)

plt.figure(figsize=(15,10))
plt.subplot(231)
plt.imshow(true_dpsi_map, cmap=cmap, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)
plt.title('True dpsi')

plt.subplot(232)
plt.imshow(model_dpsi_map, cmap=cmap, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)
plt.title('Model dpsi')

vlim = np.abs(diff_dpsi).max()
plt.subplot(233)
plt.imshow(diff_dpsi, cmap=cmap, extent=grid_obj.image_bound, vmin=-vlim, vmax=vlim)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)
plt.title('Diff dpsi')

plt.subplot(234)
plt.imshow(true_kappa_map, cmap=cmap, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)
plt.title('True kappa')

plt.subplot(235)
plt.imshow(model_kappa_map, cmap=cmap, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)
plt.title('Model kappa')

plt.subplot(236)
plt.imshow(diff_kappa, cmap=cmap, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)
plt.title('Diff kappa')

plt.tight_layout()
plt.savefig('4_macro_backward_correct.jpg')

