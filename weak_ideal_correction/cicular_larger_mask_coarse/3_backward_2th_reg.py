import autolens as al
import numpy as np
import grid_util
import potential_correction_util as pcu
import os
import pickle
from matplotlib import pyplot as plt
import copy
import pixelized_source

#----------------LOAD DATA
with open(f'./imaging.pkl','rb') as f:
    imaging = pickle.load(f)
with open(f'./aux_data.pkl','rb') as f:
    aux_data = pickle.load(f)
grid_obj = aux_data['grid_obj']

masked_imaging = imaging.apply_mask(mask=aux_data['mask_data'])
masked_imaging = masked_imaging.apply_settings(
    settings=al.SettingsImaging(sub_size=2, sub_size_inversion=2)
)

#------get psf matrix and inverse covariance matrices
pix_src_obj = pixelized_source.PixelizedSource(
    masked_imaging, 
    pixelization_shape_2d=(50, 50),
) 
pix_src_obj.build_psf_matrix()
B_matrix = copy.deepcopy(pix_src_obj.psf_blur_matrix)
pix_src_obj.inverse_covariance_matrix()
Cd_inv_matrix = pix_src_obj.inv_cov_mat

#---------------info for correction
Cf_matrix = aux_data['Cf_matrix']
Ds_matrix = aux_data['source_gradient_matrix']
Dpsi_matrix = aux_data['dpsi_gradient_matrix']
HTH_dpsi = np.matmul(grid_obj.Hx_dpsi_2nd_reg.T, grid_obj.Hx_dpsi_2nd_reg) + \
            np.matmul(grid_obj.Hy_dpsi_2nd_reg.T, grid_obj.Hy_dpsi_2nd_reg)
HTH_dpsi = HTH_dpsi*1e8

residual_1d = (aux_data['true_image_residual'].data)[~aux_data['true_image_residual'].mask]
residual_1d = np.matmul(pix_src_obj.psf_blur_matrix, residual_1d)

# d_1d = masked_imaging.image.native[~grid_obj.mask_data] #1d unmasked image data
# n_1d = masked_imaging.noise_map.native[~aux_data['grid_obj'].mask_data] #1d unmasked noise

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
plt.savefig('3_ideal_backward_correct.jpg')