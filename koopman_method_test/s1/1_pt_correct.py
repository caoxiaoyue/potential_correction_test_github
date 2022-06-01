#%%
import numpy as np
import autolens as al 
import pickle
import pixelized_mass
import pixelized_source
import os 
current_dir, current_file_name = os.path.split(os.path.abspath(__file__))

sub_size = 4
with open(f'./masked_imaging.pkl','rb') as f:
    masked_imaging = pickle.load(f)

with open(f'{current_dir}/psi2d_macro_data.pkl','rb') as f:
    psi_2d_macro = pickle.load(f)

with open(f'./positions.pkl','rb') as f:
    positions = pickle.load(f)


with open(f'{current_dir}/psi2d_macro_data.pkl','rb') as f:
    psi_2d_macro = pickle.load(f)


#%%
import iterative_solve
potential_corrector = iterative_solve.IterativePotentialCorrect(
    masked_imaging,
    shape_2d_dpsi=(100, 100),
    shape_2d_src=(50, 50),
)
potential_corrector.initialize_iteration(
    psi_2d_start=psi_2d_macro, 
    niter=2, 
    lam_s_start=10, 
    lam_dpsi_start=1e80,
    lam_dpsi_type='4th',
    psi_anchor_points=np.array([(-1.0,-1.0),(-1.0,1.0),(0.0,1.0)]),
    check_converge_points=positions,
    subhalo_fiducial_point=(1.15, 0.0),
)
potential_corrector.run_iter_solve()
#The bug is due to we do not pad the lens-potential regularization at the boundary,
#the regularization matrix of potential correction will be ill-conditioned, thus lead to inaccurate source solution 
# (because we couple the linear equatio of lens-potential and source brightness, which is different to suyu's method)