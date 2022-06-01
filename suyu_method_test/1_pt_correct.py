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
    
#%%
import iterative_solve_suyu
potential_corrector = iterative_solve_suyu.IterativePotentialCorrect(
    masked_imaging,
    shape_2d_dpsi=(100, 100),
    shape_2d_src=(50, 50),
)
potential_corrector.initialize_iteration(
    psi_2d_start=psi_2d_macro, 
    niter=10000, 
    lam_s_start=10, #source is over-regularized
    lam_dpsi_start=1.2*1e9,
    lam_dpsi_type='2nd',
    psi_anchor_points=np.array([(0.0,-1.1),(0.0,1.1),(-1.1,0.0)]),
    check_converge_points=positions,
    subhalo_fiducial_point=(1.15, 0.0),
)
potential_corrector.run_iter_solve()