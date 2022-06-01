#%%
import autolens as al 
import autolens.plot as aplt
import autofit as af
from scipy.optimize import dual_annealing

#%%
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
imaging_plotter = aplt.ImagingPlotter(imaging=masked_imaging)
imaging_plotter.subplot_imaging()

z_s = 0.609
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

#%%
def pixelized_src_fitter_from(
   masked_imaging,
   lens_galaxy, 
   z_s=1.0,
   pixelization_shape_2d=(50, 50),
   reg_strength=1.0,
):
   pixelization = al.pix.DelaunayMagnification(shape=pixelization_shape_2d)

   source_galaxy = al.Galaxy(
      redshift=z_s,
      pixelization=pixelization,
      regularization=al.reg.Constant(coefficient=reg_strength),
   )

   tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy]) 

   fitter = al.FitImaging(
      dataset=masked_imaging,
      tracer=tracer,
      settings_inversion=al.SettingsInversion(use_w_tilde=False),
      settings_pixelization=al.SettingsPixelization(use_border=True),
   )

   return fitter


def optimize_log_reg(
   x,
   masked_imaging,
   lens_galaxy,
   z_s, 
   pixelization_shape_2d, 
):
   fitter = pixelized_src_fitter_from(
      masked_imaging,
      lens_galaxy, 
      z_s,
      pixelization_shape_2d,
      10**(x[0]),
   )
   fit_log_evidence = fitter.log_evidence  
   return -1.0*fit_log_evidence

'''
ret = dual_annealing(
   optimize_log_reg, 
   bounds=[[-5,4],], 
   args=(masked_imaging, lens_galaxy, z_s, (50, 50)),
)
print('best-fit info of regularization strength')
print('------------------------')
print(ret)
print('------------------------')
'''

best_reg = 10**0.84812691 #10**(ret.x) 
best_reg = float(best_reg) 
print(best_reg)
# ------------------------
#      fun: -4661.943643979606
#  message: ['Maximum number of iteration reached']
#     nfev: 2047
#     nhev: 0
#      nit: 1000
#     njev: 23
#   status: 0
#  success: True
#        x: array([0.84812691])
# ------------------------


#%%
#------------show best-fit result
fitter = pixelized_src_fitter_from(
   masked_imaging,
   lens_galaxy, 
   z_s,
   (50, 50),
   best_reg,
)

mat_plot_2d_1 = aplt.MatPlot2D(
   title=aplt.Title(label="Hey"),
   output=aplt.Output(
      path='./output',
      filename="5_src_1",
      format="png",
   ),
)
include_2d = aplt.Include2D(mask=True)
fit_imaging_plotter = aplt.FitImagingPlotter(
   fit=fitter, include_2d=include_2d, mat_plot_2d=mat_plot_2d_1
)
fit_imaging_plotter.subplot_fit_imaging()


mat_plot_2d_2 = aplt.MatPlot2D(
   title=aplt.Title(label="Hey"),
   output=aplt.Output(
      path='./output',
      filename="5_src_2",
      format="png",
   ),
   cmap=aplt.Cmap(),
)
inversion_plotter = aplt.InversionPlotter(
    inversion=fitter.inversion, mat_plot_2d=mat_plot_2d_2
)
inversion_plotter.figures_2d_of_mapper(mapper_index=0, reconstruction=True)

# %%
