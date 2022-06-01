#%%
import autolens as al 
import autolens.plot as aplt
import autofit as af
from scipy.optimize import dual_annealing

agg = af.Aggregator.from_database(
   filename=f"result/slacs0946/par_src/result.sqlite", completed_only=True
)
agg.add_directory(directory="output/result/slacs0946/par_src/")

fit_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_imaging_gen = fit_agg.max_log_likelihood_gen_from()
fit_imaging = list(fit_imaging_gen)[0]
lens = fit_imaging.tracer.galaxies[0]
source = fit_imaging.tracer.galaxies[1]
z_l = lens.redshift
z_s = source.redshift


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


ret = dual_annealing(
   optimize_log_reg, 
   bounds=[[-5,4],], 
   args=(masked_imaging, lens, z_s, (50, 50)),
)
print('best-fit info of regularization strength')
print('------------------------')
print(ret)
print('------------------------')
best_reg = 10**(ret.x) 
best_reg = float(best_reg) #6.32211064
# ------------------------
#      fun: -4150.144458668966
#  message: ['Maximum number of iteration reached']
#     nfev: 2079
#     nhev: 0
#      nit: 1000
#     njev: 39
#   status: 0
#  success: True
#        x: array([0.80086209])
# ------------------------


#%%
#------------show best-fit result
fitter = pixelized_src_fitter_from(
   masked_imaging,
   lens, 
   z_s,
   (50, 50),
   best_reg,
)


mat_plot_2d_1 = aplt.MatPlot2D(
   title=aplt.Title(label="Hey"),
   output=aplt.Output(
      path='./output',
      filename="3_src_1",
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
      filename="3_src_2",
      format="png",
   ),
   cmap=aplt.Cmap(),
)
inversion_plotter = aplt.InversionPlotter(
    inversion=fitter.inversion, mat_plot_2d=mat_plot_2d_2
)
inversion_plotter.figures_2d_of_mapper(mapper_index=0, reconstruction=True)

# %%
