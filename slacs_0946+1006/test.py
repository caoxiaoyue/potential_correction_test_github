#%%
import autolens as al 
import autolens.plot as aplt
import autofit as af
import pickle
import gzip

with gzip.open('./output/res_1.pklz','rb') as f:
    res_1 = pickle.load(f)

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
# imaging_plotter = aplt.ImagingPlotter(imaging=masked_imaging)
# imaging_plotter.subplot_imaging()

agg = af.Aggregator.from_database(
   filename=f"result/slacs0946/par_src/result.sqlite", completed_only=True
)
fit_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_imaging_gen = fit_agg.max_log_likelihood_gen_from()
fit_imaging = list(fit_imaging_gen)[0]
lens = fit_imaging.tracer.galaxies[0]
source = fit_imaging.tracer.galaxies[1]
z_l = lens.redshift
z_s = source.redshift

# al.convert.axis_ratio_and_angle_from(lens.mass.elliptical_comps)


settings_lens = al.SettingsLens(
    positions_threshold=0.5,
)
analysis = al.AnalysisImaging(
    dataset=masked_imaging,
    positions=res_1.image_plane_multiple_image_positions,
    settings_lens=settings_lens,
    settings_pixelization=al.SettingsPixelization(use_border=True)
)

#%%