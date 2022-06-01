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

#%%
epl_model = af.Model(al.mp.EllPowerLaw)
epl_model.centre = (0.0, 0.0)
epl_model.elliptical_comps.elliptical_comps_0 = af.GaussianPrior(
    mean=lens.mass.elliptical_comps[0], sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
epl_model.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(
    mean=lens.mass.elliptical_comps[1], sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
epl_model.einstein_radius = af.GaussianPrior(
    mean=lens.mass.einstein_radius, 
    sigma=0.25*lens.mass.einstein_radius,
    lower_limit=0.5,
    upper_limit=3.0,
)
epl_model.slope = 2.2

shear_model = af.Model(al.mp.ExternalShear)
shear_model.elliptical_comps.elliptical_comps_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
shear_model.elliptical_comps.elliptical_comps_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

pixelization = al.pix.DelaunayMagnification(shape=(50, 50))
regularization = al.reg.Constant(coefficient=6.32211064)

#%%
#--------------------------
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            mass=epl_model,
            shear=shear_model,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            pixelization=pixelization,
            regularization=regularization,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix='./result',
    name='pix_src_refine_mass_fix_slope',
    unique_tag='slacs0946',
    nlive=100,
    number_of_cores=5,
)

settings_lens = al.SettingsLens(
    positions_threshold=0.5,
)
analysis = al.AnalysisImaging(
    dataset=masked_imaging,
    positions=res_1.image_plane_multiple_image_positions,
    settings_lens=settings_lens,
    settings_pixelization=al.SettingsPixelization(use_border=True)
)

result_2 = search.fit(model=model, analysis=analysis)

#%%
with gzip.open('./output/res_2_fix_slope.pklz','wb') as f:
    pickle.dump(result_2,f)