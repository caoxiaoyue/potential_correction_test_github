#%%
import autolens as al 
import autolens.plot as aplt

imaging = al.Imaging.from_fits(
    image_path='./data/data4autolens.fits',
    image_hdu=0,
    noise_map_path='./data/data4autolens.fits',
    noise_map_hdu=1,
    psf_path='./data/data4autolens.fits',
    psf_hdu=2,
    pixel_scales=0.05,
)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

# %%
mask_data = al.Mask2D.elliptical_annular(
    shape_native=imaging.shape_native,
    inner_major_axis_radius=0.0,
    inner_axis_ratio=1,
    inner_phi=175,
    outer_major_axis_radius=3.0,
    outer_axis_ratio=1,
    outer_phi=175,
    pixel_scales=0.05,
)

# mask_data = al.Mask2D.elliptical_annular(
#     shape_native=imaging.shape_native,
#     inner_major_axis_radius=0.7,
#     inner_axis_ratio=0.9,
#     inner_phi=175,
#     outer_major_axis_radius=2.1,
#     outer_axis_ratio=0.9,
#     outer_phi=175,
#     pixel_scales=0.05,
# )
masked_imaging = imaging.apply_mask(mask_data)
imaging_plotter = aplt.ImagingPlotter(imaging=masked_imaging)
imaging_plotter.subplot_imaging()

# %%
import autofit as af
lens_galaxy_model = af.Model(al.Galaxy, redshift=0.222, mass=al.mp.EllIsothermal)
source_galaxy_model = af.Model(al.Galaxy, redshift=0.609, bulge=al.lp.EllExponential)
model = af.Collection(
    galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)
)
search = af.DynestyStatic(
    path_prefix='./result',
    name='par_src',
    unique_tag='slacs0946',
    nlive=50,
    walks=5,
    # number_of_cores=16,
)
analysis = al.AnalysisImaging(dataset=masked_imaging)
result = search.fit(model=model, analysis=analysis)

import pickle
import gzip
with gzip.open('./output/res_1.pklz','wb') as f:
    pickle.dump(result,f)

# %%
# print(type(result.instance.galaxies.lens))
# mass = al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
# galaxy_this = al.Galaxy(
#     redshift=0.222, mass=mass
# )
# print(type(galaxy_this))

# %%
