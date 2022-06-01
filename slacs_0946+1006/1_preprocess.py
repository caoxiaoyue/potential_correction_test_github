#%%
import numpy as np 
from matplotlib import pyplot as plt
from astropy.io import fits
hdul = fits.open('./data/SLACSJ0946+1006_10886_54042_F814W_4_biz.fits')


#%%
image_data = hdul[0].data
image_weight = hdul[1].data
psf = hdul[-1].data
psf = psf / psf.sum()

feature_mask = hdul[4].data
junk_mask = hdul[5].data
bs_lens_light = hdul[7].data

#%%
dpix = 0.05
half_width = len(image_data)*0.5*dpix
extent = [-1.0*half_width, half_width, -1.0*half_width, half_width]

option={'origin':'lower', 'cmap':'jet', 'extent':extent}
plt.figure(figsize=(15,10))
plt.subplot(231)
plt.imshow(image_data, **option)
plt.colorbar()

plt.subplot(232)
plt.imshow(image_weight, **option)
plt.colorbar()

plt.subplot(233)
plt.imshow(psf, **option)
plt.colorbar()

plt.subplot(234)
plt.imshow(feature_mask, **option)
plt.colorbar()

plt.subplot(235)
plt.imshow(junk_mask, **option)
plt.colorbar()

plt.subplot(236)
plt.imshow(bs_lens_light, **option)
plt.colorbar()

plt.show()
plt.close()


# %%
image_weight = image_weight * junk_mask.astype('float')
image_noise = np.where(image_weight!=0.0, np.sqrt(1/image_weight), 1e8)
plt.figure()
plt.imshow(1/image_noise**2, **option)
plt.colorbar()
plt.show()
plt.close()


# %%
image_hdu = fits.PrimaryHDU(image_data-bs_lens_light)
image_hdu.header['dpix'] = 0.05
noise_hdu = fits.ImageHDU(image_noise)
psf_hdu = fits.ImageHDU(psf)
mask_irregular_hdu = fits.ImageHDU(feature_mask)

hdul = fits.HDUList([image_hdu, noise_hdu, psf_hdu, mask_irregular_hdu])
hdul.writeto('./data/data4autolens.fits',overwrite=True)

# %%
