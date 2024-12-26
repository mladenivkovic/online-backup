#!/usr/bin/env python3

# Create mass maps of baryons, DM, and
# stars, and dump them into a hdf5 file.


import swiftsimio
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
import unyt
import h5py


srcfile = "eagle_0000.hdf5"

data = swiftsimio.load(srcfile)
meta = data.metadata


imshow_kwargs = {"origin": "lower",
        "extent" : [0., meta.boxsize[0].to("Mpc"), 0., meta.boxsize[1].to("Mpc")],
        }
projection_kwargs = {"resolution": 1024, "parallel": True}

#  data.gas.m2 = data.gas.masses**2
#  m2_map = swiftsimio.visualisation.projection.project_gas(
#      data, project="m2", **projection_kwargs
#  )
mass_map = swiftsimio.visualisation.projection.project_gas(
    data, project="masses", **projection_kwargs
)


#  data.stars.m2 = data.gas.masses**2
#  sm2_map = swiftsimio.visualisation.projection.project_pixel_grid(
#      data=data.stars, boxsize=meta.boxsize, project="m2", **projection_kwargs
#  )

smass_map = swiftsimio.visualisation.projection.project_pixel_grid(
    data=data.stars, boxsize=meta.boxsize, project="masses", **projection_kwargs
)




data.dark_matter.smoothing_length = generate_smoothing_lengths(
    data.dark_matter.coordinates,
    data.metadata.boxsize,
    kernel_gamma=1.8,
    neighbours=57,
    speedup_fac=2,
    dimension=3,
)

#  data.dark_matter.m2 = data.dark_matter.masses**2
#  dm2_map = swiftsimio.visualisation.projection.project_pixel_grid(
#      data=data.dark_matter, boxsize=meta.boxsize, project="m2", **projection_kwargs
#  )

dmmass_map = swiftsimio.visualisation.projection.project_pixel_grid(
    data=data.dark_matter, boxsize=meta.boxsize, project="masses", **projection_kwargs
)





density_units = 10**6 * unyt.M_Sun / unyt.kpc**2

#  gas_mass_map = m2_map / mass_map
gas_mass_map = mass_map
gas_mass_map = gas_mass_map.to(density_units)
#  im1 = ax1.imshow(gas_mass_map.T, norm=LogNorm(), **imshow_kwargs)
#  set_colorbar(ax1, im1)

#  star_zeros = smass_map == 0.
star_nonzeros = smass_map > 0.
#  sm2_map += 1
smass_map += 1
#  star_mass_map = sm2_map / smass_map
star_mass_map = smass_map * mass_map.units
star_mass_map = star_mass_map.to(density_units)
stars_min = star_mass_map[star_nonzeros].min()
#  star_min = np.min(star_mass_map[star_nonzeros])
#  star_rm

print("linthresh stars", stars_min)
#  im2 = ax2.imshow(star_mass_map.T, norm=SymLogNorm(linthresh=stars_min), cmap="plasma",  **imshow_kwargs)
#  set_colorbar(ax2, im2)

dm_mass_map = dmmass_map * mass_map.units
dm_mass_map = dm_mass_map.to(density_units)
#  im3 = ax3.imshow(dm_mass_map.T, norm=LogNorm(), cmap="inferno", **imshow_kwargs)
#  set_colorbar(ax3, im3)

dump  = h5py.File("mass_map-"+srcfile, "w")

mm = dump.create_dataset("mass_map", mass_map.shape, dtype="f")
mm[:] = mass_map[:]

sm = dump.create_dataset("smass_map", smass_map.shape, dtype="f")
sm[:] = smass_map[:]
sm.attrs["stars_min"] = stars_min

dm = dump.create_dataset("dm_mass_map", dm_mass_map.shape, dtype="f")
dm[:] = dm_mass_map[:]

metaWrite = dump.create_group("Meta")
metaWrite.attrs["boxsize_Mpc"] = meta.boxsize[0].to("Mpc")

dump.close()

