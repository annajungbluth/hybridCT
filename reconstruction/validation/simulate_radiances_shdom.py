import at3d
import numpy as np
import xarray as xr
from collections import OrderedDict
import pylab as py

cloud_id = 55
time = 30.
wvl = 0.67
dx, dy = 0.04 # resolution
cloud_fname = f"wc_les_RICO_{dx*1e3:.0f}m_80kmx80km_T_qc_{time:.1f}h_id{cloud_id:03d}_pad.nc"
cloud_scatterer = at3d.util.load_from_netcdf(cloud_fname, density='lwc')
cloud_scatterer["reff"] = xr.ones_like(cloud_scatterer.density)*10
cloud_scatterer["delx"] = dx
cloud_scatterer["dely"] = dx
cloud_scatterer.x.data[:] = np.arange(0, len(cloud_scatterer.x.data)*dx, dx)
cloud_scatterer.y.data[:] = np.arange(0, len(cloud_scatterer.y.data)*dx, dx)
cloud_scatterer.z.data[:] = np.arange(0, len(cloud_scatterer.z.data)*dx, dx) + 1

#load atmosphere
atmosphere = xr.open_dataset('~/AT3D/data/ancillary/AFGL_summer_mid_lat.nc')
reduced_atmosphere = atmosphere.sel({'z': atmosphere.coords['z'].data[atmosphere.coords['z'].data <= 4.0]})
merged_z_coordinate = at3d.grid.combine_z_coordinates([reduced_atmosphere,cloud_scatterer])
# define the property grid - which is equivalent to the base RTE grid
rte_grid = at3d.grid.make_grid(cloud_scatterer.x.diff('x')[0],cloud_scatterer.x.data.size,
                          cloud_scatterer.y.diff('y')[0],cloud_scatterer.y.data.size,
                          np.append(0,cloud_scatterer.z.data))
cloud_scatterer_on_rte_grid = at3d.grid.resample_onto_grid(rte_grid, cloud_scatterer)

#We choose a gamma size distribution and therefore need to define a 'veff' variable.
size_distribution_function = at3d.size_distribution.gamma

cloud_scatterer_on_rte_grid['veff'] = (cloud_scatterer_on_rte_grid.reff.dims,
                                       np.full_like(cloud_scatterer_on_rte_grid.reff.data, fill_value=0.1))


# Define Sensors
sensor_dict = at3d.containers.SensorsDict()
off_nadir = np.array([70.5, 60., 45.6, 26.1])
sensor_zenith_list = np.hstack((-off_nadir, np.zeros(1), off_nadir[::-1]))
sensor_azimuth_list = [0]*5 + [180]*4
for zenith,azimuth in zip(sensor_zenith_list,sensor_azimuth_list):
    sensor_dict.add_sensor('MISR',
            at3d.sensor.orthographic_projection(wvl, cloud_scatterer, dx, dy, azimuth, zenith,
                                     altitude='TOA', stokes='I')
                          )

wavelengths = sensor_dict.get_unique_solvers()

# Get optical properties
mie_mono_tables = OrderedDict()
for wavelength in wavelengths:
    mie_mono_tables[wavelength] = at3d.mie.get_mono_table(
        'Water',(wavelength,wavelength),
        max_integration_radius=65.0,
        minimum_effective_radius=0.1,
        relative_dir='mie_tables',
        verbose=False
    )

optical_property_generator = at3d.medium.OpticalPropertyGenerator(
    'cloud', 
    mie_mono_tables,
    size_distribution_function,
    reff=np.linspace(5.0,30.0,30),
    veff=np.linspace(0.03,0.2,9),
)
optical_properties = optical_property_generator(cloud_scatterer_on_rte_grid)


# one function to generate rayleigh scattering.
rayleigh_scattering = at3d.rayleigh.to_grid(wavelengths,atmosphere,rte_grid)


# Define Solvers

# NB IF YOU REDEFINE THE SENSORS BUT KEEP THE SAME SET OF SOLVERS
# THERE IS NO NEED TO REDEFINE THE SOLVERS YOU CAN SIMPLY RERUN
# THE CELL BELOW WITHOUT NEEDING TO RERUN THE RTE SOLUTION.

solvers_dict = at3d.containers.SolversDict()
# note we could set solver dependent surfaces / sources / numerical_config here
# just as we have got solver dependent optical properties.
sza = 40
solar_mu = np.deg2rad(sza)
solar_az = 140

for wavelength in sensor_dict.get_unique_solvers():
    medium = {
        'cloud': optical_properties[wavelength],
        'rayleigh':rayleigh_scattering[wavelength]
     }
    config = at3d.configuration.get_config()
    solvers_dict.add_solver(
        wavelength,
        at3d.solver.RTE(
            numerical_params=config,
            surface=at3d.surface.lambertian(0.0),
            source=at3d.source.solar(wavelength, solar_mu, solar_az),
            medium=medium,
            num_stokes=1#sensor_dict.get_minimum_stokes()[wavelength],
        )
   )


#%%time
# solve the 4 RTEs in parallel AND get the measurements.
sensor_dict.get_measurements(solvers_dict, n_jobs=4, verbose=True)

for instrument in sensor_dict:
    print(instrument)
    sensor_images = sensor_dict.get_images(instrument)
    nviews = len(sensor_images)
    fig, ax = py.subplots(1, nviews, figsize=(4*nviews,4))
    for i, s in enumerate(sensor_images):
        s.I.T.plot(ax=ax[i], add_colorbar=False)
        if i>0:
            ax[i].set_ylabel("")
"""
for instrument in sensor_dict:
    sensor_images = sensor_dict.get_images(instrument)
    for sensor in sensor_images:
        py.figure()
        sensor.I.T.plot()
        py.title(instrument)
"""


at3d.util.save_forward_model(f'radiances_phi0_wc_les_RICO_{dx*1e3:.0f}m_80kmx80km_T_qc_{time:.1f}h_id{cloud_id:03d}.nc', sensor_dict, solvers_dict)
