import numpy as np
import astropy.constants as const

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

import pymc3 as pm
import starry
starry.config.quiet = True
starry.config.lazy = True

from . import PyMC3Model


class temp_class:
    def __init__(self):
        pass


class StarryModel(PyMC3Model):
    def __init__(self, **kwargs):
        # Inherit from PyMC3Model class
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        required = np.array(['Ms', 'Rs'])
        missing = np.array([name not in self.paramtitles for name in required])
        if np.any(missing):
            message = (f'Missing required params {required[missing]} in your '
                       f'EPF.')
            raise AssertionError(message)

        if 'u2' in self.paramtitles:
            self.udeg = 2
        elif 'u1' in self.paramtitles:
            self.udeg = 1
        else:
            self.udeg = 0
        if 'AmpCos2' in self.paramtitles or 'AmpSin2' in self.paramtitles:
            self.ydeg = 2
        elif 'AmpCos1' in self.paramtitles or 'AmpSin1' in self.paramtitles:
            self.ydeg = 1
        else:
            self.ydeg = 0

    def setup(self, time, flux, scatter_array, full_model):
        self.time = time
        self.flux = flux
        self.scatter_array = scatter_array
        self.full_model = full_model
        self.linearized = self.full_model.linearized

        self.systems = []
        for c in range(self.nchan):
            # To save ourselves from tonnes of getattr lines, let's make a
            # new object without the _c parts of the parnames
            # For example, this way we can do `temp.u1` rather than
            # `getattr(self.model, 'u1_'+c)`.
            temp = temp_class()
            for key in self.paramtitles:
                if self.linearized and key in ['fp', 'AmpCos1', 'AmpSin1',
                                               'AmpCos2', 'AmpSin2']:
                    continue
                ptype = getattr(self.parameters, key).ptype
                if (ptype not in ['fixed', 'independent']
                        and c > 0):
                    # Remove the _c part of the parname but leave any
                    # other underscores intact
                    setattr(temp, key, getattr(self.model, key+'_'+str(c)))
                else:
                    setattr(temp, key, getattr(self.model, key))
            
            # Initialize star object
            star = starry.Primary(starry.Map(udeg=self.udeg),
                                  m=temp.Ms, r=temp.Rs)

            if hasattr(self.parameters, 'limb_dark'):
                if self.parameters.limb_dark.value == 'kipping2013':
                    # Transform stellar variables to uniform used by starry
                    star.map[1] = 2*tt.sqrt(temp.u1)*temp.u2
                    star.map[2] = tt.sqrt(temp.u1)*(1-2*temp.u2)
                elif self.parameters.limb_dark.value == 'quadratic':
                    star.map[1] = temp.u1
                    star.map[2] = temp.u2
                elif self.parameters.limb_dark.value == 'linear':
                    star.map[1] = temp.u1
                elif self.parameters.limb_dark.value != 'uniform':
                    message = (f'ERROR: starryModel is not yet able to '
                               f'handle {self.parameters.limb_dark.value} '
                               f'limb darkening.\n'
                               f'       limb_dark must be one of uniform, '
                               f'linear, quadratic, or kipping2013.')
                    raise ValueError(message)
            
            # Solve Keplerian orbital period equation for Mp
            # (otherwise starry is going to mess with P or a...)
            a = temp.a*temp.Rs*const.R_sun.value
            p = temp.per*(24.*3600.)
            Mp = (((2.*np.pi*a**(3./2.))/p)**2/const.G.value/const.M_sun.value
                  - temp.Ms)

            if not hasattr(temp, 'fp'):
                temp.fp = 0

            if not self.linearized:
                planet_map = starry.Map(ydeg=self.ydeg, amp=temp.fp)
            else:
                planet_map = starry.Map(ydeg=self.ydeg)

            # Initialize planet object
            planet = starry.Secondary(
                planet_map,
                # Convert mass to M_sun units
                # m=temp.Mp*const.M_jup.value/const.M_sun.value,
                m=Mp,
                # Convert radius to R_star units
                r=temp.rp*temp.Rs,
                # Setting porb here overwrites a
                a=temp.a,
                # porb = temp.per,
                # prot = temp.per,
                # Another option to set inclination using impact parameter
                # inc=tt.arccos(b/a)*180/np.pi
                inc=temp.inc,
                ecc=temp.ecc,
                w=temp.w
            )
            # Setting porb here may not override a
            planet.porb = temp.per
            # Setting prot here may not override a
            planet.prot = temp.per
            if not self.linearized:
                if hasattr(temp, 'AmpCos1'):
                    planet.map[1, 0] = temp.AmpCos1
                if hasattr(temp, 'AmpSin1'):
                    planet.map[1, 1] = temp.AmpSin1
                if self.ydeg == 2:
                    if hasattr(temp, 'AmpCos2'):
                        planet.map[2, 0] = temp.AmpCos2
                    if hasattr(temp, 'AmpSin2'):
                        planet.map[2, 1] = temp.AmpSin2
            # Offset is controlled by Y11
            planet.theta0 = 180.0
            planet.t0 = temp.t0

            # Instantiate the system
            sys = starry.System(star, planet)
            
            if self.linearized and self.ydeg > 0:
                sys.set_data(self.flux/self.full_model.syseval(eval=False),
                             C=self.scatter_array**2)

                # Prior on map parameters
                planet_mu = np.zeros(planet.map.Ny)
                planet_mu[0] = 1e-4
                planet_L = 1e-7*np.ones(planet.map.Ny)
                planet.map.set_prior(mu=planet_mu, L=planet_L)

                pm.Potential("marginal", sys.lnlike(t=self.time))
            
            self.systems.append(sys)

    def eval(self, eval=True, channel=None, **kwargs):
        if channel is None:
            nchan = self.nchan
            channels = np.arange(nchan)
        else:
            nchan = 1
            channels = [channel, ]

        if eval:
            lib = np
            systems = self.fit.systems
        else:
            lib = tt
            systems = self.systems

        phys_flux = lib.zeros(0)
        for c in channels:
            lcpiece = systems[c].flux(self.time)
            if eval:
                lcpiece = lcpiece.eval()
        phys_flux = lib.concatenate([phys_flux, lcpiece])

        return phys_flux

    def update(self, newparams):
        super().update(newparams)

        self.fit.systems = []
        for c in range(self.nchan):
            # To save ourselves from tonnes of getattr lines, let's make a
            # new object without the _c parts of the parnames
            # For example, this way we can do `temp.u1` rather than
            # `getattr(self.model, 'u1_'+c)`.
            temp = temp_class()
            for key in self.paramtitles:
                if self.linearized and key in ['fp', 'AmpCos1', 'AmpSin1',
                                               'AmpCos2', 'AmpSin2']:
                    continue
                ptype = getattr(self.parameters, key).ptype
                if (ptype not in ['fixed', 'independent']
                        and c > 0):
                    # Remove the _c part of the parname but leave any
                    # other underscores intact
                    setattr(temp, key, getattr(self.fit, key+'_'+str(c)))
                else:
                    setattr(temp, key, getattr(self.fit, key))

            # Initialize star object
            star = starry.Primary(starry.Map(udeg=self.udeg),
                                  m=temp.Ms, r=temp.Rs)

            if hasattr(self.parameters, 'limb_dark'):
                if self.parameters.limb_dark.value == 'kipping2013':
                    # Transform stellar variables to uniform used by starry
                    star.map[1] = 2*np.sqrt(temp.u1)*temp.u2
                    star.map[2] = np.sqrt(temp.u1)*(1-2*temp.u2)
                elif self.parameters.limb_dark.value == 'quadratic':
                    star.map[1] = temp.u1
                    star.map[2] = temp.u2
                elif self.parameters.limb_dark.value == 'linear':
                    star.map[1] = temp.u1
                elif self.parameters.limb_dark.value != 'uniform':
                    message = (f'ERROR: starryModel is not yet able to handle '
                               f'{self.parameters.limb_dark.value} '
                               f'limb_dark.\n'
                               f'       limb_dark must be one of uniform, '
                               f'linear, quadratic, or kipping2013.')
                    raise ValueError(message)
            
            # Solve Keplerian orbital period equation for Mp
            # (otherwise starry is going to mess with P or a...)
            a = temp.a*temp.Rs*const.R_sun.value
            p = temp.per*(24.*3600.)
            Mp = (((2.*np.pi*a**(3./2.))/p)**2/const.G.value/const.M_sun.value
                  - temp.Ms)

            if not hasattr(temp, 'fp'):
                temp.fp = 0

            if not self.linearized:
                planet_map = starry.Map(ydeg=self.ydeg, amp=temp.fp)
            else:
                planet_map = starry.Map(ydeg=self.ydeg)

            # Initialize planet object
            planet = starry.Secondary(
                planet_map,
                # Convert mass to M_sun units
                # m=temp.Mp*const.M_jup.value/const.M_sun.value,
                m=Mp,
                # Convert radius to R_star units
                r=temp.rp*temp.Rs,
                # Setting porb here overwrites a
                a=temp.a,
                # porb = temp.per,
                # prot = temp.per,
                # Another option to set inclination using impact parameter
                # inc=tt.arccos(b/a)*180/np.pi
                inc=temp.inc,
                ecc=temp.ecc,
                w=temp.w
            )
            # Setting porb here may not override a
            planet.porb = temp.per
            # Setting prot here may not override a
            planet.prot = temp.per
            if not self.linearized:
                if hasattr(temp, 'AmpCos1'):
                    planet.map[1, 0] = temp.AmpCos1
                if hasattr(temp, 'AmpSin1'):
                    planet.map[1, 1] = temp.AmpSin1
                if self.ydeg == 2:
                    if hasattr(temp, 'AmpCos2'):
                        planet.map[2, 0] = temp.AmpCos2
                    if hasattr(temp, 'AmpSin2'):
                        planet.map[2, 1] = temp.AmpSin2
            # Offset is controlled by Y11
            planet.theta0 = 180.0
            planet.t0 = temp.t0

            # Instantiate the system
            sys = starry.System(star, planet)

            if self.linearized and self.ydeg > 0:
                sys.set_data(self.flux/self.full_model.syseval(eval=True),
                             C=self.scatter_array**2)
                
                # Reapply prior on map parameters
                planet_mu = np.zeros(planet.map.Ny)
                planet_mu[0] = 1e-4
                planet_L = 1e-7*np.ones(planet.map.Ny)
                planet.map.set_prior(mu=planet_mu, L=planet_L)

                with self.full_model.model:
                    import pymc3_ext as pmx
                    x = pmx.eval_in_model(sys.solve(t=self.time)[0])
                x[1:] /= x[0]
                print(x)
                planet.map.amp = x[0]
                planet.map[1:, :] = x[1:]

            self.fit.systems.append(sys)

        return

    @property
    def time(self):
        """A getter for the time"""
        return self._time

    @time.setter
    def time(self, time_array, time_units='BJD'):
        """A setter for the time

        Parameters
        ----------
        time_array: sequence, astropy.units.quantity.Quantity
            The time array
        time_units: str
            The units of the input time_array, ['MJD', 'BJD', 'phase']
        """
        # Check the type
        if not isinstance(time_array, (np.ndarray, tuple, list)):
            raise TypeError("Time axis must be a tuple, list, or numpy array.")

        # Set the units
        self.time_units = time_units

        # Set the array
        # self._time = np.array(time_array)
        self._time = np.ma.masked_array(time_array)
