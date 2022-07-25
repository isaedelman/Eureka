import numpy as np
import numexpr as ne
import itertools


def heqramp(rampparams, t):
    """This function creates a model that fits the HST 'hook' using a
    rising exponential.

    Parameters
    ----------
    parmparams : ndarray (1D)
        The parameters to use for the heqramp model
        (t0, r0, r1, r2, r3, pm, period).
    t :	ndarray (1D)
        Array of time/phase points.

    Returns
    -------
    ndarray
        The heqramp model evaluated at times t with parameters rampparams.

    Notes
    -----
    History:

    - 2014-06-09, Kevin Stevenson kbs@uchicago.edu
        Modified from hook2.py
    """
    t0 = rampparams[0]
    r0 = rampparams[1]
    r1 = rampparams[2]
    r2 = rampparams[3]
    r3 = rampparams[4]
    pm = rampparams[5]
    period = rampparams[6]
    
    t_batch = (t-t[0]-t0) % period
    
    return ne.evaluate('1 + pm*exp(-r0*t_batch + r1) + r2*t_batch + ' +
                       'r3*t_batch**2')


def zhouramp(rampparams, t, framenum):
    """This function creates a model that fits the HST 'hook' using the
    functional form from Zhou et al (2017).

    Parameters
    ----------
    parmparams : ndarray (1D)
        The parameters to use for the zhouramp model
        (Es, Ef, dEs, dEf, etas, etaf, Estot, Eftot, taus, tauf, flux).
    t :	ndarray (1D)
        Array of time/phase points.
    framenum : ?
        ?

    Returns
    -------
    ndarray
        The zhouramp model evaluated at times t with parameters rampparams.

    Notes
    -----
    History:

    - 2018-11-20, Kevin Stevenson kbs@stsci.edu
        Modified from heqramp.py
    """
    
    Es, Ef, dEs, dEf, etas, etaf, Estot, Eftot, taus, tauf, flux = rampparams
    istart = np.where(np.ediff1d(framenum) < 0)[0]+1
    t_batch = t-t[0]
    for ii in istart:
        t_batch[ii:] = t[ii:]-t[ii]
    iorbit2 = istart[0]
    deltaEs = np.zeros(len(t))
    deltaEf = np.zeros(len(t))
    deltaEs[iorbit2:] = dEs
    deltaEf[iorbit2:] = dEf
    
    cs = etas*flux/Estot+1./taus
    cf = etaf*flux/Eftot+1./tauf
    return (flux
            - (etas*flux-(Es+deltaEs)*cs)*np.exp(-t_batch*cs)
            - (etaf*flux-(Ef+deltaEf)*cf)*np.exp(-t_batch*cf))


def recte(rampparams, t):
    """This function creates a model that fits the HST 'hook' using the
    functional form from Zhou et al (2017).

    Parameters
    ----------
    rampparams : ndarray (1D)
        Array of ramp parameters (Es, Ef, dEs, dEf, flux).
    t : ndarray (1D)
        Array of time/phase points.

    Returns
    -------
    ndarray
        The recte model evaluated at times t with parameters rampparams.

    Notes
    -----
    History:

    - 2018-11-20, Kevin Stevenson kbs@stsci.edu
        Modified from heqramp.py
    """
    Es, Ef, dEs, dEf, flux = rampparams
    cRates = np.zeros(len(t)) + flux
    time = (t-t[0])*86400
    exptime = 103.129
    
    obsCounts = RECTE(cRates, t, exptime, Es, Ef, dEs, dEf)
    
    return obsCounts


def RECTE(cRates, tExp, exptime=180, trap_pop_s=0, trap_pop_f=0, dTrap_s=0,
          dTrap_f=0, dt0=0, mode='scanning'):
    """Hubble Space Telescope ramp effect model

    Parameters
    ----------
    cRates : ndarray
        Intrinsic count rate of each exposures, unit e/s.
    tExp : ndarray
        Start time of every exposures.
    expTime : float; optional
        Exposure time of the time series. Defaults to 180.
    trap_pop_s : int; optional
        Number of occupied traps at the beginning of the observations.
        Defaults to 0.
    trap_pop_f : int; optional
        Number of occupied traps at the beginning of the observations.
        Defaults to 0.
    dTrap_s : int; optional
        Number of extra trap added in the gap between two orbits.
        Defaults to 0.
    dTrap_f : int; optional
        Number of extra trap added in the gap between two orbits.
        Defaults to 0.
    dt0 : int; optional
        Possible exposures before very beginning, e.g., possible guiding
        adjustment. Defaults to 0.
    mode : str; optional
        The mode of operations from 'scanning', 'staring', or others.
        For scanning mode observation, the pixel no longer receive photons
        during the overhead time, in staring mode, the pixel keps receiving
        electrons. Defaults to 'scanning'.

    Returns
    -------
    obsCounts : ndarray
        The RECTE model.

    Notes
    -----
    History:

    - Original author: Daniel Apai
    - Version 0.1
        Adapted original IDL code to python by Yifan Zhou
    - Version 0.2
        Add extra keyword parameter to indicate scan or staring
        mode observations for staring mode, the detector receive flux in the
        same rate during overhead time as that during exposure
        precise mathematics forms are included
    - Version 0.2.1
        Introduce two types of traps, slow traps and fast traps
    - Version 0.3
        Fixing trapping parameters
    """
    nTrap_s = 1525.38  # 1320.0
    eta_trap_s = 0.013318  # 0.01311
    tau_trap_s = 1.63e4
    nTrap_f = 162.38
    eta_trap_f = 0.008407
    tau_trap_f = 281.463
    try:
        dTrap_f = itertools.cycle(dTrap_f)
        dTrap_s = itertools.cycle(dTrap_s)
        dt0 = itertools.cycle(dt0)
    except TypeError:
        dTrap_f = itertools.cycle([dTrap_f])
        dTrap_s = itertools.cycle([dTrap_s])
        dt0 = itertools.cycle([dt0])
    obsCounts = np.zeros(len(tExp))
    trap_pop_s = min(trap_pop_s, nTrap_s)
    trap_pop_f = min(trap_pop_f, nTrap_f)
    for i in range(len(tExp)):
        try:
            dt = tExp[i+1] - tExp[i]
        except IndexError:
            dt = exptime
        f_i = cRates[i]
        c1_s = eta_trap_s * f_i / nTrap_s + 1 / tau_trap_s  # a key factor
        c1_f = eta_trap_f * f_i / nTrap_f + 1 / tau_trap_f
        
        # number of trapped electron during one exposure
        dE1_s = ((eta_trap_s * f_i / c1_s - trap_pop_s) *
                 (1 - np.exp(-c1_s * exptime)))
        dE1_f = ((eta_trap_f * f_i / c1_f - trap_pop_f) *
                 (1 - np.exp(-c1_f * exptime)))
        dE1_s = min(trap_pop_s + dE1_s, nTrap_s) - trap_pop_s
        dE1_f = min(trap_pop_f + dE1_f, nTrap_f) - trap_pop_f
        trap_pop_s = min(trap_pop_s + dE1_s, nTrap_s)
        trap_pop_f = min(trap_pop_f + dE1_f, nTrap_f)
        obsCounts[i] = f_i * exptime - dE1_s - dE1_f
        
        # whether next exposure is in next batch of exposures
        if dt < 5 * exptime:
            # same orbits
            if mode == 'scanning':
                # scanning mode, no incoming flux between exposures
                dE2_s = - trap_pop_s * (1 - np.exp(-(dt - exptime)/tau_trap_s))
                dE2_f = - trap_pop_f * (1 - np.exp(-(dt - exptime)/tau_trap_f))
            elif mode == 'staring':
                # for staring mode, there is flux between exposures
                dE2_s = ((eta_trap_s * f_i / c1_s - trap_pop_s) *
                         (1 - np.exp(-c1_s * (dt - exptime))))
                dE2_f = ((eta_trap_f * f_i / c1_f - trap_pop_f) *
                         (1 - np.exp(-c1_f * (dt - exptime))))
            else:
                # others, same as scanning
                dE2_s = - trap_pop_s * (1 - np.exp(-(dt - exptime)/tau_trap_s))
                dE2_f = - trap_pop_f * (1 - np.exp(-(dt - exptime)/tau_trap_f))
            trap_pop_s = min(trap_pop_s + dE2_s, nTrap_s)
            trap_pop_f = min(trap_pop_f + dE2_f, nTrap_f)
        elif dt < 1200:
            # considering in orbit download scenario
            trap_pop_s = min(trap_pop_s * np.exp(-(dt-exptime)/tau_trap_s),
                             nTrap_s)
            trap_pop_f = min(trap_pop_f * np.exp(-(dt-exptime)/tau_trap_f),
                             nTrap_f)
        else:
            # switch orbit
            dt0_i = next(dt0)
            trap_pop_s = min(trap_pop_s*np.exp(-(dt-exptime-dt0_i)/tau_trap_s)
                             + next(dTrap_s), nTrap_s)
            trap_pop_f = min(trap_pop_f*np.exp(-(dt-exptime-dt0_i)/tau_trap_f)
                             + next(dTrap_f), nTrap_f)
            f_i = cRates[i + 1]
            c1_s = eta_trap_s*f_i/nTrap_s+1/tau_trap_s  # a key factor
            c1_f = eta_trap_f*f_i/nTrap_f+1/tau_trap_f
            dE3_s = (eta_trap_s*f_i/c1_s-trap_pop_s)*(1-np.exp(-c1_s*dt0_i))
            dE3_f = (eta_trap_f*f_i/c1_f-trap_pop_f)*(1-np.exp(-c1_f*dt0_i))
            dE3_s = min(trap_pop_s+dE3_s, nTrap_s)-trap_pop_s
            dE3_f = min(trap_pop_f+dE3_f, nTrap_f)-trap_pop_f
            trap_pop_s = min(trap_pop_s+dE3_s, nTrap_s)
            trap_pop_f = min(trap_pop_f+dE3_f, nTrap_f)
        trap_pop_s = max(trap_pop_s, 0)
        trap_pop_f = max(trap_pop_f, 0)

    return obsCounts
