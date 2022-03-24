import numpy as np


def compute_stack(wave_grid, waves, fluxes, ivars, masks, weights):
    '''
        Compute a stacked spectrum from a set of exposures on the specified wave_grid with proper treatment of
        weights and masking. This code uses np.histogram to combine the data using NGP and does not perform any
        interpolations and thus does not correlate errors. It uses wave_grid to determine the set of wavelength bins that
        the data are averaged on. The final spectrum will be on an ouptut wavelength grid which is not the same as wave_grid.
        The ouput wavelength grid is the weighted average of the individual wavelengths used for each exposure that fell into
        a given wavelength bin in the input wave_grid. This 1d coadding routine thus maintains the independence of the
        errors for each pixel in the combined spectrum and computes the weighted averaged wavelengths of each pixel
        in an analogous way to the 2d extraction procedure which also never interpolates to avoid correlating erorrs.
        
        Args:
        wave_grid: ndarray, (ngrid +1,)
        new wavelength grid desired. This will typically be a reguarly spaced grid created by the get_wave_grid routine.
        The reason for the ngrid+1 is that this is the general way to specify a set of  bins if you desire ngrid
        bin centers, i.e. the output stacked spectra have ngrid elements.  The spacing of this grid can be regular in
        lambda (better for multislit) or log lambda (better for echelle). This new wavelength grid should be designed
        with the sampling of the data in mind. For example, the code will work fine if you choose the sampling to be
        too fine, but then the number of exposures contributing to any given wavelength bin will be one or zero in the
        limiting case of very small wavelength bins. For larger wavelength bins, the number of exposures contributing
        to a given bin will be larger.
        waves: ndarray, (nspec, nexp)
        wavelength arrays for spectra to be stacked. Note that the wavelength grids can in general be different for
        each exposure and irregularly spaced.
        fluxes: ndarray, (nspec, nexp)
        fluxes for each exposure on the waves grid
        ivars: ndarray, (nspec, nexp)
        Inverse variances for each exposure on the waves grid
        masks: ndarray, bool, (nspec, nexp)
        Masks for each exposure on the waves grid. True=Good.
        weights: ndarray, (nspec, nexp)
        Weights to be used for combining your spectra. These are computed using sn_weights
        Returns:
        wave_stack, flux_stack, ivar_stack, mask_stack, nused
        
        wave_stack: ndarray, (ngrid,)
        Wavelength grid for stacked spectrum. As discussed above, this is the weighted average of the wavelengths
        of each spectrum that contriuted to a bin in the input wave_grid wavelength grid. It thus has ngrid
        elements, whereas wave_grid has ngrid+1 elements to specify the ngrid total number of bins. Note that
        wave_stack is NOT simply the wave_grid bin centers, since it computes the weighted average.
        flux_stack: ndarray, (ngrid,)
        Final stacked spectrum on wave_stack wavelength grid
        ivar_stack: ndarray, (ngrid,)
        Inverse variance spectrum on wave_stack wavelength grid. Erors are propagated according to weighting and
        masking.
        mask_stack: ndarray, bool, (ngrid,)
        Mask for stacked spectrum on wave_stack wavelength grid. True=Good.
        nused: ndarray, (ngrid,)
        Numer of exposures which contributed to each pixel in the wave_stack. Note that this is in general
        different from nexp because of masking, but also becuse of the sampling specified by wave_grid. In other
        words, sometimes more spectral pixels in the irregularly gridded input wavelength array waves will land in
        one bin versus another depending on the sampling.
        '''
    
    ubermask = masks & (weights > 0.0) & (waves > 1.0) & (ivars > 0.0)
    waves_flat = waves[ubermask].flatten()
    fluxes_flat = fluxes[ubermask].flatten()
    ivars_flat = ivars[ubermask].flatten()
    vars_flat = 1 / ivars_flat
    weights_flat = weights[ubermask].flatten()
    
    # Counts how many pixels in each wavelength bin
    nused, wave_edges = np.histogram(waves_flat,bins=wave_grid,density=False)
    
    # Calculate the summed weights for the denominator
    weights_total, wave_edges = np.histogram(waves_flat,bins=wave_grid,density=False,weights=weights_flat)
    
    # Calculate the stacked wavelength
    wave_stack_total, wave_edges = np.histogram(waves_flat,bins=wave_grid,density=False,weights=waves_flat*weights_flat)
    wave_stack = (weights_total > 0.0)*wave_stack_total/(weights_total+(weights_total==0.))
    
    # Calculate the stacked flux
    flux_stack_total, wave_edges = np.histogram(waves_flat,bins=wave_grid,density=False,weights=fluxes_flat*weights_flat)
    flux_stack = (weights_total > 0.0)*flux_stack_total/(weights_total+(weights_total==0.))
    
    # Calculate the stacked ivar
    var_stack_total, wave_edges = np.histogram(waves_flat,bins=wave_grid,density=False,weights=vars_flat*weights_flat**2)
    var_stack = (weights_total > 0.0)*var_stack_total/(weights_total+(weights_total==0.))**2
    ivar_stack = 1 / var_stack
    
    # New mask for the stack
    mask_stack = (weights_total > 0.0) & (nused > 0.0)
    
    return wave_stack, flux_stack, ivar_stack, mask_stack, nused
