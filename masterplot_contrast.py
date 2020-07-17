import os
import astropy.io.fits as fits
import numpy as np
import scipy
import scipy.ndimage as ndi
import matplotlib.pylab as plt
import pyklip.klip
import pyklip.instruments.Instrument as Instrument
import pyklip.parallelized as parallelized
import pyklip.rdi as rdi
import pyklip.fakes as fakes
import glob
from astropy.table import Table
from astropy.table import join
from astropy.table import vstack
import pandas as pd
import pdb
from tqdm import tqdm
import sys

# Only test 1 through 9 for rdi 
# try specifying minrot to be 0, 3.4, 6.8 and 10 

all_corrected_contrasts = []
all_separations = []
all_minrots = []
all_kl = []
all_smooth_contrast = []

minrots = [0, 0.1, 3.4, 6.8]
for minrot in minrots:
    numbasis = [1,2,3,4]
    numbasis_index = [0,1,2,3]
    for i in numbasis_index:                     

        # Let's specifiy the important variables first. Change this as needed.

        mode = "ADI" 
        datadir = "notebooks/Eps_Eri_Sim/"
        filtername = "f444m"
        pas_list = [0, 10, 3.3, 6.7]
        rollnames_list = ["roll1", "roll2", "roll3", "roll4"]
        unocculted_psf = "offaxistarget.fits"
        center_x = 67 # Center (x) of the unocculted psf
        center_y = 67 # Center (y) of the unocculted psf
        IWA = 12 # Set this to be what you want your innermost contrast curve separation to be
        OWA = 45 # Set this to be what you want your outermost contrast curve separation to be

        # Mask for real planets
        mask = "off"
        x_positions= None
        y_positions= None

        #Read in transmission profile of coronagraph used
        coronagraph = pd.read_csv("notebooks/MASK430R.csv", names = ["rad_dist", "trans"])

        # Make lists of filenames
        roll_filenames_list =  ["Target/TargetCube.fits"]
        ref_filenames_list = "Reference/ReferenceCube.fits"

        # Change KLIP parameters here
        annuli = 1
        subsections = 1
        #movement = 1



        def generate_datasets(datadir, roll_filenames_list, ref_filenames_list, rollnames_list, pas_list, num_datasets=1, mode = 'ADI'):
            """
            Generates multiple generic datasets based on the two JWST roll angles 0° and 10°
            
            Args:
                datadir (str): The directory the data is contained in
                roll_filenames_list (list: str): A list of the names of all the files you'd like to read in for each roll angle
                ref_filenames_list (list:str): A list of the names of all the files you'd like to read in for the reference library
                rollnames_list (list: str): A list of the names you'd like to call your roll angles
                pas_list (list: float): A list of all the position/roll angles of your data
                num_datasets(int): Number of datasets to be generated. Default is 1.
                data_type (str): The type of data reduction you'd like to generate data for (adi or rdi). adi is default
            
            Returns:
                list: List of generated datasets
                list: psf library
            """
            if mode not in ('ADI','RDI'):
                raise ValueError(f"Uknown data type {mode}")
            
            # Use this function if mode is adi
            def _process_adi(datadir, roll_filenames_list, ref_filenames_list, rollnames_list, pas_list, num_datasets):
                datasets = []
                psflibs = []
                for dataset in range(num_datasets):
                    
                    # Read in your data
                    data = [fits.getdata(f"{datadir}{filename}") for filename in roll_filenames_list]
                    
                    
                    # Combine your data if you read in multiple roll angles (if you read in the full cube already, this function will still work and just return your cube)
                    full_seq = np.concatenate(data, axis=0)

                    # Create an array of all the roll angles. First check to see if each image has 1 or more frames 
                    if full_seq.shape[0] > len(pas_list):
                        for d in data:
                            pas = np.ravel([[pa]*d.shape[0] for pa in pas_list])
                            
                    elif full_seq.shape[0] == len(pas_list):
                        pas = np.array(pas_list)

                    # For each image, the (x,y) center where the star is is just the center of the image
                    centers = np.array([np.array(frame.shape) // 2.0 for frame in full_seq])
                    
                    # Give each roll angle a name so we can refer to them.  First check to see if each image has 1 or more frames 
                    if full_seq.shape[0] > len(rollnames_list):
                        rollnames = []
                        for rn in rollnames_list:
                            rollnames += [f"{rn}_{d}" for d in range(data[0].shape[0])]
                            
                    elif full_seq.shape[0] == len(rollnames_list):
                        rollnames = rollnames_list

                    # Define dataset
                    dataset = Instrument.GenericData(full_seq, centers, IWA=4, parangs=pas, filenames=rollnames)
                    dataset.flipx = False # Get the right handedness of the data
                    dataset.OWA = round((dataset.input.shape[-1])/2) # Set OWA
                    
                    psflib = None
                    if num_datasets > 1:
                        datasets.append(dataset)
                        psflibs.append(psflib)
                    else:
                        datasets = dataset
                        psflibs = psflib
                return datasets, psflibs
            
            # Use this function if mode is rdi
            def _process_rdi(datadir, roll_filenames_list, ref_filenames_list, rollnames_list, pas_list, num_datasets):
                
                datasets = []
                psflibs = []
                for dataset in range(num_datasets):
                    # Read in your data
                    data = [fits.getdata(f"{datadir}{filename}") for filename in roll_filenames_list]

                    # Combine your data if you read in multiple roll angles (if you read in the full cube already, this function will still work and just return your cube)
                    full_seq = np.concatenate(data, axis=0)

                    # Create an array of all the roll angles. First check to see if each image has 1 or more frames 
                    if full_seq.shape[0] > len(pas_list):
                        for d in data:
                            pas = np.ravel([[pa]*d.shape[0] for pa in pas_list])
                            
                    elif full_seq.shape[0] == len(pas_list):
                        pas = np.array(pas_list)

                    # For each image, the (x,y) center where the star is is just the center of the image
                    centers = np.array([np.array(frame.shape) // 2.0 for frame in full_seq])
                    
                    # Give each roll angle a name so we can refer to them.  First check to see if each image has 1 or more frames 
                    if full_seq.shape[0] > len(rollnames_list):
                        rollnames = []
                        for rn in rollnames_list:
                            rollnames += [f"{rn}_{d}" for d in range(data[0].shape[0])]
                            
                    elif full_seq.shape[0] == len(rollnames_list):
                        rollnames = rollnames_list

                    # Define dataset
                    dataset = Instrument.GenericData(full_seq, centers, IWA=4, parangs=pas, filenames=rollnames)
                    dataset.flipx = False # Get the right handedness of the data
                    dataset.OWA = round((dataset.input.shape[-1])/2) # Set OWA
                    
                    # read in ref star
                    with fits.open(f"{datadir}/{ref_filenames_list}") as hdulist:
                        ref_cube = hdulist[0].data
                    
                    # Combine both science target and reference target images into a psf library array
                    psflib_imgs = np.append(ref_cube, full_seq, axis=0)

                    ref_filenames = ["ref_{0}".format(i) for i in range(ref_cube.shape[0])]
                    
                    psflib_filenames = np.append(ref_filenames, rollnames, axis=0)
                    
                    # All frames aligned to image center (Which are the same size)
                    ref_center = np.array(ref_cube[0].shape)/2

                    # make the PSF library
                    # we need to compute the correlation matrix of all images vs each other since we haven't computed it before
                    psflib = rdi.PSFLibrary(psflib_imgs, ref_center, psflib_filenames, compute_correlation=True)

                    if num_datasets > 1:
                        datasets.append(dataset)
                        psflibs.append(psflib)
                    else:
                        datasets = dataset
                        psflibs = psflib
                        
                return datasets, psflibs
            
            
            return _process_adi(datadir, roll_filenames_list, ref_filenames_list, rollnames_list, pas_list, num_datasets) if mode == 'ADI' else _process_rdi(datadir, roll_filenames_list, ref_filenames_list, rollnames_list, pas_list, num_datasets) if mode == 'RDI' else None


        # Specifying KLIP params. Change as desired
        outputdir = "./"
        fileprefix = f"pyklip-{filtername}-{mode}-k50a9s4m1"
        annuli = annuli
        subsections = subsections
        minrot = minrot

        # Generate dataset for use
        dataset, psflib = generate_datasets(datadir, roll_filenames_list = roll_filenames_list, ref_filenames_list = ref_filenames_list, rollnames_list = rollnames_list, pas_list = pas_list, mode = mode)


        if mode == 'RDI':
            psflib.prepare_library(dataset)
            
        # Run pyKLIP RDI
        parallelized.klip_dataset(dataset, outputdir=outputdir, fileprefix=fileprefix, annuli=annuli, 
                            subsections=subsections, numbasis=numbasis, minrot = minrot, 
                                mode=mode, psf_library=psflib)

        # Read in the KLIP-ed dataset
        filesuffix = "-KLmodes-all.fits"

        with fits.open(f"{fileprefix}{filesuffix}") as hdulist:
            reduced_cube = hdulist[0].data
            reduced_centers = [hdulist[0].header["PSFCENTX"], hdulist[0].header["PSFCENTY"]]

        # Read in the KLIP-ed dataset
        filesuffix = "-KLmodes-all.fits"

        with fits.open(f"{fileprefix}{filesuffix}") as hdulist:
            reduced_cube = hdulist[0].data
            reduced_centers = [hdulist[0].header["PSFCENTX"], hdulist[0].header["PSFCENTY"]]

        def masking(mask = "off", x_positions = None, y_positions = None, psf_fwhm = 6):
            """
            This function masks any real planets in your data given their x and y positions.
            
            Args:
                mask (str): Keyword to specify whether or not you have any planets you'd like to mask
                x_positions (int): x positions of planets to be masked
                y_positions (int): x positions of planets to be masked (must be in same order as x)
            """
            if mask == 'on':
                
                
            # Plot the KL10 Cube (index of 2)
                fig = plt.figure(figsize=(10, 3))
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.imshow(reduced_cube[i], interpolation="nearest", cmap="inferno", vmin = np.nanpercentile(reduced_cube[i], 1), vmax  = np.nanpercentile(reduced_cube[i], 99))
                
                # Place green circles around the real planets
                for j in range(len(x_positions)):
                    circle = plt.Circle((x_positions[j], y_positions[j]), 4, fill=False, edgecolor="green", ls="-", linewidth=3)
                    ax1.add_artist(circle)
                plt.gca().invert_yaxis()
                ax1.set_xlabel("pixels")
                ax1.set_ylabel("pixels")
                ax1.set_title("PSF Subtracted Image")
                
                # Create an array with the indices are that of KL mode frame with index 2
                ydat, xdat = np.indices(reduced_cube[i].shape)
                
                # Mask the first planet
                for x, y in zip(x_positions, y_positions):
                    
                    # Create an array with the indices are that of KL mode frame with index 2
                    distance_from_star = np.sqrt((xdat - x) ** 2 + (ydat - y) ** 2)
                    
                    # Mask
                    reduced_cube[i][np.where(distance_from_star <= 2 * psf_fwhm)] = np.nan
                    
                    post_mask_cube = reduced_cube[i]
                
                # Plot the new masked data
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.imshow(post_mask_cube, interpolation="nearest", cmap="inferno", vmin = np.nanpercentile(reduced_cube[i], 1), vmax  = np.nanpercentile(reduced_cube[i], 99))
                plt.gca().invert_yaxis()
                ax2.set_xlabel("pixels")
                ax2.set_ylabel("pixels")
                ax2.set_title("Real Planets Masked")
                
            elif mask == 'off':
                post_mask_cube = reduced_cube[i]
                
            return post_mask_cube


        psf_fwhm = 6
        masked_cube = masking(mask = mask, x_positions=x_positions, y_positions=y_positions, psf_fwhm = psf_fwhm)

        # Measuring the contrast in the image
        contrast_seps, contrast = pyklip.klip.meas_contrast(dat=masked_cube, iwa=(IWA-3), owa=OWA, resolution=(psf_fwhm), center=reduced_centers, low_pass_filter=False)


        # Read in unocculted PSF
        with fits.open(f"{datadir}/{unocculted_psf}") as hdulist:
            psf_cube = hdulist[0].data
            psf_head = hdulist[0].header

        if len(psf_cube.shape) == 2:
            psf_frame = psf_cube

        elif len(psf_cube.shape) == 3:
            # Collapse reference psf in time
            psf_frame = np.nanmean(psf_cube, axis = 0)

        # Find the centroid
        bestfit = fakes.gaussfit2d(psf_frame, center_x , center_y, searchrad=3, guessfwhm=2, guesspeak=1, refinefit=True)

        psf_xcen, psf_ycen = bestfit[2:4]
        peak_flux = bestfit[0]

        # Recenter PSF to that location
        x, y = np.meshgrid(np.arange(-20, 20.1, 1), np.arange(-20, 20.1, 1))
        x += psf_xcen
        y += psf_ycen

        psf_stamp = scipy.ndimage.map_coordinates(psf_frame, [y, x])


        norm_contrast = contrast / peak_flux



        # Create the throughput correction function
        def transmission_corrected(input_stamp, input_dx, input_dy):

            """
            Args:
                input_stamp (array): 2D array of the region surrounding the fake planet injection site
                input_dx (array): 2D array specifying the x distance of each stamp pixel from the center
                input_dy (array): 2D array specifying the y distance of each stamp pixel from the center
                
            Returns:
                output_stamp (array): 2D array of the throughput corrected planet injection site.
                """

            # Calculate the distance of each pixel in the input stamp from the center
            distance_from_center = np.sqrt((input_dx) ** 2 + (input_dy) ** 2)

            # Interpolate to find the transmission value for each pixel in the input stamp (we need to turn the columns into arrays so np.interp can accept them)
            distance = np.array(coronagraph["rad_dist"])
            transmission = np.array(coronagraph["trans"])
            trans_at_dist = np.interp(distance_from_center, distance, transmission)

            # Reshape the interpolated array to have the same dimensions as the input stamp
            transmission_stamp = trans_at_dist.reshape(input_stamp.shape)

            # Make the throughput correction
            output_stamp = transmission_stamp * input_stamp

            return output_stamp

        def multiple_planet_injection(datadir, filtername, seps, input_pas, num_datasets, input_contrasts, mode):
        
            """
            Injects multiple fake planets across multiple datasets.

            Args:
                datadir (str): The name of the directory that the data is contained in
                filtername (str) The name of the filter to be used
                seps (list: int): List of separations each planet should be injected at
                input_pas (list: int): List of position angles to inject fake planets at 
                num_datastes(int): The number of datasets to be generated. This is equal to the number of interations of planet injection/number of position angle changes
                input_contrasts(list: float): List of contrasts planets should be injected at
            Returns:
                retrieved_fluxes_all (list): All retrieved planet fluxes
                pas_all (list): All position angles used for injection
                planet_seps_all (list): All planet separations used for injection
                input_contrasts_all (list): All planet contrasts used for injection
            """
            
            pas_all = []
            retrieved_fluxes_all = []
            planet_seps_all = []
            input_contrasts_all = []
            
            # Generate desired number of datasets: number of loops at each separation
            datasets, psflibs = generate_datasets(datadir, roll_filenames_list = roll_filenames_list, ref_filenames_list = ref_filenames_list, rollnames_list = rollnames_list, pas_list = pas_list, mode = mode, num_datasets = num_datasets)

            # Begin fake planet injection and retrieval, changing position angle each time
            for dataset_num, dataset, psflib in zip(range(len(datasets)), datasets, psflibs):
                if mode == 'RDI':
                    psflib.prepare_library(dataset)

                # Create stamps of the point spread function to be injected as a fake planet
                psf_stamp_input = np.array([psf_stamp for j in range(12)])
                
                # Clock the position angles of the injected planets by 40 each time
                input_pas = [x+40*dataset_num for x in input_pas]

                start_over = False

                # Inject fake planets
                for input_contrast, sep, pa in zip(input_contrasts, seps, input_pas):

                    # Check the distance between the planet to be injected and the real planets. We don't want to inject fake planets too close to the two planets already in the data.
                    if x_positions is not None:
                        check_sep_x = sep * np.cos((pa + 90))
                        check_sep_y = sep * np.sin((pa + 90))
                        dist_p1 = np.sqrt((check_sep_x - x_positions[0])**2 + (check_sep_y - y_positions[0])**2)
                        dist_p2 = np.sqrt((check_sep_x - x_positions[1])**2 + (check_sep_y - y_positions[1])**2)

                        # Make sure fake planets won't be injected within a 12 pixel radius of the real planets
                        if dist_p1 > 12 and dist_p2 > 12:

                            planet_fluxes = psf_stamp_input * input_contrast
                            fakes.inject_planet(frames=dataset.input, centers=dataset.centers, inputflux=planet_fluxes, astr_hdrs=dataset.wcs, radius=sep, pa=pa, field_dependent_correction=transmission_corrected)

                        # If the fake planet to be injected is within a 12 pixel radius of the real planets, start the loop over
                        else:
                            start_over = True
                        
                    elif x_positions is None:
                        planet_fluxes = psf_stamp_input * input_contrast
                        fakes.inject_planet(frames=dataset.input, centers=dataset.centers, inputflux=planet_fluxes, astr_hdrs=dataset.wcs, radius=sep, pa=pa, field_dependent_correction=transmission_corrected)


                    if start_over:
                        continue
                    

                # Run KLIP on datasets with injected planets: Set output directory
                outputdir = "notebooks/contrastcurves"
                fileprefix = f"FAKE_KLIP_{mode}_A9K5S4M1_{str(dataset_num)}{str(n_sep_loops)}"
                filename =  f"FAKE_KLIP_{mode}_A9K5S4M1_{str(dataset_num)}{str(n_sep_loops)}-KLmodes-all.fits"
                

                # Run KLIP 
                parallelized.klip_dataset(dataset, outputdir=outputdir, fileprefix=fileprefix, algo="klip", annuli=annuli, subsections=subsections, minrot=minrot, numbasis=numbasis, mode=mode, verbose=False, psf_library=psflib)

                # Open one frame of the KLIP-ed dataset
                klipdataset = os.path.join(outputdir, filename)
                with fits.open(klipdataset) as hdulist:
                    outputfile = hdulist[0].data
                    outputfile_centers = [hdulist[0].header["PSFCENTX"], hdulist[0].header["PSFCENTY"]]
                outputfile_frame = outputfile[2]

                # Retrieve planet fluxes
                retrieved_planet_fluxes = []
                for input_contrast, sep, pa in zip(input_contrasts, seps, input_pas):

                    fake_flux = fakes.retrieve_planet_flux(frames=outputfile_frame, centers=outputfile_centers, astr_hdrs=dataset.output_wcs[0], sep=sep, pa=pa, searchrad=7)
                    retrieved_planet_fluxes.append(fake_flux)
                retrieved_fluxes_all.extend(retrieved_planet_fluxes)
                pas_all.extend(input_pas)
                planet_seps_all.extend(seps)
                input_contrasts_all.extend(input_contrasts)
                
            return retrieved_fluxes_all, pas_all, planet_seps_all, input_contrasts_all
            
        if not sys.warnoptions:
            import warnings
            warnings.simplefilter("ignore")
            
        # Define separation variables 
        min_sep = IWA
        max_sep = OWA
        nplanets = 3
        dist_bt_planets = 3
        num_datasets = 2 #Change to larger number 
        input_pas = [0, 30, 60]

        # Maximum separation of first iteration
        max_sep_1 = min_sep + (dist_bt_planets * (nplanets-1))

        # Number of times to iterate to get to max desired separation (max desired sep - max sep in first iteration)
        # Add 1 because loop will start at 0
        n_sep_loops = int((((max_sep - min_sep)/(dist_bt_planets)) + 1)/nplanets)

        retrieved_fluxes_all = []
        output_pas_all = []
        planet_seps_all = []
        output_contrasts_all = []

        for n in tqdm(range(n_sep_loops)):
            # Create array of separations and contrasts to be injected at, spaced by desired distance b/t planets
            seps = np.arange(min_sep + (9*n), max_sep_1+1 + (9*n), dist_bt_planets)
            input_contrasts = (np.interp(seps, contrast_seps, norm_contrast))*5
            
            retrieved_fluxes, output_pas, output_planet_seps, output_contrasts = multiple_planet_injection(datadir, filtername, seps, input_pas, num_datasets, input_contrasts, mode)
            
            retrieved_fluxes_all.extend(retrieved_fluxes)
            output_pas_all.extend(output_pas)
            planet_seps_all.extend(output_planet_seps)
            output_contrasts_all.extend(output_contrasts)

            # Create a table of all variables
            flux_sep = Table([retrieved_fluxes_all, planet_seps_all, output_contrasts_all, output_pas_all], names=("flux", "separation", "input_contrast", "pas"))
            flux_sep["input_flux"] = flux_sep["input_contrast"] * bestfit[0]

        # Calculate throughput and add it to the table
        flux_sep["throughput"] = flux_sep["flux"] / flux_sep["input_flux"]

        # Group by separation
        med_flux_sep = flux_sep.group_by("separation")

        # Calculate the median value for each separation group
        med_flux_sep = med_flux_sep.groups.aggregate(np.nanmedian)

        # Find the 5 sigma contrast for each separation used in calculation
        med_flux_sep['5sig_contrast']=np.interp(med_flux_sep['separation'],contrast_seps, norm_contrast)

        # Normalize the noise contrast by the measured throughput level at that separation
        med_flux_sep["corrected_contrast"] = (med_flux_sep["5sig_contrast"] / med_flux_sep["throughput"])

        # Find slope and intercept of best fit line
        m, b = np.polyfit(med_flux_sep['separation'],med_flux_sep['throughput'], 1)

        # Calibrate contrast curve w/ best fit line
        y = m*med_flux_sep['separation']+b
        raw_contrast = np.interp(med_flux_sep['separation'],contrast_seps, norm_contrast)
        contrast_bestfit = raw_contrast/y
        
        kls = [i]*len(med_flux_sep["corrected_contrast"])
        moves = [minrot]*len(med_flux_sep["corrected_contrast"])

        all_corrected_contrasts.extend(med_flux_sep["corrected_contrast"])
        all_kl.extend(kls)
        all_minrots.extend(moves)
        all_separations.extend(med_flux_sep["separation"])
        all_smooth_contrast.extend(contrast_bestfit)

df = pd.DataFrame(list(zip(all_corrected_contrasts, all_smooth_contrast, all_kl, all_minrots, all_separations)), columns = ['med_contrast', 'smooth_contrast', 'kl', 'minrot', 'separation'])
df.to_csv('contrast_meas.csv', index = False)
