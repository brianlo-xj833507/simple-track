import object_tracking
import numpy as np
import datetime
import os
import user_functions

if __name__ == '__main__':
    ##################################################################
    # THE FOLLOWING PARAMETERS SHOULD BE CHANGED BASED ON THE DATA (RESOLUTION ETC.)
    ##################################################################

    # Integer number (dimensions user-defined) to identify MINIMUM time difference between consecutive data files
    # Example 1: Radar data 5-minutes apart with time stamp in filename, dt = 5
    # Example 2: Satellite brightness temperatures hourly with time stamp in filename, dt = 1
    # NB. When writing storms, (dx,dy) will have units PIXELS per TIME STEP (specified by dt),
    # so already scaled by number of missing files

    # TODO: Check dt time units
    dt = 5.

    # Maximum separation in time allowed between consecutive images
    dt_tolerance = 15.

    # under_t: Is the variable of interest smaller than a threshold
    # True = labelling areas *under* the threshold (e.g. brightness temperature),
    # False = labelling areas *above* threshold (e.g. rainfall)
    under_t = False

    # threshold: Threshold used to identify objects
    # with value of variable greater than this threshold
    threshold = 3.

    # minpixel: The minimum number of pixels for an object to be tracked
    minpixel = 4.

    # squarelength: The size in pixels of individual square regions for which fft will calculate displacement vectors
    # (should be large enough to cover several mid-sized objects)
    squarelength = 100.  # Must divide (x,y) lengths of the array!

    # rafraction: The minimum fractional cover of objects required in order for fft
    # to calculate displacement vectors (dx, dy)
    rafraction = 0.01

    # dd_tolerance: The maximum difference (in number of pixels) allowed between adjacent displacement vectors
    dd_tolerance = 3.

    # halopixel: Radius of halo in pixels for orphan storms
    # big halo assumes storms may spawn "children" at a distance multiple pixels away
    halopixel = 5.

    # flagwrite: For writing storm history data in a text file
    # If False, then no text files with object information is included in the output. [Default should be True]
    flagwrite = True

    # doradar: For calculating radar range and azimuth if real-time tracking with a single site radar
    # If True, then calculate range and azimuth for real-time tracking with radar (e.g. Chilbolton).
    # False for any other use, radar coordinates not relevant [Default should be False]
    doradar = False

    # misval: Preferred value to used for missing values.
    misval = -999

    # struct2d: Defines the neighbour-searching function, can be changed to
    # e.g. np.ones((3,3)) is 8-point connectivity, chessboard metric
    # e.g. np.array([[0,1,0],[1,1,1],[0,1,0]]) is 4-point connectivity, manhattan metric
    struct2d = np.ones((3, 3))

    # flagplot: For plotting images (vectors and IDs). Also set plot_type...
    # If True, a few images are included in the output (plotting function defined in "user_functions.py"
    # [Trials should set this to True, long runs could set it to False to save time]
    flagplot = True

    # flagplottest: For plotting fft correlations (testing only, very slow, lots of plots)
    # If True, numerous test images are included to check the displacement vector calculations [Default should be False]
    flagplottest = False

    if flagplot or flagplottest:
        plot_type = '.png'
        if plot_type == '.eps':
            my_dpi_global = 150  # for rasterized plots, this controls the size and quality of the final plot
        elif plot_type == '.png':
            my_dpi_global = 300

    ##################################################################
    # THE FOLLOWING PARAMETERS CAN BE CHANGED, BUT SHOULD NOT BE
    ##################################################################

    # lapthresh: Minimum overlap fraction required for objects to be considered potentially
    # the same between consecutive images [Default is 0.6]
    lapthresh = 0.6

    ##################################################################
    # AUTOMATIC SET UP OF DISCRETE VALUES BASED ON USER INPUT PARAMETERS
    # ESSENTIAL - NO NEED TO CHANGE THESE
    ##################################################################
    # squarehalf: To determine grid spacing for coarse grid of (dx,dy) estimates
    # areastr: For filename identifier of area threshold used
    # thr_str: For filename identifier of variable threshold used
    # fftpixels: Minimum number of thresholded pixels needed to calculate (dx,dy)
    # halosq: To identify if new cell is nearby existing cell
    ##################################################################

    squarehalf = int(squarelength / 2)
    areastr = str(int(minpixel))
    thr_str = str(int(threshold))
    sql_str = str(int(squarelength))
    fftpixels = squarelength ** 2 / int(1. / rafraction)
    halosq = halopixel ** 2

    ##################################################################
    # AUTOMATIC SET UP OF TEXT STRING FOR INFORMATION ON LABELLING
    # NOT ESSENTIAL
    ##################################################################

    label_method = 'Rainfall rate > ' + thr_str + 'mm/hr'

    ##################################################################
    # THE REMAINDER IS THE SET UP FOR THE EXAMPLE DATA
    # THIS SHOULD BE ADJUSTED RELEVANT TO THE USER DATA
    # !!!TEST: MAKE SURE xall AND yall DIVIDE IN squarelength!!!
    ##################################################################

    # TODO: Put this grid length check somewhere else! It is now hardcoded to accept grids of 400 * 300.
    xmat, ymat = np.meshgrid(range(-200, 200), range(-150, 150))
    xall = np.size(xmat, 0)  # Only used to check grid dimensions
    yall = np.size(xmat, 1)  # Only used to check grid dimensions
    if np.fmod(xall, squarelength) != 0 or np.fmod(yall, squarelength) != 0:
        raise ValueError('Your grid does not match a multiple of squares as defined by squarelength')

    #################################################################
    # For test data, try the following:
    #
    # All data (5-minute intervals)
    # DATA_DIR = './data/'
    # IMAGES_DIR = './output/'
    # Sparse data (10-minute intervals), to test similarity in vector fields (scaling by num_dt working correctly)
    # DATA_DIR = './data/'
    # IMAGES_DIR = './output/'
    # Missing data (10-minute intervals, 1 file missing), to test dt_tolerance
    # DATA_DIR = './data/'
    # IMAGES_DIR = './output/'
    #################################################################
    # TODO: Change DATA_DIR and IMAGES_DIR!
    DATA_DIR = './data/'
    IMAGES_DIR = './output/'
    filelist = os.listdir(DATA_DIR)
    filelist = np.sort(filelist)
    if doradar:
        rarray = np.sqrt(xmat ** 2 + ymat ** 2)
        azarray = np.rad2deg(np.arctan2(xmat, ymat)) % 360.0
        # azarray = np.arctan(xmat/ymat)
        # # Quadrant 2
        # azarray[np.where((xmat < 0) & (ymat >= 0))] = azarray[np.where((xmat < 0) & (ymat >= 0))] + 2 * np.pi
        # # Quadrant 3 and 4
        # azarray[np.where(ymat < 0)] = azarray[np.where(ymat < 0)] + np.pi
        # azarray = 180 * azarray / np.pi
        azarray[np.where(np.isnan(azarray) == 1)] = 0

    #   Initialise variables
    OldData, OldLabels, oldvar, newvar, prev_time = [], [], [], [], []
    newwas = 1
    plot_vectors = False

    start_time = datetime.datetime(2012, 8, 25, 14, 5, 0, 0)
    oldhourval = []
    oldminval = []
    oldmask = []
    newmask = []
    num_dt = []

    for nt in range(len(filelist)):
        # Load new image
        # TODO: Time interval is currently hardcoded
        now_time = start_time + datetime.timedelta(seconds=300. * nt)
        var, file_ID, hourval, minval = user_functions.loadfile(DATA_DIR + filelist[nt])
        print(file_ID)
        write_file_ID = f"S{sql_str}_T{thr_str}_A{areastr}_{file_ID}"
        NewLabels = object_tracking.label_storms(var, minpixel, threshold, struct2d, under_t)
        # oldmask, newmask, USED FOR DERIVING (dx,dy)
        # THESE CAN BE CHANGED USING EXPERT KNOWLEDGE
        # e.g. use raw data rather than binary masks,
        # if displacement information is contained in structures within objects
        # NB If raw data are used (i.e. not zeros and ones) then fftpixels needs to be changed to remain sensible
        if len(OldLabels) > 1:
            # CHECK TIME DIFFERENCE BETWEEN CONSECUTIVE IMAGES
            dtnow = user_functions.timediff(oldhourval, oldminval, hourval, minval)
            num_dt = dtnow / dt
            if dtnow > dt_tolerance:
                print('Data are too far apart in time --- Re-initialise objects')
                OldData, OldLabels, oldvar, newvar, prev_time = [], [], [], [], []
                newwas = 1
                plot_vectors = False
                continue
            oldmask = np.where(OldLabels >= 1, 1, 0)
            newmask = np.where(NewLabels >= 1, 1, 0)

        # Call object tracking routine
        # NewData: list of objects and properties
        # newwas: final label number
        # NewLabels: array with object IDs from [1, nummax] as found by label_storms
        # newumat, newvmat: arrays with (dx,dy) displacement between two images (NB not displacement per dt!!!)
        # wasarray: array with object IDs consistent across images (i.e. tracked IDs)
        # lifearray: array with object lifetime consistent across images
        NewData, newwas, NewLabels, newumat, newvmat, wasarray, lifearray = object_tracking.track_storms(OldData,
                                                                                                         var,
                                                                                                         newwas,
                                                                                                         NewLabels,
                                                                                                         OldLabels,
                                                                                                         xmat,
                                                                                                         ymat,
                                                                                                         fftpixels,
                                                                                                         dd_tolerance,
                                                                                                         halosq,
                                                                                                         squarehalf,
                                                                                                         oldmask,
                                                                                                         newmask,
                                                                                                         num_dt,
                                                                                                         lapthresh,
                                                                                                         misval,
                                                                                                         doradar,
                                                                                                         under_t,
                                                                                                         IMAGES_DIR,
                                                                                                         write_file_ID,
                                                                                                         flagplottest)
        # Write tracked storm information
        if flagwrite:
            object_tracking.write_storms(write_file_ID, start_time, now_time, label_method, squarelength, rafraction,
                                         newwas, NewData, doradar, misval, IMAGES_DIR)

        # Plot tracked storm information (see user_functions.plot_example)
        if flagplot:
            user_functions.plot_example(write_file_ID, nt, var, xmat, ymat, newumat, newvmat, num_dt, wasarray,
                                        lifearray, threshold, IMAGES_DIR, plot_vectors)

        # Save tracking information in preparation for next image
        OldData = NewData
        OldLabels = NewLabels
        oldvar = var
        oldhourval = hourval
        oldminval = minval
        plot_vectors = True
