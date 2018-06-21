
from PV_analysis.util import util
import scipy.constants as const


def Trupke2007(li_imag, Ut, ce_imag, Ap, lisc_imag=0, cesc_imag=0, temp=300):
    '''
    Takes two input images (corrected by the short circuit images) photoluminescence images.
    The first image minus the short circuit image is used to calibrate the
    optical constant. This assums:

    1. The terminal voltage is equal to the local voltage (U~t~ = U~i~).
      i.e. Series resistance has no impact
    2. The calibration C~i~ is not injeciton dependent. This is only true
    if the sample is in low injeciton. This calibration makes the method robust
    against J~0~ variation.
    3. The short circuit image represents the voltage independent carriers
    of the open circuit image.

    With this calibration, any image can be converted into a voltage,
    and hence voltage drop map. The second image, is then used to determine
    a Delta V map

    The voltage drop is then caculated as,
        $\Delta U = U_t - V~t~ \ln\left(\frac{I_{PL}}{C_i} \right)$

    The lateral current determined from the same image:
        $J_i = J_{sc} - J_0 \frac{I_{PL}}{C_i}$

    Where J0 is taken as the ratio of Jsc to exp(Uex/Vt). This estimation of the
    current assumes that the change in the PL signal goes to the fingers. This
    does not account for changes in lateral currents.

    Finally:
        $ R_{s,i} = \frac{\Delta U_{R_s,i} A_i}{J_i}$

    Inputs:
        il_imag: (ndarray, representing counts per unit time)
            The first a PL image at low intenisty (~0.1 sun).
        Uil: (float)
            Terminal voltage measuirement of il_imag
        ec_imag: (ndarray, representing counts per unit time)
            The second PL image is at high intensisty with current extraction.
        Uec: (float)
            Terminal voltage measuirement of the ec image
        Ap: (float)
            The area of a pixel.
        ilsc_imag: (ndarray, representing counts per unit time, optional)
            The short circuit image of the il_imag
        ecsc_imag: (ndarray, representing counts per unit time, optional)
            The short circuit image of the ec_imag

    Output:
        Rs: (nd array)
            A series resistance image


    doi: 10.1002/pssr.200903175

    Example that it can be usd for for inhomgenious Jo. (DOI: 10.1002/pssr.200903175)
    '''

    # get the calibration values
    img_cal = li_imag - lisc_imag
    Ci = util.get_Ci_negliableRs(img_cal, Uil)

    # get the voltage drop image
    img_rs = ce_imag - cesc_imag
    DU = Uec - util.voltage_map(img_rs, Ci, temp=temp)

    # J0 is estimated to be constant
    J0 = Jsc / (np.exp(Uec / const.k * temp / const.e) - 1.)
    Ji = Jsc - J0 * img_rs / Ci

    # finally caculate Rs
    Rs = DU * Ap / Ji

    return Rs


def Hinken2007(images):
    '''
    Notes:
        Only uses EL

    As yet not implimented

    Example that it can not be usd for for inhomgenious Jo. (DOI: 10.1002/pssr.200903175)

    DOI: 10.1063/1.2804562
    '''
    pass


def Kampwerth2008(images):
    '''
    This follows on from the work of Trupke2007. It remoes of the need to
    conversion of luminescence intensities into absolute voltages, and
    make assumptions about J~dark~(U~i~).

    The publication used 6 images.

    The idea is to take one PL image. A second PL image is taken at a different
    illumination intenisty, and a bias that removes the impact of the change in
    generation. The pixels that have the same PL intensity in both images can
    then be analyised with:

    $R~s,i~ = \frac{\Delta U_{term}}{\Delta J_{sc}}$

    Where *U~term~* is the different between the two images terminal voltage,
    and *J~sc~* is the diffrence between the light generated current. Of course
    only a few pixels will be of the same intenisty, so by takeing several images
    at the second illumination intenisty at different electrical bias and performing
    a fit, the intenisty at which each pixel is the same as the original image
    can be found.

    Note: All images require correction to the voltage independent carriers.

    input:
        images: (list of image classes)
            list of image classes in the following index
                0: A Open circuit PL image at an illumination intenisty of A
                1: A Short circuit PL image at an illumination intensity of A
                2: A PL image biased at V1 and with a illumination intenisty of B
                3: A PL image biased at V2 and with a illumination intenisty of B
                4: A PL image biased at V3 and with a illumination intenisty of B
                5: A Short circuit PL image with a illumination intensity of B

    output:
        Rs image

    DOI: 10.1063/1.2982588
    '''

    # normalise to exposure
    image_data = []
    Vt = []
    Jt = []
    for imag in images:
        image_data.append(image.image / image.exposure)
        Ut.append(image.Ut)
        Jt.append(image.Jt)

    # correct the images to the voltage independent carriers
    image_data[0] = image_data[0] - image_data[1]

    image_data[2] = image_data[2] - image_data[5]
    image_data[3] = image_data[3] - image_data[5]
    image_data[4] = image_data[4] - image_data[5]

    # Returns quadratic then  linear then constant coef
    Fit = np.polyfit(Vt[2:4], image_data[2:4], 2
                     ).reshape((2,
                                image_data.shape[0],
                                image_data.shape[1]))

    # the voltage at which the pixel are the same brightness is then
    Ut_2 = (-FitVals[1] + np.sqrt(
        FitVals[1]**2 - 4 * FitVals[0] *
        (FitVals[2] - image_data[0])) / (2 * FitVals[0]))

    DU = Vt[0] - Ut_2

    Rs = -Du / (Jt[1] - Jt[5])

    return Rs


def Haunschild2009(images):
    '''
    Only uses EL
    DOI: 10.1002/pssr.200903175
    '''
    pass


def Glatthaar2010(images):
    '''
    DOI: 10.1002/pssr.200903290
    '''
    pass
