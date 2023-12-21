# Raster Constants
BAND_FIRST = 1
NODATA = -32767
INIT_DEFAULT = -1
FREQUENCY_FLAGGED = 1


class CrossSectionKey:
    VT = "vt"
    HZ = "hz"
    ALL = "all"


class TestDemFile:

    """Default file paths for testing input data."""

    NOISY = r"C:\Users\LETHIER\work\waveform\data\dem\known_data\bc_092g025_1m_clipped.tif"
    CLEAN_1m = r"C:\Users\LETHIER\work\waveform\data\dem\known_data\bc_092g025_1m_clipped_clean.tif"
    SRC_NOISY = r"C:\Users\LETHIER\work\waveform\data\dem\source\bc_092g025_xl1m_utm10_170713.tif"
    CLEAN_15cm = r"C:\Users\LETHIER\work\waveform\data\dem\known_data\bc_092g025_15cm_clipped_clean.tif"


class WaveletCoeffThreshold:

    """Constants for waveform coefficient thresholds."""

    coeff_0P50 = 0.50
    coeff_1p00 = 1.00
    coeff_2p00 = 2.00


class WaveletType:

    """Constants for Wavelet type names."""

    DB1 = "db1"
    DB2 = "db2"
    DB4 = "db4"
    MORL = "morl"
    HAAR = "haar"


class DecompositionLevel:

    """Constants for waveform decomposition levels."""

    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5
    L10 = 10
    L12 = 12



