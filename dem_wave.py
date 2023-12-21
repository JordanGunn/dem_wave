# system imports
import os
import pywt
import tqdm
import rasterio
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
from scipy.signal import butter, filtfilt

# user imports
from const import *


class _PlotData:

    def __init__(self, index: int, coeffs: np.ndarray, data: np.ndarray, levels: int, direction: str):
        self.index = index
        self.levels = levels
        self.coeffs = coeffs
        self.data = data
        self.direction = direction


class _CrossSection:

    KEY = ""
    SAMPLING_RATE_DEFAULT = 1000
    CUTOFF_FREQUENCY_DEFAULT = 0.50

    def __init__(self, index: int, start: int, length: int = 0):

        """
        Initialize a CrossSection object

        :param start: Starting row-index in a 2D array.
        :param length: Number of pixels to take cross section along.
        """
        self._data = None
        self._index = index
        self._start = start
        self._offset = 0
        self._nodata_value = NODATA
        self._end = (self._start + length) if length else length
        self._sampling_rate = self.SAMPLING_RATE_DEFAULT
        self._cutoff_frequency = self.CUTOFF_FREQUENCY_DEFAULT
        self._nyquist_sampling_rate = 0.50 * self._sampling_rate
        self._normal_cutoff = self._cutoff_frequency / self._nyquist_sampling_rate

    @property
    def sampling_rate(self) -> int:

        """Get the sampling rate property."""

        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate: int):

        """Set the sampling rate property."""

        while not self._is_valid_sample_rate(sampling_rate):
            sampling_rate *= 10

        self._nyquist_sampling_rate = 0.5 * self._sampling_rate
        self._normal_cutoff = self._cutoff_frequency / self.nyquist_sampling_rate

    @property
    def index(self) -> int:

        """Get the sampling rate property."""

        return self._index

    @property
    def offset(self) -> int:

        """Get the sampling rate property."""

        return self._offset

    @property
    def nyquist_sampling_rate(self) -> float:

        """Get the sampling rate property."""

        return self._nyquist_sampling_rate

    @property
    def cutoff_frequency(self):
        """Get the cutoff frequency property."""

        return self._cutoff_frequency

    @cutoff_frequency.setter
    def cutoff_frequency(self, cutoff_frequency: float):

        if cutoff_frequency <= 0.0:
            raise ValueError("Cutoff frequency must be a positive float value")
        else:
            self._cutoff_frequency = cutoff_frequency
            self._normal_cutoff = self._cutoff_frequency / self._nyquist_sampling_rate

    @property
    def normal_cutoff(self) -> float:
        """Get the normal_cutoff property."""

        return self._normal_cutoff

    def _is_valid_sample_rate(self, sampling_rate) -> bool:
        """Check if sampling rate meets minimum constraints."""

        return sampling_rate <= ((self._end - self._start) * 2)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        """Abstract setter."""
        pass

    @property
    def nodata_value(self):
        return self._nodata_value

    @nodata_value.setter
    def nodata_value(self, nodata_value):
        self._nodata_value = nodata_value

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end


class CrossSectionVt(_CrossSection):

    KEY = "vt"

    def __init__(self, index: int, start: int, length: int = 0):
        super().__init__(index, start, length)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        """Set the data property."""

        data = data[self.start:self._end, self._index]
        data_half = data[:int(data.size / 2)]
        if data_half[0] == self.nodata_value:
            data_half_nodata = data_half[data_half != self._nodata_value]
            self._offset = data_half_nodata.size - data_half.size

        self._data = data[data != self._nodata_value]


class CrossSectionHz(_CrossSection):

    KEY = "hz"

    def __init__(self, index: int, start: int, length: int = 0):
        super().__init__(index, start, length)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        """Set the data property."""

        data = data[self._index, self.start:self.end]
        data_half = data[:int(data.size / 2)]
        if data_half[0] == self.nodata_value:
            data_half_nodata = data_half[data_half != self._nodata_value]
            self._offset = data_half_nodata.size - data_half.size
        self._data = data[data != self._nodata_value]


class _HighFrequencyCell:

    def __init__(self, row: int, col: int, value: float):
        self._row = row
        self._col = col
        self._value = value

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

    @property
    def value(self):
        return self._value


class DemWave:

    THRESHOLD_COEFF_L4 = 0.35
    MAX_ENERGY_L4 = 0.50

    def __init__(self, file: str, wavelet_type: str, decomp_levels: int):
        """

        :param file:
        :param wavelet_type:
        :param decomp_levels:
        """

        self.rows = []
        self.file = file
        self.length = -1
        self.plot_data = []
        self.cross_sections = {"hz": [], "vt": []}
        self.high_frequencies = []
        self.wavelet_type = wavelet_type
        self.decomp_levels = decomp_levels
        self._max_energy = self.MAX_ENERGY_L4
        self._threshold_coeff = self.THRESHOLD_COEFF_L4

        try:
            with rasterio.open(file) as dem:
                self.data = dem.read(BAND_FIRST)
                self.width = dem.width
                self.height = dem.height
                self.bounds = dem.bounds
                self._transform = dem.transform
                self.extent = [
                    self._transform[2], self._transform[2] + dem.width * self._transform[0],  # (min x, max x)
                    self._transform[5] + dem.height * self._transform[4], self._transform[5]  # (min y, max y)
                ]
        except rasterio.errors.RasterioIOError:
            print(f"\nERROR:\n\tFile {file} does not exist ...")

    def write_result(self, name: str = "", outdir: str = ""):

        # Open the source dataset
        src_ds = gdal.Open(self.file, gdal.GA_ReadOnly)

        # Create the destination dataset
        if not name:
            name = os.path.basename(self.file)
            name = name.split(".")[0] + "_waveform.tif"

        name = os.path.join(outdir, name)
        driver = gdal.GetDriverByName("GTiff")
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        num_bands = src_ds.RasterCount
        data_type = src_ds.GetRasterBand(1).DataType
        dst_ds = driver.Create(name, width, height, num_bands, data_type)

        gdal_array.CopyDatasetInfo(src_ds, dst_ds)
        # Copy the data from the source to the destination
        band = dst_ds.GetRasterBand(BAND_FIRST)
        data = band.ReadAsArray()
        data = (data * 0) + NODATA  # set everything to nodata value
        freqs = tqdm.tqdm(self.high_frequencies, desc="Writing high frequency pixels")
        for freq in freqs:
            data[freq.row, freq.col] = freq.value

        print(f"\n  Writing results to {name} ...")
        band.WriteArray(data)
        gdal.SieveFilter(band, None, band, 2)

        src_ds = None   # gdal requires this to free up memory and flush changes
        dst_dst = None  # gdal requires this to free up memory and flush changes

    def add_cross_section(self, cross_section: _CrossSection):

        if isinstance(cross_section, CrossSectionVt):
            self.cross_sections["vt"].append(cross_section)
        elif isinstance(cross_section, CrossSectionHz):
            self.cross_sections["hz"].append(cross_section)
        else:
            raise ValueError("Argument cross_section must derived from parent class CrossSection")

    def get_cross_sections(self, index: int = 0, width: int = -1, direction: str = "hz") -> Union[list, None]:

        sections = None
        if self.cross_sections:
            if width == -1:
                sections = self.cross_sections[direction][index:]
            else:
                sections = self.cross_sections[direction][index:(index + width)]

        return sections

    def set_cross_sections(self, length: int = -1, direction: str = "hz"):

        is_hz = (direction == "hz")
        along, entire = (self.height, self.width) if is_hz else (self.width, self.height)
        length = entire if length <= 0 else length
        section_obj = CrossSectionHz if is_hz else CrossSectionVt

        start = 0
        for index in range(along):
            cs = section_obj(index=index, start=start, length=length)
            cs.data = self.data
            self.add_cross_section(cs)

    def find_high_frequency(self, cross_sections: list[_CrossSection]):

        if not cross_sections:
            raise ValueError("ERROR: 'self.cross_sections' not set.")

        ignored = []

        sections = tqdm.tqdm(cross_sections, desc="Checking for high frequency areas in cross sections")
        for cross_section in sections:
            wavelet = self.wavelet_type
            level = self.decomp_levels

            try:
                data = self.high_pass_filter(cross_section)
                if data.size == 0:  # check if there is no data in the cross section
                    continue
            except ValueError:
                ignored.append(cross_section)
                continue

            # Perform wavelet packet decomposition
            try:
                coeffs = pywt.wavedec(data, wavelet, level=level, mode='symmetric')
            except UserWarning:
                continue

            max_coeffs_last = np.abs(max(coeffs[-1]))
            threshold = np.abs(max(coeffs[-1])) * self._threshold_coeff
            self._store_plot_data(coeffs, cross_section, data, level)
            if max_coeffs_last >= self._max_energy:
                high_freq_indices = np.where(np.abs(coeffs[-1]) > threshold)[0]
                time_points_high_freq = high_freq_indices * (len(data) / len(coeffs[-1]))

                if time_points_high_freq.size > 0:
                    high_freqs = []
                    for p in time_points_high_freq.tolist():
                        p = int(p)
                        ind = cross_section.index
                        args = (ind - cross_section.offset, p) \
                            if cross_section.KEY == "hz" else (p - cross_section.offset, ind)
                        cell = _HighFrequencyCell(*args, FREQUENCY_FLAGGED)
                        high_freqs.append(cell)

                    self.high_frequencies.extend(high_freqs)

    def _store_plot_data(self, coeffs, cross_section: _CrossSection, data, level):
        """

        :param coeffs:
        :param cross_section:
        :param data:
        :param level:
        :return:
        """

        index = cross_section.index

        plot_data = _PlotData(index, coeffs, data, level, cross_section.KEY)
        self.plot_data.append(plot_data)

    @staticmethod
    def write_plot(plot_data: _PlotData, name="", outdir=""):

        index = plot_data.index
        coeffs = plot_data.coeffs
        data = plot_data.data
        level = plot_data.levels
        outdir = os.path.join(outdir, plot_data.direction)

        # Plot the original signal
        plt.figure(figsize=(10, 6))
        plt.subplot(level + 2, 1, 1)
        plt.plot(data, label='Original Signal')
        plt.title('Original Signal')
        plt.legend()

        # Plot the wavelet packet decomposition coefficients
        level_plots = []
        for j, coeff in enumerate(coeffs):
            plt.subplot(level + 2, 1, j + 2)
            plt.plot(coeff, label=f'Level {j} Coefficients')
            plt.title(f'Level {j} Coefficients')
            plt.legend()
            level_plots.append(plt)

        along = "row" if plot_data.direction == "hz" else "col"

        os.makedirs(outdir, exist_ok=True)
        name = name + f"_{along}{index}.png"
        out = os.path.join(outdir, name)

        plt.savefig(out)
        plt.close()

    @staticmethod
    def high_pass_filter(cross_section: _CrossSection, order=4) -> np.ndarray:
        """

        :param cross_section:
        :param order:
        :return:
        """

        data = cross_section.data
        normal_cutoff = cross_section.normal_cutoff

        params = butter(order, normal_cutoff, btype='high', analog=False)
        b, a = params[0], params[1]
        filtered_signal = filtfilt(b, a, data)

        return filtered_signal


def main():

    # create a DemWave object (intialize with {filepath, "db1", "4"})
    dem_wave = DemWave(TestDemFile.SRC_NOISY, WaveletType.DB2, decomp_levels=4)

    # compute all the cross-sections horizontally
    dem_wave.set_cross_sections(direction=CrossSectionKey.HZ)
    dem_wave.set_cross_sections(direction=CrossSectionKey.VT)

    # extract the first 100 cross-sections from all the cross-sections
    hz_sections = dem_wave.get_cross_sections(direction=CrossSectionKey.HZ)
    vt_sections = dem_wave.get_cross_sections(direction=CrossSectionKey.VT)

    # dem_wave.find_high_frequency(hz_sections)
    dem_wave.find_high_frequency(vt_sections)

    outdir = r"C:\Users\LETHIER\work\waveform\data\dem\out"

    dem_wave.write_result(outdir=outdir)

    # plot_out = r"C:\Users\LETHIER\work\waveform\data\cross_sections"
    # for plot in dem_wave.plot_data:
    #     dem_wave.write_plot(plot, outdir=plot_out)


if __name__ == "__main__":
    main()
