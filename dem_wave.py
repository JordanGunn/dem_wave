# system imports
import os
import pywt
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# user imports
from const import *


class _PlotData:

    def __init__(self, row, col, coeffs, data, levels):
        self.row = row
        self.col = col
        self.levels = levels
        self.coeffs = coeffs
        self.data = data

class _CrossSection:

    SAMPLING_RATE_DEFAULT = 1000
    CUTOFF_FREQUENCY_DEFAULT = 0.50

    def __init__(self, row_start: int, col_start: int, length: int, vertical: bool = False):

        """
        Initialize a CrossSection object

        :param row_start: Starting row-index in a 2D array.
        :param col_start: Starting col-index in a 2D array.
        :param length: Number of pixels to take cross section along.
        :param vertical: Boolean value - controls whether to take x-section vertically (N-S) or Horizontally (E-W)
        """
        self._data = None
        self.vertical = vertical
        self._nodata_value = NODATA
        self._col_start = col_start
        self._row_start = row_start
        self._sampling_rate = self.SAMPLING_RATE_DEFAULT
        self._cutoff_frequency = self.CUTOFF_FREQUENCY_DEFAULT
        self._nyquist_sampling_rate = 0.50 * self._sampling_rate
        self._normal_cutoff = self._cutoff_frequency / self._nyquist_sampling_rate

        self._col_end = (self._col_start + length) if not vertical else self._col_start
        self._row_end = (self._row_start + length) if vertical else self._row_start

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

        return sampling_rate <= ((self._col_end - self._col_start) * 2)

    def change_axis(self):
        self.vertical = not self.vertical

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        """Set the data property."""

        if self.vertical:
            self._data = data[self._row_start:self._row_end, self._col_start]
        else:
            self._data = data[self._row_start, self._col_start: self._col_end]
        self._data = self._data[self._data != self._nodata_value]

    @property
    def nodata_value(self):
        return self._nodata_value

    @nodata_value.setter
    def nodata_value(self, nodata_value):
        self._nodata_value = nodata_value

    @property
    def row_start(self):
        return self._row_start

    @property
    def col_start(self):
        return self._col_start


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
        self.cross_sections = []
        self.high_frequencies = []
        self.wavelet_type = wavelet_type
        self.decomp_levels = decomp_levels

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

        # copy the dataset information
        # Open the source dataset
        with rasterio.open(self.file, 'r') as src_ds:
            # Create a new dataset using the profile of the source dataset
            profile = src_ds.profile

            # You can modify or add additional metadata here if needed
            # For example:
            # profile['transform'] = ...

            # Create the destination dataset
            if not name:
                name = os.path.basename(self.file)
                name = name.split(".")[0] + "_waveform.tif"

            name = os.path.join(outdir, name)
            with rasterio.open(name, 'w', **profile) as dst_ds:
                # Copy the data from the source to the destination
                data = src_ds.read(BAND_FIRST)
                data = (data * 0) + NODATA  # set everything to nodata value
                for freq in self.high_frequencies:
                    data[freq.row, freq.col] = freq.value

                dst_ds.write(data, BAND_FIRST)



    def add_cross_section(self, cross_section: _CrossSection):
        self.cross_sections.append(cross_section)

    def get_cross_sections(self, index, width):

        if not self.cross_sections:
            return None
        else:
            return self.cross_sections[index:(index + width)]

    def set_cross_sections(self, length: int = -1, vertical: bool = False):

        length = self.width if (length < 0) else length

        col_start = 0
        for row in range(self.height):
            cs = _CrossSection(
                row_start=row, col_start=col_start,
                length=length, vertical=vertical
            )
            cs.data = self.data
            self.add_cross_section(cs)

    def find_high_frequency(self, cross_sections: list[_CrossSection] = None):

        cross_sections = self.cross_sections if (not cross_sections) else cross_sections

        if not cross_sections:
            raise ValueError("ERROR: 'self.cross_sections' not set.")

        for cross_section in cross_sections:

            wavelet = self.wavelet_type
            level = self.decomp_levels
            data = self.high_pass_filter(cross_section)
            if data.size == 0:  # check if there is no data in the cross section
                continue

            # Perform wavelet packet decomposition
            try:
                coeffs = pywt.wavedec(data, wavelet, level=level, mode='symmetric')
            except UserWarning:
                continue

            max_coeffs_last = np.abs(max(coeffs[-1]))
            threshold = np.abs(max(coeffs[-1])) * 0.25
            self._store_plot_data(coeffs, cross_section, data, level)

            if max_coeffs_last >= 0.5:
                high_freq_indices = np.where(np.abs(coeffs[-1]) > threshold)[0]
                time_points_high_freq = high_freq_indices * (len(data) / len(coeffs[-1]))

                if time_points_high_freq.size > 0:
                    high_freqs = [
                        _HighFrequencyCell(
                            cross_section.row_start, int(point), FREQUENCY_FLAGGED
                        ) for point in time_points_high_freq.tolist()
                    ]
                    self.high_frequencies.extend(high_freqs)

    def _store_plot_data(self, coeffs, cross_section, data, level):
        """

        :param coeffs:
        :param cross_section:
        :param data:
        :param level:
        :return:
        """

        row = cross_section.row_start
        col = cross_section.col_start

        plot_row = row if not cross_section.vertical else 0
        plot_col = col if cross_section.vertical else 0
        plot_data = _PlotData(plot_row, plot_col, coeffs, data, level)
        self.plot_data.append(plot_data)

    @staticmethod
    def write_plot(plot_data: _PlotData, name="", outdir=""):

        row = plot_data.row
        col = plot_data.col
        coeffs = plot_data.coeffs
        data = plot_data.data
        level = plot_data.levels

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

        name = name + f"_row{row}_col{col}.png"
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
    dem_wave = DemWave(TestDemFile.NOISY, WaveletType.DB2, decomp_levels=4)

    # compute all the cross-sections horizontally
    dem_wave.set_cross_sections(vertical=False)

    # extract the first 5 cross-sections from all the cross-sections
    sections = dem_wave.get_cross_sections(0, 100)

    dem_wave.find_high_frequency(sections)

    outdir = r"C:\Users\LETHIER\work\waveform\data\dem\out"

    dem_wave.write_result(outdir=outdir)

    plot_out = r"C:\Users\LETHIER\work\waveform\data\cross_sections"
    for plot in dem_wave.plot_data:
        dem_wave.write_plot(plot, outdir=plot_out)


if __name__ == "__main__":
    main()
