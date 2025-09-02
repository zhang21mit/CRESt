import numpy as np
import pandas as pd
import sigfig
from PyEIS import EIS_exp, Parameters
from sklearn.linear_model import LinearRegression


class Analyzer:
    def __init__(self, data_dir, technique, sample_area, **kwargs):
        self.sample_area = sample_area
        self.technique = technique

        if self.technique == 'LSV':
            self.data = self.load_data(f'{data_dir}.csv')
            assert 'I/mA' in self.data.columns
            assert 'Ewe/V' in self.data.columns
        elif self.technique == 'CA':
            self.data = self.load_data(f'{data_dir}.csv')
            assert 'time/s' in self.data.columns
            assert 'I/mA' in self.data.columns
            assert 'Ewe/V' in self.data.columns
        elif self.technique == 'PEIS' or 'GEIS':
            path, data = data_dir.rsplit('/', 1)
            self.eis_data = EIS_Analyzer(path, [f'{data_dir}.mpt'])

    @staticmethod
    def load_data(data_dir):
        return pd.read_csv(
            data_dir,
            sep=';',
        )

    @staticmethod
    # get the closest potential value given a specific current value, consider multi modal LSV curve
    def get_closest_value(look_up_value, df, look_up_col_name, return_col_name):
        # reverse the df to efficiently find the last up going curve
        df_rev = df.iloc[::-1]
        for i, value in df_rev[look_up_col_name].items():
            if value - look_up_value < 0:
                return df[return_col_name].iloc[i:i + 2].mean()

    # get the maximum current from LSV data
    def get_max_i(self):
        assert self.technique == 'LSV'
        return round(self.data['I/mA'].max(), 1)

    def iR_correction(self, Rs):
        assert self.technique == 'LSV'
        self.data['Ewe/V'] = self.data['Ewe/V'] - self.data['I/mA'] * Rs


class OERAnalyzer(Analyzer):
    def __init__(self, data_dir, technique, ref_potential, pH, sample_area, **kwargs):
        """
        :param ref_potential: ref potential vs SHE 
        :param pH: pH of electrolyte
        :param current_density: target current density of calculating overpotential
        """
        super().__init__(data_dir, technique, sample_area, **kwargs)
        self.ref_potential = ref_potential
        self.pH = pH

        # try to see if iR_correction in kwargs
        if 'iR_correction' in kwargs and kwargs['iR_correction'] is True:
            assert 'Rs' in kwargs, 'Rs is needed for iR correction'
            self.iR_correction(kwargs['Rs'])

    # get the alkline OER overpotential in the data
    def get_overpotential(self, current_density=10):
        assert self.technique == 'LSV'

        # calculate the absolute current value
        I_overpotential = current_density * self.sample_area
        observed_potential = self.get_closest_value(I_overpotential, self.data, 'I/mA', 'Ewe/V')
        overpotential = observed_potential + (self.ref_potential + 0.059 * self.pH) - 1.23
        return sigfig.round(overpotential, 4)

    # get the tafel slope from LSV data
    def tafel_slope_analysis(self, tafel_range, p_low, p_high, I_low, I_high):
        """
        :param tafel_range: specify the choice of tafel slope calculation range, either 'current' or 'potential'
        :param p_low: low end of the tafel slope calculation range, in observed potential
        :param p_high: high end of the tafel slope calculation range, in observed potential
        :param I_low: low end of the tafel slope calculation range, in current density
        :param I_high: high end of the tafel slope calculation range, in current density
        :return: tafel slope, r2 of the linear fit
        """
        assert self.technique == 'LSV'
        assert tafel_range == 'current' or 'potential', 'tafel_range must be current or potential'
        if tafel_range == 'current':
            assert I_low is not None and I_high is not None, 'I_low and I_high must be specified'
            p_low = self.get_closest_value(I_low * self.sample_area, self.data, 'I/mA', 'Ewe/V')
            p_high = self.get_closest_value(I_high * self.sample_area, self.data, 'I/mA', 'Ewe/V')
        elif tafel_range == 'potential':
            assert p_low is not None and p_high is not None, 'p_low and p_high must be specified'
        onset_data = self.data[(self.data['Ewe/V'] > p_low) & (self.data['Ewe/V'] < p_high)]
        if len(onset_data) == 0:
            return np.nan, np.nan
        else:
            onset_data = onset_data.assign(log_I=lambda data: np.log10(data['I/mA']))
            x = onset_data.loc[:, 'log_I'].values.reshape(-1, 1)
            y = onset_data.loc[:, 'Ewe/V'].values
            model = LinearRegression()
            model.fit(x, y)
            return sigfig.round(model.coef_[0], 4), sigfig.round(model.score(x, y), 4)

    def get_tafel_slope(self, tafel_range, p_low=None, p_high=None, I_low=None, I_high=None):
        return self.tafel_slope_analysis(tafel_range, p_low, p_high, I_low, I_high)[0]

    def get_tafel_slope_fit(self, tafel_range, p_low=None, p_high=None, I_low=None, I_high=None):
        return self.tafel_slope_analysis(tafel_range, p_low, p_high, I_low, I_high)[1]


class AlklineOERAnalyzer(OERAnalyzer):
    def __init__(
            self, 
            data_dir, 
            ref_potential=0.098, 
            pH=14, 
            sample_area=1,
            technique='LSV', 
            **kwargs
    ):
        super().__init__(
            data_dir, 
            technique, 
            ref_potential, 
            pH, 
            sample_area,
            **kwargs
        )


class AcidicOERAnalyzer(OERAnalyzer):
    def __init__(
            self,
            data_dir,
            ref_potential=0.197,
            pH=1,
            sample_area=1,
            technique='LSV',
            **kwargs
    ):
        super().__init__(
            data_dir,
            technique,
            ref_potential,
            pH,
            sample_area,
            **kwargs
        )


class AlklineFORAnalyzer(Analyzer):

    # get the maximum power from LSV data
    def get_max_power(self, ref_potential, counter_potential):
        assert self.technique == 'LSV'

        # working potential vs RHE
        working_potential = self.data['Ewe/V'] - ref_potential

        # voltage output calculation, create a pd series, calculate abs between working potential and counter potential
        voltage_output = working_potential.apply(lambda x: abs(x - counter_potential))

        # calculate the power at each time frame
        power = voltage_output * self.data['I/mA']

        return sigfig.round(power.max(), 4)


class AcidicMORAnalyzer(Analyzer):

    # get the total energy output from CA data
    def get_total_energy(self, ref_potential, counter_potential):
        assert self.technique == 'CA'

        # working potential vs RHE
        working_potential = self.data['Ewe/V'] - ref_potential

        # voltage output calculation, create a pd series, calculate abs between working potential and counter potential
        voltage_output = working_potential.apply(lambda x: abs(x - counter_potential))

        # calculate the power at each time frame
        power = voltage_output * self.data['I/mA']

        # calculate the energy output by integrating the power over time
        time_diff = self.data['time/s'].diff()
        energy_output = (power * time_diff).cumsum()

        return sigfig.round(energy_output.iloc[-1], 4)

    # get the fitting result from EIS data
    def get_EIS_fitting_result(self, circuit, param_df):
        assert self.technique == 'PEIS' or 'GEIS'
        return self.eis_data.get_fitting_result(circuit, param_df)

    # get the maximum power from LSV data
    def get_max_power(self, ref_potential, counter_potential):
        assert self.technique == 'LSV'

        # working potential vs RHE
        working_potential = self.data['Ewe/V'] - ref_potential

        # voltage output calculation, create a pd series, calculate abs between working potential and counter potential
        voltage_output = working_potential.apply(lambda x: abs(x - counter_potential))

        # calculate the power at each time frame
        power = voltage_output * self.data['I/mA']

        return sigfig.round(power.max(), 4)


class EIS_Analyzer(EIS_exp):
    def __init__(self, path, data):
        """
        :param path: data storage path
        :param data: data file name
        """
        super().__init__(path, data)

    def get_fitting_result(self, circuit, param_df):
        """
        :param circuit: ecm model, e.g. 'R-RQ'
        :param param_df: e.g.
            pd.DataFrame(columns=['name', 'value', 'min', 'max'], data=[
                    ['Rs', 50, 0.1, 100],
                    ['R', 200, 1, 1000],
                    ['Q', 1, 1e-4, 10],
                    ['n', 0.8, 0.5, 2]
                ]
            )
        """
        fit_params = Parameters()
        # add each parameter value, min, max to fit_params from param_df
        for i, row in param_df.iterrows():
            fit_params.add(row['name'], value=row['value'], min=row['min'], max=row['max'])
        self.EIS_fit(fit_params, circuit)
        fit_result = {
            param_name: [sigfig.round(self.__getattribute__(f'fit_{param_name}')[0], 4)]
            for param_name in param_df['name']
        }
        return pd.DataFrame.from_dict(fit_result)
