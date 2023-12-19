import pandas as pd
import numpy as np

df = pd.read_csv(r'acs_data2.csv')

old_column_names = [
    'Cycle Number', 'Heat Rate 1 [C/min]', 'Ramp 1 Duration [min]', 'Temperature Dwell 1 [min]',
    'Heat Rate 2 [C/min]', 'Ramp 2 Duration [min]', 'Temperature Dwell 2 [min]',
    'Vacuum Pressure (*Patm) [Pa]', 'Vacuum Start Time [min]', 'Vacuum Duration [min]',
    'Autoclave Pressure (*Patm) [Pa]', 'Autoclave Start Time [min]', 'Autoclave Duration [min]',
    'AD. Porosity (%)', 'PR. Porosity (%)', 'Eff. Porosity (%)',
    'Max (Fiber Volume Fraction) (%)', 'Cure Cycle Total Time [min]', 'AD. Volume [m^3]','PR. Volume [m^3]'
]

new_column_names = [
    'cycle_number', 'heat_rate_1_c_min', 'ramp_1_duration_min', 'temperature_dwell_1_min',
    'heat_rate_2_c_min', 'ramp_2_duration_min', 'temperature_dwell_2_min',
    'vacuum_pressure_patm_pa', 'vacuum_start_time_min', 'vacuum_duration_min',
    'autoclave_pressure_patm_pa', 'autoclave_start_time_min', 'autoclave_duration_min',
    'ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent',
    'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min', 'ad_volume', 'pr_volume'
]

df.rename(columns=dict(zip(old_column_names, new_column_names)), inplace=True)

df = df.fillna(df.mean())

# Separating dependent variables
dependent_variables = [
    'ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent',
    'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min', 'ad_volume', 'pr_volume'
]

independent_variables = [column for column in df.columns if column not in dependent_variables]

# Removing correlated variables from the set of independent variables so as to avoid multicollinearity problems
correlation_threshold = 0.95
correlations = df[independent_variables].corr().abs()
upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

# Defining function to remove random rows from the dataset for out of sample testing of the model
def remove_random_rows(df):
    df.drop(columns=to_drop, inplace=True)

    random_indices = np.random.choice(df.index, size=30, replace=False)
    random_rows = df.loc[random_indices]

    # Remove these rows from the original DataFrame
    df = df.drop(random_indices)
    # Write the randomly selected rows to a .csv file
    random_rows.to_csv('oos_testing_data.csv', index=False)
    return df

df= remove_random_rows(df)
df.to_csv('acs_data2_cleaned.csv', index=False)