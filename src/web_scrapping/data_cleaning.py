import pandas as pd
import os
import re


def clean_data(dir:str):
    impt_cols = [
        'Make', 'Model', 'Year', 'Picture ID', 'MSRP', 'Front Wheel Size (in)', 'SAE Net Horsepower @ RPM',
        'Displacement', 'Engine Type', 'Width, Max w/o mirrors (in)', 'Height, Overall (in)',
        'Length, Overall (in)', 'Gas Mileage', 'Drivetrain', 'Passenger Capacity', 'Passenger Doors',
        'Body Style']

    df = pd.read_csv(os.path.join(dir, 'specs-and-pics.csv'), dtype=str, index_col=0).T
    df_id = df.loc[:, df.columns.str.startswith('Picture')]
    df_id.index = df_id.index.to_series().apply(lambda x: ' '.join(x.split(' ')[:-1]))
    df_spec = df[impt_cols]

    def split_text(engine):
      pattern = re.compile(r'(Gas)?(/)?(Electric)?\s?((I|V)-?[0-9])?')
      matches = pattern.finditer(engine)
      matches_lst = [i for i in matches]
      return matches_lst[-2].group(0)

    # Data Cleaning
    df_spec['SAE Net Horsepower @ RPM'] = df_spec['SAE Net Horsepower @ RPM'].str[:3]
    df_spec['Front Wheel Size (in)'] = df_spec['Front Wheel Size (in)'].str[:2]
    df_spec['MSRP'] = df_spec['MSRP'].str.replace(',', '').str.replace('$', '')
    df_spec['Displacement'] = df_spec['Displacement'].apply(lambda x: x.split('/')[0] if type(x) == str else x)
    df_spec['Engine Type'] = df_spec['Engine Type'].apply(lambda x: split_text(x) if type(x) == str else x)
    df_spec['Engine Type'] = df_spec['Engine Type'].apply(lambda x: x.replace('-', ''))
    df_spec['Gas Mileage'] = df_spec['Gas Mileage'].str[:6]
    df_spec['Drivetrain'] = df_spec['Drivetrain'].str.replace('-', ' ').str.replace('All', '4')
    df_spec['Drivetrain'] = df_spec['Drivetrain'].str.replace('-', ' ').str.replace('Front', 'F')
    df_spec['Drivetrain'] = df_spec['Drivetrain'].str.replace('-', ' ').str.replace('Rear', 'R')
    df_spec['Drivetrain'] = df_spec['Drivetrain'].str.replace('Wheel Drive', 'WD')
    df_spec['Drivetrain'] = df_spec['Drivetrain'].str.replace(' ', '')


    df_spec.loc[df_spec['Body Style'].str.contains('Pickup', na=False), 'Body Style'] = 'Pickup'
    df_spec.loc[df_spec['Body Style'].str.contains('Sport Utility', na=False), 'Body Style'] = 'SUV'
    df_spec.loc[df_spec['Body Style'].str.contains('van', na=False), 'Body Style'] = 'Van'
    df_spec['Body Style'] = df_spec['Body Style'].str.replace('Hatchback', '2dr')
    df_spec['Body Style'] = df_spec['Body Style'].str.replace(' Car', '')

    df_spec['Width, Max w/o mirrors (in)'] = df_spec['Width, Max w/o mirrors (in)'].apply(lambda x: x.split('.')[0] if type(x) == str else x)
    df_spec['Height, Overall (in)'] = df_spec['Height, Overall (in)'].apply(lambda x: x.split('.')[0] if type(x) == str else x)
    df_spec['Length, Overall (in)'] = df_spec['Length, Overall (in)'].apply(lambda x: x.split('.')[0] if type(x) == str else x)

    df_spec.rename(columns={'SAE Net Horsepower @ RPM': 'SAE Net Horsepower'})

    df_spec['ID'] = df_spec.iloc[:, :4].apply(lambda x: '_'.join(x.astype(str)), axis=1)

    df_spec.to_csv(os.path.join(dir,'specs-cleaned.csv'))
    df_id.to_csv(os.path.join(dir,'pics-and-ids.csv'))

