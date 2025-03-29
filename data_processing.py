def data_processing(df):
    if df.columns[0] == '#Name':
        df.columns = list(df.columns[1:]) + [""]

        column_to_check = 'Car.CFL.Tire.FrcC.y'
        last_non_zero_index = df[column_to_check][::-1].ne(0).idxmax()
        df = df.iloc[:last_non_zero_index + 1]
    return df

def experiments_data_processing(df):
    if df.columns[0] == '#Name':
        df.columns = list(df.columns[1:]) + [""]

        column_to_check = 'Car.YawRate'
        last_non_zero_index = df[column_to_check][::-1].ne(0).idxmax()
        df = df.iloc[:last_non_zero_index + 1]
    return df