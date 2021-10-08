import pandas as pd
import numpy as np
import warnings
import csv
import copy
from datetime import date, timedelta

DF = pd.read_csv('../input_dataset/rl-kpis_mod.csv', low_memory=False)


def getHeaders():
    headersList = DF.columns

    # rl-site and met-stations
    # conFeat = ['height']
    # catFeat = ['clutter_class']

    # met-rel
    # conFeat = ['temp', 'temp_max', 'temp_min', 'wind_dir', 'wind_speed', 'wind_dir_max', 'wind_speed_max', 'humidity',
    #            'precipitation', 'precipitation_coeff', 'pressure', 'pressure_sea_level']
    # catFeat = []

    # rl-kpis
    conFeat = ['mw_connection_no',
               'link_length', 'severaly_error_second', 'error_second', 'unavail_second', 'avail_time',
               'bbe', 'rxlevmax', 'scalibility_score', 'capacity']
    catFeat = ['type', 'tip', 'direction', 'polarization', 'card_type',
               'adaptive_modulation', 'freq_band', 'modulation']

    # forecast_day1 conFeat = ['temp_max_day1','temp_min_day1','humidity_max_day1','humidity_min_day1',
    # 'wind_dir_day1','wind_speed_day1'] catFeat = ['weather_day1']

    return headersList, conFeat, catFeat


def processContinuous(conFeat, data):
    conHead = ['Count', 'Miss %', 'Card.', 'Min', '1st Qrt.', 'Mean', 'Median', '3rd Qrt', 'Max', 'Std. Dev.']

    conOutDF = pd.DataFrame(index=conFeat, columns=conHead)
    conOutDF.index.name = 'FEATURE_NAME'
    columns = data[conFeat]

    # COUNT
    count = columns.count()
    conOutDF[conHead[0]] = count

    # MISS %
    percents = [''] * len(conFeat)
    for col in columns:
        percents[conFeat.index(col)] = 0.00

    conOutDF[conHead[1]] = percents

    # CARDINALITY
    conOutDF[conHead[2]] = columns.nunique()

    # MINIMUM
    conOutDF[conHead[3]] = columns.min()

    # 1ST QUARTILE
    conOutDF[conHead[4]] = columns.quantile(0.25)

    # MEAN
    conOutDF[conHead[5]] = round(columns.mean(), 2)

    # MEDIAN
    conOutDF[conHead[6]] = columns.median()

    # 3rd QUARTILE
    conOutDF[conHead[7]] = columns.quantile(0.75)

    # MAX
    conOutDF[conHead[8]] = columns.max()

    # STANDARD DEVIATION
    conOutDF[conHead[9]] = round(columns.std(), 2)

    return conOutDF


def processCategorical(catFeat, data):
    catHead = ['Count', 'Miss %', 'Card.', 'Mode', 'Mode Freq', 'Mode %','2nd Mode', '2nd Mode Freq', '2nd Mode %']

    catOutDF = pd.DataFrame(index=catFeat, columns=catHead)
    catOutDF.index.name = 'FEATURE_NAME'
    columns = data[catFeat]

    # COUNT
    count = columns.count()
    catOutDF[catHead[0]] = count

    # CARDINALITY
    catOutDF[catHead[2]] = columns.nunique()

    # preparing arrays for storing data
    amt = len(catFeat)
    missPercents = [''] * amt
    modeFreqs = [''] * amt
    modes = [''] * amt
    modes2 = [''] * amt
    modePercents = [''] * amt
    modeFreqs2 = [''] * amt
    modePercents2 = [''] * amt

    for col in columns:
        values = columns[col].value_counts()
        index = catFeat.index(col)

        # MISS %
        try:
            NullCount = values.loc[' ']
            percent = (NullCount / count[index]) * 100
            missPercents[index] = round(percent, 2)

            catOutDF['Card.'][index] -= 1
        except Exception as e:
            missPercents[index] = 0.00

        # MODES
        mode = values.index[0]
        # mode2 = values.index[1]
        modes[index] = mode
        # modes2[index] = mode2

        # MODE FREQ
        modeCount = values.loc[mode]
        # modeCount2 = values.loc[mode2]
        modeFreqs[index] = modeCount
        # modeFreqs2[index] = modeCount2

        # MODE %
        miss = missPercents[index]

        modePer = (modeCount / (count[index] * ((100 - miss) / 100))) * 100
        modePercents[index] = round(modePer, 2)

        # modePer2 = (modeCount2 / (count[index] * ((100 - miss) / 100))) * 100
        # modePercents2[index] = round(modePer2, 2)

    catOutDF[catHead[1]] = missPercents
    catOutDF[catHead[3]] = modes
    catOutDF[catHead[4]] = modeFreqs
    catOutDF[catHead[5]] = modePercents
    # catOutDF[catHead[6]] = modes2
    # catOutDF[catHead[7]] = modeFreqs2
    # catOutDF[catHead[8]] = modePercents2

    return catOutDF


def main():
    warnings.simplefilter(action='ignore', category=Warning)

    allHead, conFeat, catFeat = getHeaders()

    # READ DATA, with headers joined on
    data = DF

    # PROCESS DATA
    conOutDF = processContinuous(conFeat, data)
    catOutDF = processCategorical(catFeat, data)

    # WRITE TO FILES
    conOutDF.to_csv("kpis_con1.csv")
    catOutDF.to_csv("kpis_cat1.csv")


if __name__ == '__main__':
    main()
