import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from datetime import datetime
import os
import re

teststartdate = '2020-01-25'
period = 14 # moving average period is 14 days
figsize = (1500 / 50, 400 / 50)
alpha=.90
coef = 90

os.chdir('/home/saul/corona')


def getVicdata():

    print(f'Victoria')
    dailydata = pd.read_csv('cases_daily_state.csv', parse_dates=['Date'], sep=',')

    vicdata = dailydata[['Date', 'VIC']] # keep two variables Date and VIC
    print("Head ", vicdata.head())
    vicdata['Date'] = vicdata['Date'].apply(lambda x: x.replace("/", "-"))
    print("Head After", vicdata.head())

    #print((vicdata['VIC'].describe()))

    vicdata['newDate'] = vicdata['Date'].apply(lambda x: str(x) + '-2020')

    vicdata['newDate2'] = vicdata['newDate'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))

    summarydata = pd.pivot_table(data=vicdata, values=['VIC'], index=['newDate2'], aggfunc=np.sum)

    flattened = pd.DataFrame(summarydata.to_records())

    flattened.set_index('newDate2', inplace=True)

    vicdata = vicdata[['newDate', 'VIC']]

    plotVivCov19(flattened)

def plotVivCov19(flattened):

    rolling = flattened.rolling(period,
                                win_type='gaussian',
                                min_periods=1,
                                center=True).mean(std=2).round()

    fig, ax = plt.subplots(figsize=figsize)
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    ax.set_title(f"VIC COVID-19 Cases", fontweight='bold')
    ax.set_ylabel('covid - 19 cases', fontweight='bold')
    ax.set_xlabel('Date', fontweight='bold')
    # ax.grid(which='major', axis='y', c='k', alpha=.3, zorder=-2)
    # ax.margins(0)

    print(flattened)

    ax.set_xlim(pd.Timestamp(teststartdate), flattened.index.get_level_values('newDate2')[-1] + pd.Timedelta(days=1))

    fig.set_facecolor('w')

    # Plot graphs
    ax.plot(flattened, color='blue', linestyle='dashdot', label='Detected Covid-19 cases')

    ax.legend(['Detected Cov-19 cases'])
    ax.plot(rolling, color='red', zorder=1, alpha=alpha, label = 'Fortnightly Moving Average of Detected Covid-19 cases')

    legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
    plt.show()

if __name__ == '__main__':
    getVicdata()

