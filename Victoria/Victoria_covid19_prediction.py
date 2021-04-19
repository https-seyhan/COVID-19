import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from datetime import datetime


teststartdate = '2020-01-25'
period = 14 # moving average period is 14 days
figsize = (1500 / 50, 400 / 50)
alpha=.90
coef = 90

os.chdir('/home/saul/corona')


def getVicdata():
    fig, ax = plt.subplots(figsize=figsize)
    print(f'Victoria')
    dailydata = pd.read_csv('cases_daily_state.csv', parse_dates=['Date'], sep=',')


    vicdata = dailydata[['Date', 'VIC']] # keep two variables Date and VIC

    print((vicdata['VIC'].describe()))

    vicdata['newDate'] = vicdata['Date'].apply(lambda x: str(x) + '/20')
    vicdata['newDate'] = vicdata['newDate'].apply(lambda x: pd.to_datetime(x))
    #vicdata['newDate'] = datetime.strptime(vicdata['newDate'], '%d/%m/%y')

    vicdata = vicdata[['newDate', 'VIC']]

    summaydata = pd.pivot_table(data=vicdata, values=['VIC'], index=['newDate'], aggfunc=np.sum)
    flattened = pd.DataFrame(summaydata.to_records())
    flattened.set_index('newDate', inplace=True)

    rolling = flattened.rolling(period,
                                 win_type='gaussian',
                                 min_periods=1,
                                 center=True).mean(std=2).round()

    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    ax.set_title(f"NSW COVID-19 Cases", fontweight='bold')
    ax.set_ylabel('covid - 19 cases', fontweight='bold')
    ax.set_xlabel('Date', fontweight='bold')
    # ax.grid(which='major', axis='y', c='k', alpha=.3, zorder=-2)
    # ax.margins(0)

    #ax.set_xlim(pd.Timestamp(teststartdate), flattened.index.get_level_values('newDate')[-1] + pd.Timedelta(days=1))
    ax.set_xlim(pd.Timestamp(teststartdate), pd.Timestamp('2020-06-25'))
    fig.set_facecolor('w')

    # Plot graphs
    ax.plot( flattened, color='blue', linestyle='dashdot',  label='Detected Covid-19 cases')

    ax.legend(['Detected Cov-19 cases'])
    #vicdata.plot(vicdata['newDate'], vicdata['VIC'])
    #vicdata.plot()
    plt.show()

if __name__ == '__main__':
    getVicdata()

