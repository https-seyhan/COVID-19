import pandas as pd
import numpy as np
import csv
import seaborn as sb
import operator
import os
import re
from urllib.request import urlopen
from matplotlib import pyplot as plt
from pandas.plotting import table
from matplotlib.dates import date2num, num2date
from collections import Counter
from matplotlib.colors import ListedColormap
from matplotlib import dates as mdates
from matplotlib import ticker
from scipy import stats as sps
from scipy.interpolate import interp1d
from scipy import stats
from datetime import datetime

#Global variables
R_T_MAX = 6
GAMMA = 1 / 14 # 1 divided by the moving average
teststartdate = '2020-03-01'
period = 14 # moving average period is 14 days
figsize = (1500 / 50, 400 / 50)
alpha=.90
coef = 90

os.chdir('/home/saul/corona')

def getVicdata():
    print(f'Victoria')
    dailydata = pd.read_csv('cases_daily_state.csv', parse_dates=['Date'], sep=',')
    vicdata = dailydata[['Date', 'VIC']] # keep two variables Date and VIC
    vicdata['Date'] = vicdata['Date'].apply(lambda x: x.replace("/", "-"))
    vicdata['newDate'] = vicdata['Date'].apply(lambda x: str(x) + '-2020')
    vicdata['newDate2'] = vicdata['newDate'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))
    summarydata = pd.pivot_table(data=vicdata, values=['VIC'], index=['newDate2'], aggfunc=np.sum)
    flattened = pd.DataFrame(summarydata.to_records())
    flattened.set_index('newDate2', inplace=True)
    vicdata = vicdata[['newDate2', 'VIC']]

    rolling = flattened.rolling(period,
                                win_type='gaussian',
                                min_periods=1,
                                center=True).mean(std=2).round()
    calculateTotalCases(vicdata)
    calculatenewcasestotalratio(vicdata)
    #print("New Column Names ", vicdata.columns)
    #print("New cases ratio describe ", vicdata['newcasestotalratio'].describe())

    posteriors, log_likelihood = get_posteriors(rolling, vicdata['newcasestotalratio'], sigma=.25)
    #get_posteriors(rolling, sigma=0.25)
    plotPosteriors(posteriors)

    hdi = highest_density_interval(posteriors, p=alpha, debug=True)
    # Note that this takes a while to execute - it's not the most efficient algorithm
    hdis = highest_density_interval(posteriors, p=alpha)

    plotCoeffs(posteriors, hdi, hdis)
    most_likely = posteriors.idxmax().rename('ML')
    # Look into why you shift -1
    result = pd.concat([most_likely, hdis], axis=1)
    plot_rt(result)

def plotCoeffs(posteriors, hdi, hdis):
    most_likely = posteriors.idxmax().rename('ML')
    # Look into why you shift -1
    result = pd.concat([most_likely, hdis], axis=1)
    most_likely_values = posteriors.idxmax(axis=0)

    ax = most_likely_values.plot(marker='o',
                                 label='Most Likely',
                                 title=f'$R_t$ by day',
                                 c='blue',
                                 markersize=4,
                                 figsize=figsize)
    ax.fill_between(hdi.index,
                    hdi['Low_' + str(coef)],
                    hdi['High_' + str(coef)],
                    color='blue',
                    alpha=.1,
                    lw=0,
                    label='HDI')
    ax.legend();
    plt.show()

def plotPosteriors(posteriors):
    ax = posteriors.plot(title=' Improved Approach: VIC - Daily Posterior for $R_t$',
                         legend=False,
                         lw=1,
                         c='k',
                         alpha=.5,
                         xlim=(0.4, 6),
                         figsize=figsize)
    ax.set_xlabel('$R_t$');
    plt.show()

def highest_density_interval(posteriors, p, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if (isinstance(posteriors, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(posteriors[col], p=p) for col in posteriors],
                            index=posteriors.columns)

    cumsum = np.cumsum(posteriors.values)
    #print("Posterior Values", posteriors.values)
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    # Find the smallest range (highest density)
    best = (highs - lows).argmin()

    low = posteriors.index[lows[best]]
    high = posteriors.index[highs[best]]

    return pd.Series([low, high],
                     index=[f'Low_{p * 100:.0f}',
                            f'High_{p * 100:.0f}'])

def calculateTotalCases(vicdata):

    vicdata['total_cases'] = vicdata['VIC'].rolling(min_periods=1, window=1000).sum()

def calculatenewcasestotalratio(vicdata):
    #calculate new tests to total tests ratio. This ratio indicates the undetected and asymptomatic COVID-19 cases.
    #asymptomatic cases result in less accurate models due to its nature.
    vicdata['newcasestotalratio']  = vicdata['VIC'] / vicdata['total_cases']

#Calculate Bayesian posteriors
def get_posteriors(ma, newtotalratio, sigma=0.15):

    # We create an array for every possible value of Rt

    r_t_range = np.linspace(0, R_T_MAX, len(newtotalratio))
    #r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 10 + 5)
    ma = ma.VIC # get new cases to be used in the lambda calculation


    # (1) Calculate Lambda
    sumtwovecs = np.exp(GAMMA * ((r_t_range[:, None] - 1)))
    #print("Sum two vectors ", sumtwovecs)

    #previous model
    #lam = ma[:-1].values * sumtwovecs
    #print("New Total Ratio ", newtotalratio[:, None])
    #improved model
    lam = ma[:-1].values * sumtwovecs + newtotalratio[:, None]
    #print("Lamda ", lam)
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data=sps.poisson.pmf(ma[1:].values, lam),
        index=r_t_range,
        columns=ma.index[1:])

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                              ).pdf(r_t_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    prior0 = np.ones_like(r_t_range) / len(r_t_range)

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.

    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=ma.index,
        data={ma.index[0]: prior0}
    )

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(ma.index[:-1], ma.index[1:]):
        # (5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        # (5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator / denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)

    return posteriors, log_likelihood

def plotVicCov19(flattened, rolling):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    ax.set_title(f"VIC COVID-19 Cases", fontweight='bold')
    ax.set_ylabel('covid - 19 cases', fontweight='bold')
    ax.set_xlabel('Date', fontweight='bold')
    # ax.grid(which='major', axis='y', c='k', alpha=.3, zorder=-2)
    # ax.margins(0)

    ax.set_xlim(pd.Timestamp(teststartdate), flattened.index.get_level_values('newDate2')[-1] + pd.Timedelta(days=1))

    fig.set_facecolor('w')

    # Plot graphs
    ax.plot(flattened, color='blue', linestyle='dashdot', label='Detected Covid-19 cases')

    ax.legend(['Detected Cov-19 cases'])
    ax.plot(rolling, color='red', zorder=1, alpha=alpha, label = 'Fortnightly Moving Average of Detected Covid-19 cases')

    legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
    plt.show()

def plot_rt(result):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(f"VIC")
    ax.set_title(f'Improved Approach: Real-time $R_t$ for VIC')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # Colours
    ABOVE = [1, 0, 0]
    MIDDLE = [1, 1, 1]
    BELOW = [0, 0, 0]
    cmap = ListedColormap(np.r_[
                              np.linspace(BELOW, MIDDLE, 25),
                              np.linspace(MIDDLE, ABOVE, 25)
                          ])
    color_mapped = lambda y: np.clip(y, .5, 1.5) - .5

    index = result['ML'].index.get_level_values('newDate2')
    values = result['ML'].values

    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    ax.annotate('R0 jumps to greater than 1', (mdates.date2num(index[99]), values[99]),
                xytext=(50, 50), textcoords='offset points',
                arrowprops=dict(facecolor='black')
                )
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low_' + str(coef)].values,
                     bounds_error=False,
                     fill_value='extrapolate')

    highfn = interp1d(date2num(index),
                      result['High_' + str(coef)].values,
                      bounds_error=False,
                      fill_value='extrapolate')

    extended = pd.date_range(start=pd.Timestamp(teststartdate),
                             end=index[-1] + pd.Timedelta(days=1))

    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);

    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 6.0)
    ax.set_xlim(pd.Timestamp(teststartdate), result.index.get_level_values('newDate2')[-1] + pd.Timedelta(days=1))
    fig.set_facecolor('w')
    #plt.show()

    print("Dates ", index[99:106])
    print("R0 values ", values[99:106])

    #jumpinMay = pd.DataFrame(index= 'NewDate2', index[99:106], values[99:106])
    jumpinMay = pd.DataFrame({'Date': index[98:106],
                              'R0': values[98:106]})

    plotDataFrame(jumpinMay)
    #ax.legend(jumpinMay.values)
    print(jumpinMay)


def plotDataFrame(jumpinMay):
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    table(ax, np.round(jumpinMay, 4),
          loc='upper left', colWidths=[0.2, 0.2, 0.2])
    jumpinMay.plot(ax=ax, ylim=(1, 2), legend=None)
    plt.show()



if __name__ == '__main__':
    getVicdata()

