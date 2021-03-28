import pandas as pd
import numpy as np
import csv
import seaborn as sb
import operator
from urllib.request import urlopen
from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from collections import Counter
from matplotlib.colors import ListedColormap
from matplotlib import dates as mdates
from matplotlib import ticker
from scipy import stats as sps
from scipy.interpolate import interp1d

dailytests ="https://covid.ourworldindata.org/data/owid-covid-data.csv"
dailytests = pd.read_csv(dailytests, parse_dates=['date'], squeeze=True, sep=',')
AUdailytests = dailytests[dailytests['location'] == 'Australia']

keepAU = ['date', 'total_tests', 'new_cases', 'total_cases','population']

AUdailytests = AUdailytests[keepAU]

#drop na
#AUdailytests = AUdailytests[AUdailytests['total_tests'].notna()]

AUdailytests.head()
AUdailytests['new_cases'].astype(float)
AUdailytests['DeltaCase'] = AUdailytests['new_cases'].diff()
AUdailytests['DeltaCase'] = AUdailytests['DeltaCase'].fillna(0)
AUdailytests.set_index("date")

fig, ax = plt.subplots(figsize=(1500 / 50, 400 / 50))
ax.set_title(f"Daily New Cases Change", fontweight='bold')
ax.set_ylabel(' Change of new cases from previous day (day - [day-1])', fontweight='bold')
ax.set_xlabel('Date', fontweight='bold')

plt.plot(AUdailytests['date'], AUdailytests['DeltaCase'], color='tab:red')
plt.show()
#calculate ratio of tested vs, population

AUdailytests['testratio'] =  AUdailytests.apply(lambda row: row['total_tests'] / row['population'], axis=1)
#calculate new tests to total tests ratio
AUdailytests['newcasestotalratio'] = AUdailytests['new_cases'] / AUdailytests['total_cases']

#Plot ratio
AUdailytests.set_index("date")
fig, ax = plt.subplots(figsize=(1500 / 50, 400 / 50))
ax.set_title(f"New cases vs. Total cases ratio", fontweight='bold')
ax.set_ylabel('Ratio', fontweight='bold')
ax.set_xlabel('Date', fontweight='bold')
plt.plot(AUdailytests['date'], AUdailytests['newcasestotalratio'], color='tab:red')
plt.show()

#Daily realtime corona cases data
keep = ['notification_date', 'postcode']
dailycases = "https://data.nsw.gov.au/data/dataset/aefcde60-3b0c-4bc0-9af1-6fe652944ec2/resource/21304414-1ff1-4243-a5d2-f52778048b29/download/covid-19-cases-by-notification-date-and-postcode-local-health-district-and-local-government-area.csv"

coronadata = pd.read_csv(dailycases, parse_dates=['notification_date'], squeeze=True, sep=',')

#keep notification_date and postcode
#Clean data
coronadata = coronadata[(coronadata['postcode'] != 0)]
coronadata = coronadata[(coronadata['postcode'].notna())]
coronadata = coronadata[keep]
coronadata['notification_date'] = pd.to_datetime(coronadata['notification_date'])

#create new cases
coronadata['cases'] = 1

def plotNSW():
    fig, ax = plt.subplots(figsize=(1500 / 50, 400 / 50))
    summaydata = pd.pivot_table(data=coronadata, values=['cases'], index=['notification_date'], aggfunc=np.sum)
    flattened = pd.DataFrame(summaydata.to_records())
    flattened.set_index('notification_date', inplace=True)
    print("Flattened ", flattened.head(5))

    rolling = flattened.rolling(14,
                                 win_type='gaussian',
                                 min_periods=1,
                                 center=True).mean(std=2).round()

    print("Rolling ", rolling.head(10))

    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.set_title(f"NSW COVID-19 Cases", fontweight='bold')
    ax.set_ylabel('covid - 19 cases', fontweight='bold')
    ax.set_xlabel('Date', fontweight='bold')
    #ax.grid(which='major', axis='y', c='k', alpha=.3, zorder=-2)
    #ax.margins(0)

    ax.set_xlim(pd.Timestamp('2020-01-22'), flattened.index.get_level_values('notification_date')[-1] + pd.Timedelta(days=1))
    fig.set_facecolor('w')

    # Plot graphs
    ax.plot(flattened, color='blue', linestyle='dashdot', zorder=1, alpha=.90, label = 'Detected Covid-19 cases')
    ax.legend(['Detected Cov-19 cases'])
    ax.plot(rolling, color='red', zorder=1, alpha=.90, label = 'Fortnightly Moving Average of Detected Covid-19 cases')

    legend = ax.legend(loc='upper left', shadow=True, fontsize='large')

    # Put a nicer background color on the legend.
    legend.get_frame()
    plt.show()
    return rolling

movingAverage = plotNSW()

def get_posteriors(ma, sigma=0.15):
    GAMMA = 1 / 14 # 1 divided by the moving average

    # We create an array for every possible value of Rt
    R_T_MAX = 6
    r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 10 + 6)
    ma = ma.cases # get cases to be input the the lambda calculation
    # (1) Calculate Lambda

    lam = ma[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

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
    prior0 /= prior0.sum()

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

# Note that we're fixing sigma to a value just for the example
posteriors, log_likelihood = get_posteriors(movingAverage, sigma=.25)

ax = posteriors.plot(title=' NSW - Daily Posterior for $R_t$',
           legend=False,
           lw=1,
           c='k',
           alpha=.3,
           xlim=(0.4,6))

ax.set_xlabel('$R_t$');
plt.show()

def highest_density_interval(pmf, p, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns

    if (isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)

    cumsum = np.cumsum(pmf.values)

    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p

    lows, highs = (total_p > p).nonzero()

    # Find the smallest range (highest density)
    best = (highs - lows).argmin()

    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]

    return pd.Series([low, high],
                     index=[f'Low_{p * 100:.0f}',
                            f'High_{p * 100:.0f}'])


hdi = highest_density_interval(posteriors, p=.75, debug=True)


#Note that this takes a while to execute - it's not the most efficient algorithm
hdis = highest_density_interval(posteriors, p=.75)

most_likely = posteriors.idxmax().rename('ML')

# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)

most_likely_values = posteriors.idxmax(axis=0)

ax = most_likely_values.plot(marker='o',
                             label='Most Likely',
                             title=f'$R_t$ by day',
                             c='blue',
                             markersize=4)

ax.fill_between(hdi.index,
                hdi['Low_75'],
                hdi['High_75'],
                color='blue',
                alpha=.1,
                lw=0,
                label='HDI')

ax.legend();
plt.show()

def plot_rt(result, ax):
    ax.set_title(f"NSW")

    # Colors
    ABOVE = [1, 0, 0]
    MIDDLE = [1, 1, 1]
    BELOW = [0, 0, 0]
    cmap = ListedColormap(np.r_[
                              np.linspace(BELOW, MIDDLE, 25),
                              np.linspace(MIDDLE, ABOVE, 25)
                          ])
    color_mapped = lambda y: np.clip(y, .5, 1.5) - .5

    index = result['ML'].index.get_level_values('notification_date')
    values = result['ML'].values

    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)

    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low_75'].values,
                     bounds_error=False,
                     fill_value='extrapolate')

    highfn = interp1d(date2num(index),
                      result['High_75'].values,
                      bounds_error=False,
                      fill_value='extrapolate')

    extended = pd.date_range(start=pd.Timestamp('2020-01-22'),
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
    ax.set_ylim(0.0, 5.0)
    ax.set_xlim(pd.Timestamp('2020-01-22'), result.index.get_level_values('notification_date')[-1] + pd.Timedelta(days=1))
    fig.set_facecolor('w')


fig, ax = plt.subplots(figsize=(600 / 72, 400 / 72))

plot_rt(result, ax)
ax.set_title(f'Real-time $R_t$ for NSW')
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.show()
