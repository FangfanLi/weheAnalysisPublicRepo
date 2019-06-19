'''
Detect content-based traffic differentiation
For each ISP-Replay-aggregationGroup combination:
Detect differentiation and fixed-rate throttling using average throughputs of original and bit-inverted replays.

Methodology in detail:

The detection methodology works in two steps:
1. Detect differentiation between original and bit-inverted replays
2. Identify fixed-rate throttling if differentiation is detected in the first step. Fixed

We group tests using ISP-replay-aggregationGroup.
There are several assumptions, first, throttling policies should be different among ISPs.
Second, if a certain ISP throttles, the throttling policy will be application/replay specific.
Last, throttling policies might change overtime, to account for the temporal change,
we can group tests using quarter, month (i.e., aggregationGroup).


Account for confounding factors in step 1:

Delayed throttling detection (only applied to original replays):
    Find delayed throttling period using PELT algorithm to detect change point in transmission timeseries.
    Tests with possible delayed throttling:
    1. *one* change point detected,
    2. throughput samples before and after this change point are from different distributions (K-S test)
    For each test with possible delayed throttling, get average throughputs before and after the change point.
    Delayed throttling is detected if
    1. Aggregated average throughputs before and after change points are from different distribution (KS test).
    2. Get two stats 1. time passed and 2. MB transmitted at change points. Use Kernel Density Estimation (KDE)
     to check whether time passed/MB transmitted at the change point aggregate at certain value, time/byte - based delay

Only throughput samples after the delayed period are used for throttling detection.

Throttling detection contains 3 tests, throttling detected only if all three tests passed:
    1. The average throughputs from original and random replays of the population are from different distributions (KS tests)
    2. KDE tests found values with high densities unique to the original replays in the population
    3. KDE tests found values with high densities from positive tests (tests detected differentiation individually),
       and the values are consistent with the values found in step 2

If the three tests passed, the values that are detected from both step 2 and step 3 are the throttling rates.

Several plots are generated for each test when throttling is detected:
    plot the KDE result/CDF of the delayed throttling detection
    plot the CDF/distribution of the average throughputs from all tests (both original and random)
    plot the CDF/distribution of the average throughputs from positive tests (both original and random)

Additional analysis:

OS difference:
    print the positive test ratio for iOS and android

Geographic difference (only support US state level analysis):
    plot the number of tests and positive test ratio in different stats

location of tests:
   plot the positive and negative tests on map based on their GPS locations

TimeOfDay effect:
    detect whether there is time of day effect via a chi-square test, plot the positive test ratio over 24 hours period

Users can specify how to aggregate the tests (quarter, month, week or all together (default)).
'''

import json, os, numpy, copy
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import geopandas as gpd
import pandas as pd
import sys
import ruptures as rpt
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema
from datetime import datetime as dt
import datetime
import math
from scipy.stats import ks_2samp
from scipy.stats import chisquare
import random
from matplotlib.ticker import MaxNLocator
import seaborn as sns

sns.set(style="darkgrid")


def readStatesPolygon():
    df = gpd.read_file("cb_2017_us_state_500k.shp")

    statesPolygons = {}

    for index, row in df.iterrows():
        statesPolygons[row['STUSPS']] = row['geometry']

    return statesPolygons


def get_US_states(longitude, latitude):
    statesPolygons = readStatesPolygon()
    geoPoint = Point(longitude, latitude)

    for state in statesPolygons:
        if geoPoint.within(statesPolygons[state]):
            return state

    return ''


def list2CDF(myList):
    myList = sorted(myList)

    x = [0]
    y = [0]

    for i in range(len(myList)):
        x.append(myList[i])
        y.append(float(i + 1) / len(myList))

    return x, y


# draw vertical lines and set parameters for plots
def drawXlines_CDF(vlines):
    colors = ['#a6611a', '#018571', '#ca0020', '#0571b0']
    count = 0
    for vline in vlines:
        rate = vlines[vline]
        plt.axvline(x=rate, color=colors[count % len(colors)], label='{}_{} Mbps'.format(vline, rate))
        count += 1


def drawYlines_CDF(vlines):
    colors = ['#a6611a', '#018571', '#ca0020', '#0571b0']
    count = 0
    for vline in vlines:
        rate = vlines[vline]
        plt.axhline(y=rate, color=colors[count % len(colors)], label='{}_{} Mbps'.format(vline, rate))
        count += 1


def plotCDF(CDFs, vlines, indiv_plotsDir, plotTitile='', xlabel='Throughput (Mbits/sec)'):
    fig, ax = plt.subplots()
    count = 0
    for test in CDFs:
        if ('Bit-inverted' in test) or ('After' in test):
            col = '#404040'
        else:
            col = '#0571b0'
        (x, y) = CDFs[test]
        plt.plot(x, y, '-', linewidth=2, color=col, label=test)
        count += 1

    vcolors = ['#a6611a', '#018571', '#ca0020', '#0571b0']
    count = 0
    for vline in vlines:
        plt.axvline(x=vline, linewidth=10, alpha=0.25, color=vcolors[count % len(vcolors)],
                    label='{} Mbps'.format(vline))
        count += 1

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(xlabel)
    plt.ylim((0, 1.1))
    # set font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    plt.legend(prop={'size': 15})
    fig.tight_layout()
    plt.savefig(indiv_plotsDir + '{}_xPutsCDF.png'.format(plotTitile),
                bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close('all')


def plotThroughput(tests, hlines, indiv_plotsDir, plotTitile):
    fig, ax = plt.subplots(figsize=(15, 6))
    count = 0
    for test in tests:
        if 'Bit-inverted' in test:
            col = '#404040'
        else:
            col = '#0571b0'
        ts, xPuts = tests[test]
        plt.plot(ts, xPuts, color=col, label=test)
        count += 1

    # drawYlines_CDF(hlines)
    plt.legend(prop={'size': 20})
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (Mbps)')
    fig.tight_layout()
    # set font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    plt.savefig(indiv_plotsDir + plotTitile + '_xputs.png', bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close('all')


# Create three plots
# 1. throughput CDFs (both original and random test in the same graph)
# 2. throughput CDF with only original test, with average throughput and detected throttling rate
# 3. throughput overtime, both original and random in the same graph
# plotTitile should include carrierName, replayName, and the uniqueTestID
def plotIndividualTest(timestampsOriginal, xPutsOriginal,
                       timestampsRandom, xPutsRandom,
                       vlines, indiv_plotsDir, plotTitle):
    if not os.path.exists(indiv_plotsDir):
        os.mkdir(indiv_plotsDir)

    xPutsOriginalSorted = sorted(xPutsOriginal)
    xPutsRandomSorted = sorted(xPutsRandom)
    # plot only the > 0 throughputs out for CDFs
    xPutsOriginalSorted = [x for x in xPutsOriginalSorted if x > 0]
    xPutsRandomSorted = [x for x in xPutsRandomSorted if x > 0]

    # Throughput CDF for comparing both original and random CDFs
    Xoriginal, Yoriginal = list2CDF(xPutsOriginalSorted)
    Xrandom, Yrandom = list2CDF(xPutsRandomSorted)

    twoCDFs = {'Original replay': (Xoriginal, Yoriginal), 'Bit-inverted replay': (Xrandom, Yrandom)}

    plotCDF(twoCDFs, vlines, indiv_plotsDir, plotTitle)

    tests_xputs = {'Original replay': (timestampsOriginal, xPutsOriginal),
                   'Bit-inverted replay': (timestampsRandom, xPutsRandom)}

    # Throughput over time, both original and random
    plotThroughput(tests_xputs, vlines, indiv_plotsDir, plotTitle)


# This function plot out the distribution of the negative tests
def plotNegative(negativeXputs, throttlingRates, plotDir, plotTitile=''):
    negativeXputs.sort()
    interval = negativeXputs[-1] / float(100)
    bins = []

    for i in range(100):
        bins.append(i * interval)

    plt.figure()
    fig, ax = plt.subplots()

    plt.hist(negativeXputs, bins)

    throttlingRatesCount = 0
    for throttlingRate in throttlingRates:
        plt.axvline(throttlingRate, color='r',
                    label='Throttling rate {}: {} Mbps'.format(throttlingRatesCount, throttlingRate))
        throttlingRatesCount += 1

    plt.xlabel('Average Throughput (Mbps)')
    plt.ylabel('Number of tests without differentiation')
    plt.legend()
    # set font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    fig.tight_layout()
    plt.savefig(plotDir + '/' + plotTitile + '_negative.png', bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close('all')


def groupTestsPerHour(positiveTests, negativeTests):
    # each hour is a list of tests where each member in the list is a two elements list [], which stores
    # [num positive tests, num negative tests] during that hour in the aggregationGroup
    # {'01' : [[num positive tests, num negative tests], [], ...], '02': [...], ...}
    testsGroupedByHour = {}
    # {'01': [[avgXputsOriginal1, ...], [avgXputsBitInverted, ...]]}
    throughputsGroupedByHour = {}
    for onePositiveTest in positiveTests:
        incomingTime = onePositiveTest['geoInfo']['localTime']
        if incomingTime:
            incomingHour = incomingTime.split(' ')[1].split('-')[0].split(':')[0]
            if incomingHour not in testsGroupedByHour:
                testsGroupedByHour[incomingHour] = [0, 0]
                throughputsGroupedByHour[incomingHour] = [[], []]
            testsGroupedByHour[incomingHour][0] += 1
            throughputsGroupedByHour[incomingHour][0].append(numpy.mean(onePositiveTest['original_xPuts'][0]))
            throughputsGroupedByHour[incomingHour][1].append(numpy.mean(onePositiveTest['random_xPuts'][0]))
    for oneNegativeTest in negativeTests:
        incomingTime = oneNegativeTest['geoInfo']['localTime']
        if incomingTime:
            incomingHour = incomingTime.split(' ')[1].split('-')[0].split(':')[0]
            if incomingHour not in testsGroupedByHour:
                testsGroupedByHour[incomingHour] = [0, 0]
                throughputsGroupedByHour[incomingHour] = [[], []]
            testsGroupedByHour[incomingHour][1] += 1
            throughputsGroupedByHour[incomingHour][0].append(numpy.mean(oneNegativeTest['original_xPuts'][0]))
            throughputsGroupedByHour[incomingHour][1].append(numpy.mean(oneNegativeTest['random_xPuts'][0]))

    return testsGroupedByHour, throughputsGroupedByHour


# plot out the stats over time, show the mean with
# two lines:
# 1. 'num of tests' for plotting out numTotalTests per hour
# 2. 'num positive tests' for number of positive tests per hour
def plotTestsPerHour(testsPerHour, fileDir, plotTitle):
    # set value to zero if no test from that hour
    for hour in range(0, 24):
        hourStr = str(hour)
        if hour < 10:
            hourStr = '0' + hourStr
        if hourStr not in testsPerHour:
            testsPerHour[hourStr] = [0, 0]

    testHours = sorted(testsPerHour.keys())
    x = []
    meanNumsTotalTests = []

    meansPositiveRatio = []

    for testHour in testHours:
        # num positive tests during this hour
        numPositiveTestsThisHour = testsPerHour[testHour][0]
        # total tests during this hour
        numTotalTestsThisHour = numPositiveTestsThisHour + testsPerHour[testHour][1]
        # positive ratio during this hour
        if not numTotalTestsThisHour:
            positiveRatioThisHour = 0
        else:
            positiveRatioThisHour = numPositiveTestsThisHour / float(numTotalTestsThisHour)
        meanNumsTotalTests.append(numTotalTestsThisHour)
        meansPositiveRatio.append(positiveRatioThisHour)
        hourFormatted = dt.strptime(testHour, '%H')
        x.append(hourFormatted)

    # plot num of tests
    fig, ax = plt.subplots()
    ax.errorbar(x, meanNumsTotalTests, fmt='-o', ecolor='lightgray', color='black',
                elinewidth=3, capsize=0)
    # ax.set_ylabel('Number of tests', color='#1f77b4')
    ax.set_ylabel('Number of tests')
    ax.set_xlabel('Time (Local to the Client)')
    # ax.tick_params('y', colors='#1f77b4')
    fig.autofmt_xdate()
    # plt.legend(loc='upper right')
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)
    # set font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    plotTitle = plotTitle[:40]
    plt.savefig(fileDir + '/' + plotTitle + '.png', bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close('all')

    # plot ratio of positive tests
    fig, ax = plt.subplots()
    ax.errorbar(x, meansPositiveRatio, fmt='-o', ecolor='lightgray', color='black',
                elinewidth=3, capsize=0)
    # ax.set_ylabel('Ratio of positive tests', color='#ff7f0e')
    ax.set_ylabel('Ratio of positive tests')
    ax.set_xlabel('Time (Local to the Client)')
    # ax.tick_params('y', colors='#ff7f0e')
    fig.autofmt_xdate()
    # plt.legend(loc='upper right')
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)
    # set font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    plt.savefig(fileDir + '/' + plotTitle + '_ratioPerHour.png', bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close('all')


def plotTestsPerState(testsPerState, overallPositiveRatio, fileDir, plotTitle):
    # sl state locations
    sl = []
    for i in range(8):
        for j in range(11):
            sl.append((0.08 * j, 0.08 * i))

    states = {'HI': sl[0], 'TX': sl[3], 'FL': sl[8],
              'OK': sl[14], 'LA': sl[15], 'MS': sl[16], 'AL': sl[17], 'GA': sl[18],
              'AZ': sl[23], 'NM': sl[24], 'KS': sl[25], 'AR': sl[26], 'TN': sl[27], 'NC': sl[28], 'SC': sl[29],
              'CA': sl[33], 'UT': sl[34], 'CO': sl[35], 'NE': sl[36], 'MO': sl[37], 'KY': sl[38], 'WV': sl[39],
              'VA': sl[40], 'MD': sl[41], 'DE': sl[42],
              'OR': sl[44], 'NV': sl[45], 'WY': sl[46], 'SD': sl[47], 'IA': sl[48], 'IN': sl[49], 'OH': sl[50],
              'PA': sl[51], 'NJ': sl[52], 'CT': sl[53], 'RI': sl[54],
              'WA': sl[55], 'ID': sl[56], 'MT': sl[57], 'ND': sl[58], 'MN': sl[59], 'IL': sl[60], 'MI': sl[61],
              'NY': sl[63], 'MA': sl[64],
              'WI': sl[71], 'VT': sl[75], 'NH': sl[76],
              'AK': sl[77], 'ME': sl[87]}

    fig, ax = plt.subplots(figsize=(11, 13))
    maxNumTests = 0
    maxPositiveRatio = 0
    minPositiveRatio = 1
    # find out the standard deviation of positive ratio
    allPositiveRatios = []
    for state in testsPerState:
        countTestState = testsPerState[state]
        numTests = countTestState[0] + countTestState[1]
        positiveRatio = float(countTestState[0]) / float(numTests)
        if numTests > maxNumTests:
            maxNumTests = numTests
        if positiveRatio > maxPositiveRatio:
            maxPositiveRatio = positiveRatio
        if positiveRatio < minPositiveRatio:
            minPositiveRatio = positiveRatio
        allPositiveRatios.append(positiveRatio)

    stdPositiveRatio = round(float(numpy.std(allPositiveRatios)), 2)

    # check whether the min or the max is more std away from national mean
    # the one that is further away takes the whole 50% of the color bar and sets the scale
    if not stdPositiveRatio:
        return
    if not maxNumTests:
        return

    # half of the bar translates to what color
    if (maxPositiveRatio - overallPositiveRatio) > (overallPositiveRatio - minPositiveRatio):
        numStdFromMedianColorToEnd = (maxPositiveRatio - overallPositiveRatio) / float(stdPositiveRatio)
        print('max', numStdFromMedianColorToEnd, maxPositiveRatio, overallPositiveRatio, stdPositiveRatio)
    else:
        numStdFromMedianColorToEnd = (overallPositiveRatio - minPositiveRatio) / float(stdPositiveRatio)
        print('min', numStdFromMedianColorToEnd, maxPositiveRatio, overallPositiveRatio, stdPositiveRatio)

    if not numStdFromMedianColorToEnd:
        return

    cmap = plt.get_cmap('coolwarm')

    maxNormalizedNumTests = 0
    minNormalizedNumTests = 1
    for state in states:
        h = 0.08
        x = states[state][0]
        y = states[state][1]
        ax.add_patch(Rectangle((x, y), width=h, height=h, alpha=0.2, color='gray', ec='w'))
        ax.text(x, y + 0.06, state, dict(size=13))
        if state not in testsPerState:
            continue
        countTestState = testsPerState[state]
        numTests = countTestState[0] + countTestState[1]
        normalizedNumTests = float(math.log(numTests)) / float(math.log(maxNumTests))
        positiveRatio = float(countTestState[0]) / float(numTests)

        if positiveRatio > overallPositiveRatio:
            colorAdjustment = ((positiveRatio - overallPositiveRatio) / float(
                stdPositiveRatio)) / numStdFromMedianColorToEnd * 0.5 + 0.5
        else:
            colorAdjustment = 0.5 - ((overallPositiveRatio - positiveRatio) / float(
                stdPositiveRatio)) / numStdFromMedianColorToEnd * 0.5

        if normalizedNumTests > maxNormalizedNumTests:
            maxNormalizedNumTests = normalizedNumTests
        elif normalizedNumTests < minNormalizedNumTests:
            minNormalizedNumTests = normalizedNumTests

        # circle size is normalized to the state with most tests
        ax.add_patch(
            Circle((x + 0.04, y + 0.04), 0.05 * normalizedNumTests, color=cmap(colorAdjustment),
                   ec="w"))

    if numpy.isnan(stdPositiveRatio):
        stdPositiveRatio = 0

    plt.text(0.20, -0.16, '1 Standard Deviation (SD) = {} %'.format(int(stdPositiveRatio * 100)), dict(size=20))
    # sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm'), norm=plt.Normalize(vmin=minRatio, vmax=maxRatio))
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm'))
    # fake up colorbar
    sm._A = []
    cbar = fig.colorbar(sm, ticks=[0.1, 0.3, 0.5, 0.7, 0.9], orientation='horizontal', pad=0.02, shrink=1)
    # get the correct tick value
    tenPercentTick = round(0.8 * numStdFromMedianColorToEnd, 1)
    thirtyPercentTick = round(0.4 * numStdFromMedianColorToEnd, 1)
    cbar.ax.set_xticklabels(['-{} SD'.format(tenPercentTick), '-{} SD'.format(thirtyPercentTick),
                             '{}%'.format(int(overallPositiveRatio * 100)), '+{} SD'.format(thirtyPercentTick),
                             '+{} SD'.format(tenPercentTick)])

    for item in cbar.ax.get_xticklabels():
        item.set_fontsize(20)
    plt.axis('off')
    plt.savefig(fileDir + '/' + plotTitle + '.png', bbox_inches='tight')


def checkVarianceInThrottling(testsPerGroup):
    minNumTests = 10
    throttlingRatios = []
    for group in testsPerGroup:
        # skip group with less than minNumTests tests
        numTotalTests = testsPerGroup[group][0] + testsPerGroup[group][1]
        if numTotalTests < minNumTests:
            continue

        throttlingRatios.append(testsPerGroup[group][0] / float(numTotalTests))

    stdRatiosAmongGroups = round(float(numpy.std(throttlingRatios)), 2)
    return stdRatiosAmongGroups


def autolabel(rects, ax):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h),
                ha='center', va='bottom')


def condenseSamples(throughputs, timestamps):
    condensedThroughputs = []
    condensedTimestamps = []
    previousThroughput = 0

    for i in range(len(throughputs)):
        if i % 2:
            previousThroughput = throughputs[i]
        else:
            condensedThroughputs.append((previousThroughput + throughputs[i]) / 2.0)
            condensedTimestamps.append(timestamps[i])

    return condensedThroughputs, condensedTimestamps


def condenseSamplesToSize(throughputs, timestamps, sampleSize):
    resultThroughputs = throughputs
    resultTimestamps = timestamps

    while len(resultThroughputs) > sampleSize:
        resultThroughputs, resultTimestamps = condenseSamples(resultThroughputs, resultTimestamps)

    return resultThroughputs, resultTimestamps


# merge the overlapping range(s)
def sanityCheckProminentValues(prominentIndexRanges, logProb):
    updatedProminentIndexRanges = []
    if not prominentIndexRanges:
        return [], []
    elif len(prominentIndexRanges) == 1:
        prominentIndexRange = prominentIndexRanges[0]
        return [prominentIndexRange[1]], [(prominentIndexRange[0], prominentIndexRange[2])]
    else:
        prominentIndexRanges.sort(key=lambda x: x[1])
        while prominentIndexRanges:
            prominentIndexRange = prominentIndexRanges.pop(0)
            if not updatedProminentIndexRanges:
                updatedProminentIndexRanges.append(prominentIndexRange)
            else:
                previousProminentIndexRange = updatedProminentIndexRanges.pop(-1)
                # if overlaping, combine them into one prominentIndex
                if previousProminentIndexRange[2] >= prominentIndexRange[0]:
                    indexMoreProminent = previousProminentIndexRange[1]
                    if logProb[prominentIndexRange[1]] > logProb[previousProminentIndexRange[1]]:
                        indexMoreProminent = prominentIndexRange[1]
                    updatedProminentIndexRanges.append(
                        (previousProminentIndexRange[0], indexMoreProminent, prominentIndexRange[2]))
                # else push both of them into the list
                else:
                    updatedProminentIndexRanges.append(previousProminentIndexRange)
                    updatedProminentIndexRanges.append(prominentIndexRange)
        indexLocalMaximums = []
        truePositiveRanges = []
        # sort by the density of the local maximum
        # indexLocalMaximums in ascending order of the density
        updatedProminentIndexRanges.sort(key=lambda x: logProb[x[1]])
        for indexRange in updatedProminentIndexRanges:
            indexLocalMaximums.append(indexRange[1])
            truePositiveRanges.append((indexRange[0], indexRange[2]))

        return indexLocalMaximums, truePositiveRanges


# return the most prominent values, the local maximums
# along with the ranges that are considered true positives (near local maximums with density higher than threshold)
def prominentValuesDetection(densities, samples, samplingLeft, samplingRight, threshold):
    prominentValuesWithRange = []
    samplesList = list(samples.reshape(1, -1)[0])
    indexLocalMaximums = argrelextrema(densities, numpy.greater)[0]
    indexLocalMaximums = list(indexLocalMaximums)
    averagedDensity = 1.0 / (samplingRight - samplingLeft)
    # If samples are not extremely closed to each other, the prominent values have to be one magnitude higher
    # than the average density value
    densityThreshold = averagedDensity
    if threshold:
        densityThreshold = threshold
        # densityThreshold = math.log(averagedDensity) + magnitude

    # sorting the local maximums that are above the densityThreshold based on the density values
    indexLocalMaximums = [x for x in indexLocalMaximums if densities[x] > densityThreshold]
    indexLocalMaximums.sort(key=lambda x: densities[x])

    # get the true positive range(s)
    if indexLocalMaximums:
        for localMaximum in indexLocalMaximums:
            leftIndex = localMaximum
            rightIndex = localMaximum + 1
            # left boundary
            while (leftIndex > -1) and (densities[leftIndex] > densityThreshold):
                leftIndex -= 1
            # right boundary
            while (rightIndex < len(densities) - 1) and (densities[rightIndex] > densityThreshold):
                rightIndex += 1
            prominentValuesWithRange.append((leftIndex, localMaximum, rightIndex))

        # there might be overlapping regions
        indexLocalMaximums, indexTruePositiveRanges = sanityCheckProminentValues(prominentValuesWithRange, densities)

        truePositiveRanges = []
        for indexRange in indexTruePositiveRanges:
            truePositiveRanges.append((round(samplesList[indexRange[0]], 2), round(samplesList[indexRange[1]], 2)))

        allLocalMaxValues = list(samples[indexLocalMaximums].reshape(1, -1)[0])
        allLocalMaxValues = [round(x, 2) for x in allLocalMaxValues]
        allLocalMaxDensities = list(densities[indexLocalMaximums])
        allLocalMaxDensities = [round(x, 2) for x in allLocalMaxDensities]

        return allLocalMaxValues, allLocalMaxDensities, truePositiveRanges
    else:
        return [], [], []


def meanAfterBoosting(throughputSamples, timestamps, boostingBytes, boostingTime):
    # sanity check, needs at least 50 samples
    if len(timestamps) < 50:
        return None, None
    samplingInterval = timestamps[1] - timestamps[0]
    throughputSamplesAfterBoost = []
    timestampsAfterBoost = []
    # if byte based boosting
    if boostingBytes:
        # use the most prominent boost byte value
        boostingByte = boostingBytes[-1]
        currentBytes = 0
        for throughputSample in throughputSamples:
            if currentBytes > boostingByte:
                throughputSamplesAfterBoost.append(throughputSample)
            else:
                currentBytes += throughputSample * samplingInterval / 8
    # else no boosting detected
    else:
        throughputSamplesAfterBoost = throughputSamples

    # if almost no sample after boost
    if len(throughputSamplesAfterBoost) < 10:
        return None, None
    else:
        return float(numpy.mean(throughputSamplesAfterBoost)), throughputSamplesAfterBoost


# count how many bytes have being transmitted so far
def countMB(throughputSamples, timeStamps, index):
    if not timeStamps:
        return 0
    interval = timeStamps[1] - timeStamps[0]
    currentMB = 0
    for i in range(index):
        currentMB += throughputSamples[i] * interval / 8

    return currentMB


def samplesAfterBytes(throughputSamples, timeStamps, thresholdBytes):
    if not throughputSamples:
        return [], []
    resultThroughputSamples = []
    resultTimeStamps = []
    afterByte = False
    timestampsThreshold = timeStamps[0]
    for index in range(len(throughputSamples)):
        currMB = countMB(throughputSamples, timeStamps, index)
        if currMB >= thresholdBytes:
            if not afterByte:
                timestampsThreshold = timeStamps[index]
                afterByte = True
            resultThroughputSamples.append(throughputSamples[index])
            resultTimeStamps.append(timeStamps[index] - timestampsThreshold)
    return resultThroughputSamples, resultTimeStamps


# make sure the range is at least greater than 0.5
def rationalizeTruePositiveRanges(truePositiveRanges):
    rationalizedTruePositiveRanges = []
    for truePositiveRange in truePositiveRanges:
        if (truePositiveRange[1] - truePositiveRange[0]) < 0.5:
            meanRate = (truePositiveRange[0] + truePositiveRange[1]) / 2.0
            rationalizedTruePositiveRange = (meanRate - 0.25, meanRate + 0.25)
            rationalizedTruePositiveRanges.append(rationalizedTruePositiveRange)
        else:
            rationalizedTruePositiveRanges.append(truePositiveRange)

    return rationalizedTruePositiveRanges


# Check whether there is Diff detected based on the stats
def checkDiff(list1, list2, numTotalTests=1):
    pThreshold = 0.05
    differentiation = False
    ks2dVal, ks2pVal = ks_2samp(list1, list2)
    # If ks2pVal > 1 - alpha, then we check how many samples are greater than 1 - alpha
    # Elif ks2pVal < 1 - alpha, then we check how many samples are smaller than 1 - alpha
    greater = True
    if ks2pVal < pThreshold:
        greater = False
    acceptRatio = sampleKS2(list1, list2, greater=greater, pThreshold=pThreshold)
    baselineThroughput = min(max(list1), max(list2))
    avg1 = numpy.mean(list1)
    avg2 = numpy.mean(list2)
    areaDiff = (avg2 - avg1) / float(baselineThroughput)
    if acceptRatio > 0.95 and ks2pVal < float(pThreshold) / float(numTotalTests) and abs(areaDiff) > 0.1:
        differentiation = True

    return differentiation


def sampleKS2(list1, list2, greater=True, pThreshold=0.05, sub=0.5, r=100):
    '''
    Taken from NetPolice paper:

    This function uses Jackknife, a commonly-used non-parametric re-sampling method,
    to verify the validity of the K-S test statistic. The idea is to randomly select
    half of the samples from the two original input sets and apply the K-S test on
    the two new subsets of samples. This process is repeated r times. If the results
    of over B% of the r new K-S tests are the same as that of the original test, we
    conclude that the original K-S test statistic is valid.
    '''

    results = []
    accept = 0.0

    for i in range(r):
        sub1 = random.sample(list1, int(len(list1) * sub))
        sub2 = random.sample(list2, int(len(list2) * sub))
        res = ks_2samp(sub1, sub2)
        results.append(res)

        pVal = res[1]
        if greater:
            if pVal > pThreshold:
                accept += 1
        else:
            if pVal < pThreshold:
                accept += 1
    # dVal_avg = numpy.average([x[0] for x in results])
    # pVal_avg = numpy.average([x[1] for x in results])

    return accept / r


# remove the top percentage% data from values
# as they might be outliers and skew the plot
def removeHighest(values, percentage):
    values.sort()
    numSamples = len(values)
    values = values[: int((100 - percentage) * numSamples / 100)]
    return values


def getRange(dataSamples):
    sampleStartThreshold = -1

    # ignore highest 1%
    dataSamples.sort()
    dataSamples = dataSamples[: int(99 / 100 * len(dataSamples))]

    maxSample = max(dataSamples)
    minSample = min(dataSamples)

    # spaning the sampling period 1.5 times on each direction
    # when the sampling interval is small
    midSample = (maxSample + minSample) / 2
    extendedSamplingPeriod = (maxSample - midSample) * 1.5
    samplingLeft = midSample - extendedSamplingPeriod
    samplingRight = midSample + extendedSamplingPeriod

    # if sampling interval extremely small (i.e., censoring)
    # make it at least 2
    if (samplingRight - samplingLeft) < 1:
        samplingLeft -= 1
        samplingRight += 1

    if samplingLeft < sampleStartThreshold:
        samplingLeft = sampleStartThreshold

    return samplingLeft, samplingRight


def getQuaterFromMonth(month):
    quarter = 1
    if 1 <= month < 4:
        quarter = 1
    elif 4 <= month < 7:
        quarter = 2
    elif 7 <= month < 10:
        quarter = 3
    elif 10 <= month:
        quarter = 4

    return quarter


'''
For each ISP, create 3 subdirectories: timeAnalysis, testPlots, individualPlots

Create plots in the 3 subdir accordingly, tests can be aggregated per month/week

timeAnalysis: three graphs for the ISP (default: all replays combined)
              1. number of tests per hour (only if we have test cases that cover the entire 24 hours)
              2. ratio of positive tests per hour
              3. number of (negative and positive) tests each day

diffAnalysis: two sets of plots for each replay
           1. average throughputs for the positive tests
           2. average throughputs for the negative tests, with a line identifying the detected throttling rate

individualPlots for each test(a. CDF with random/original b. throughput/time) 
          (limited to 100 plots in each category):
          1. positive tests
          2. positive tests special cases (average throughput outside of throttling ranges)
          3. negative tests that have stat (average throughput) above detected throttling rate (i.e., true negatives)
          4. negative tests special cases (average throughput within throttling ranges)
'''


class singleISPstats(object):
    def __init__(self, ISPdir, aggregation=None):
        self.carrierName = ISPdir.split('/')[-1]
        self.carrierName = ''.join(e for e in self.carrierName if e.isalnum())
        self.bandwidthThreshold = 0.1
        self.delayedThrottlingDetectionPenalty = 20
        # at most 50 plots for each category (e.g., positive/negative)
        self.individualPlotCnt = 50
        # the values detected as with high densities
        # only if they have data points more than this threshold
        # i.e., more than aggPercentage % of the data has value X
        self.aggPercentage = 2
        self.ISPdir = ISPdir
        self.timeDir = ISPdir + '/timeAnalysis/'
        self.diffDir = ISPdir + '/diffAnalysis/'
        self.boostDir = self.diffDir + 'boost/'
        self.indTestsDir = ISPdir + '/individualTests/'
        self.indTestsTruePositiveDir = self.indTestsDir + 'TruePositive/'
        self.indTestsFalsePositiveDir = self.indTestsDir + 'FalsePositive/'
        self.indTestsTrueNegativeDir = self.indTestsDir + 'TrueNegative/'
        self.indTestsFalseNegativeDir = self.indTestsDir + 'FalseNegative/'
        self.plotDataDir = ISPdir + '/plotData/'
        # each of the dictionaries stores data for each aggregation group and replay
        # self.throttlingStats = {'month (2018, 02)': {'Youtube': {'Throttling rates': [],
        #                                             'KDE values for throttling rates': [],
        #                                             'Detected Boosting Bytes': [],
        #                                             'KDE values for boosting bytes': [],
        #                                             'Detected Boosting Time': [],
        #                                             'KDE values for boosting time': [],},
        #                                              'Netflix': {}, ...}
        #                         'month (2018, 03)': {}, ... }'
        self.throttlingStats = {}
        # the list of tests for each aggregation group and replay
        self.allTests = {}
        # the list of positive and negative tests
        self.positiveTests = {}
        self.negativeTests = {}
        # the list of true positive tests
        self.truePositiveTests = {}
        self.creatDir()
        self.loadJson(ISPdir, aggregation)

    # create the parent directories
    def creatDir(self):
        for plots_dir in [self.timeDir, self.diffDir, self.boostDir, self.indTestsDir, self.plotDataDir,
                          self.indTestsTruePositiveDir, self.indTestsFalsePositiveDir,
                          self.indTestsTrueNegativeDir, self.indTestsFalseNegativeDir]:
            if not os.path.exists(plots_dir):
                os.mkdir(plots_dir)

    # every positive test is considered true positive at the beginning
    def loadJson(self, ISPdir, aggregation):
        # each file contains tests for the particular replay
        # can combine different replays here
        for file in os.listdir(ISPdir):
            # skip the plots/dirs
            if 'json' not in file:
                continue

            filePath = ISPdir + '/' + file
            fileName = file.split('.')[0]
            allTests = json.load(open(filePath, 'r'))

            startDate = '2018-01-17'
            allTests = self.filterTests(allTests, startDate)
            replayName = fileName.split(')_')[1]
            if not aggregation:
                if replayName not in self.allTests:
                    self.allTests[replayName] = {}
                self.allTests[replayName]['all'] = allTests
            elif aggregation in ['month', 'week', 'half', 'quarter']:
                self.aggregateTests(allTests, replayName, aggregation)
            else:
                print('\r\n wrong aggregation', self.carrierName, aggregation)
                sys.exit(-1)
        if aggregation:
            self.dumpAggregatedTests()

    # If the tests are separated into different aggregation groups (e.g., quarters)
    # dump the tests from different aggregation groups into separated json files
    # e.g., carrier_replay_(2018, 1 quarter).json
    def dumpAggregatedTests(self):
        for replayName in self.allTests:
            for aggregationGroup in self.allTests[replayName]:
                json.dump(self.allTests[replayName][aggregationGroup],
                          open('{}/{}_{}_{}.json'.format(self.ISPdir, self.carrierName, replayName, aggregationGroup), 'w'))

    # filter out the tests before the startDate
    def filterTests(self, allTests, startDate):
        allTestsAfterFiltering = []
        for test in allTests:
            incomingDate = test['timestamp'].split(' ')[0]
            if incomingDate > startDate:
                allTestsAfterFiltering.append(copy.deepcopy(test))

        return allTestsAfterFiltering

    # aggregating tests into months or weeks
    # aggregatedTests = {'Youtube'' : {'2018,02 month':  [{test_month_(2018, 02)_1}, ...], '2018,03 month', ...},
    #                    'Netflix' : {}, ...}
    def aggregateTests(self, allTests, replayName, aggregation):
        # for each test, adding two additional fields for aggregation, month and week
        for test in allTests:
            incomingTime = test['geoInfo']['localTime']
            if not incomingTime:
                incomingTime = test['timestamp']
            incomingYMD = incomingTime.split(' ')[0]
            incomingYear = int(incomingYMD.split('-')[0])
            incomingMonth = int(incomingYMD.split('-')[1])
            incomingDate = int(incomingYMD.split('-')[2])
            midYear = dt.strptime('2018,07', '%Y,%m').date()
            if aggregation == 'quarter':
                incomingQuarter = getQuaterFromMonth(incomingMonth)
                aggregationGroup = '{},{} quarter'.format(incomingYear, incomingQuarter)
            elif aggregation == 'week':
                isoWeek = datetime.date(incomingYear, incomingMonth, incomingDate).isocalendar()
                aggregationGroup = '{},{} week'.format(isoWeek[0], isoWeek[1])
            elif aggregation == 'month':
                aggregationGroup = '{},{} month'.format(incomingYear, incomingMonth)
            else:
                yearMonth = dt.strptime('{},{}'.format(incomingYear, incomingMonth), '%Y,%m').date()
                if yearMonth < midYear:
                    aggregationGroup = 'Before Jul 2018'
                else:
                    aggregationGroup = 'After Jul 2018'

            # first test for this replay in this ISP
            if replayName not in self.allTests:
                self.allTests[replayName] = {
                    aggregationGroup: []}

            # first test for this aggregation group
            if aggregationGroup not in self.allTests[replayName]:
                self.allTests[replayName][aggregationGroup] = []

            self.allTests[replayName][aggregationGroup].append(test)

    def performAnalysis(self, replayName, aggregationGroup):
        self.separatePositiveTests(replayName, aggregationGroup)
        # if there exists positive tests
        if self.positiveTests[replayName][aggregationGroup]:
            # detect whether fixed rate throttling
            delayedBytes, delayedTime = self.getDelayedThrottlingStat(replayName, aggregationGroup)
            self.detectFixRateThrottling(replayName, aggregationGroup, delayedBytes, delayedTime)

    def countTests(self, replayName, aggregationGroup):
        self.separatePositiveTests(replayName, aggregationGroup)
        return len(self.allTests[replayName][aggregationGroup]), len(self.positiveTests[replayName][aggregationGroup])

    # Include a test if it has a original/random replay pair
    # Separate tests into positive and negative based on individual differentiation result with Bonferroni correction
    def separatePositiveTests(self, replayName, aggregationGroup):

        numTests = len(self.allTests[replayName][aggregationGroup])
        positiveTests = []
        negativeTests = []

        for oneTest in self.allTests[replayName][aggregationGroup]:
            originalXputs = oneTest['original_xPuts'][0]
            randomXputs = oneTest['random_xPuts'][0]
            # Bonferroni correction
            if checkDiff(originalXputs, randomXputs, numTests):
                positiveTests.append(copy.deepcopy(oneTest))
            else:
                negativeTests.append(copy.deepcopy(oneTest))

        if replayName not in self.positiveTests:
            self.positiveTests[replayName] = {}
            self.negativeTests[replayName] = {}
        # add the positive test to positiveTests
        self.positiveTests[replayName][aggregationGroup] = positiveTests
        self.negativeTests[replayName][aggregationGroup] = negativeTests
        print('\r\n ', replayName, aggregationGroup, len(positiveTests), len(negativeTests))

        return

    def getAveThroughputsAfterBoost(self, replayName, aggregationGroup, boostBytes, boostTime):
        positiveAveThroughputsOriginal = []
        positiveAveThroughputsRandom = []
        negativeAveThroughputsOriginal = []
        negativeAveThroughputsRandom = []

        if replayName == 'Netflix' and (not boostBytes):
            # For Netflix , sampling only after the first 2 MB since the Netflix replay starts slow
            boostBytes = [2]

        # get the average throughputs after boosting period if boosting is detected
        for positiveTest in self.positiveTests[replayName][aggregationGroup]:
            meanAfterBoostOriginal, samplesAfterBoostOriginal = meanAfterBoosting(positiveTest['original_xPuts'][0],
                                                                                  positiveTest['original_xPuts'][1],
                                                                                  boostBytes,
                                                                                  boostTime)
            meanAfterBoostRandom, samplesAfterBoostRandom = meanAfterBoosting(positiveTest['random_xPuts'][0],
                                                                              positiveTest['random_xPuts'][1],
                                                                              boostBytes,
                                                                              boostTime)
            if meanAfterBoostOriginal:
                positiveAveThroughputsOriginal.append(meanAfterBoostOriginal)
                # if not enough sample after boosting period for the random replay
                if meanAfterBoostRandom:
                    positiveAveThroughputsRandom.append(meanAfterBoostRandom)
                else:
                    positiveAveThroughputsRandom.append(numpy.mean(positiveTest['random_xPuts'][0]))

        for negativeTest in self.negativeTests[replayName][aggregationGroup]:
            negativeAveThroughputsOriginal.append(float(numpy.mean(negativeTest['original_xPuts'][0])))
            negativeAveThroughputsRandom.append(float(numpy.mean(negativeTest['random_xPuts'][0])))

        return positiveAveThroughputsOriginal, positiveAveThroughputsRandom, negativeAveThroughputsOriginal, negativeAveThroughputsRandom

    def getThroughputsAfterBoost(self, replayName, aggregationGroup, boostBytes, boostTime):
        positiveThroughputsOriginal = []
        positiveThroughputsRandom = []
        negativeThroughputsOriginal = []
        negativeThroughputsRandom = []

        if replayName == 'Netflix' and (not boostBytes):
            # For Netflix , sampling only after the first 2 MB since the Netflix replay starts slow
            boostBytes = [2]

        # get the average throughputs after boosting period if boosting is detected
        for positiveTest in self.positiveTests[replayName][aggregationGroup]:
            meanAfterBoostOriginal, samplesAfterBoostOriginal = meanAfterBoosting(positiveTest['original_xPuts'][0],
                                                                                  positiveTest['original_xPuts'][1],
                                                                                  boostBytes,
                                                                                  boostTime)
            meanAfterBoostRandom, samplesAfterBoostRandom = meanAfterBoosting(positiveTest['random_xPuts'][0],
                                                                              positiveTest['random_xPuts'][1],
                                                                              boostBytes,
                                                                              boostTime)
            if meanAfterBoostOriginal:
                positiveThroughputsOriginal += samplesAfterBoostOriginal
                # if not enough sample after boosting period for the random replay
                if meanAfterBoostRandom:
                    positiveThroughputsRandom += samplesAfterBoostRandom
                else:
                    positiveThroughputsRandom += positiveTest['random_xPuts'][0]

        for negativeTest in self.negativeTests[replayName][aggregationGroup]:
            negativeThroughputsOriginal += negativeTest['original_xPuts'][0]
            negativeThroughputsRandom += negativeTest['random_xPuts'][0]

        return positiveThroughputsOriginal, positiveThroughputsRandom, negativeThroughputsOriginal, negativeThroughputsRandom

    def plotIndividualPositiveTests(self, replayName, aggregationGroup, throttlingRatesDetected, boostBytes, boostTime):
        if not boostBytes:
            boostBytes = [0]
        elif replayName == 'Netflix' and (not boostBytes):
            # For Netflix , sampling only after the first 2 MB since the Netflix replay starts slow
            boostBytes = [2]
        cntPlot = 0
        # plot individual true positive test
        for positiveTest in self.truePositiveTests[replayName][aggregationGroup]:
            # plot positive if true positive
            if cntPlot < self.individualPlotCnt:
                xPutsOriginal = positiveTest['original_xPuts'][0]
                timestampsOriginal = positiveTest['original_xPuts'][1]
                xPutsRandom = positiveTest['random_xPuts'][0]
                timestampsRandom = positiveTest['random_xPuts'][1]
                xPutsOriginal, timestampsOriginal = samplesAfterBytes(
                    xPutsOriginal, timestampsOriginal, boostBytes[-1])
                xPutsRandom, timestampsRandom = samplesAfterBytes(
                    xPutsRandom, timestampsRandom, boostBytes[-1])

                plotIndividualTest(timestampsOriginal, xPutsOriginal,
                                   timestampsRandom, xPutsRandom,
                                   throttlingRatesDetected, self.indTestsTruePositiveDir + replayName + '/',
                                   '{}_{}_{}_{}'.format(replayName, self.carrierName, replayName,
                                                        positiveTest['uniqueTestID']))
                cntPlot += 1

    # 1. The distributions of 1. average throughputs-random and 2. average throughputs-original are different
    def checkThrottling(self, positiveAveThroughputsOriginal, positiveAveThroughputsRandom,
                        negativeAvgThroughputsOriginal, negativeAvgThroughputsRandom):

        allAvgThroughputsOriginal = positiveAveThroughputsOriginal + negativeAvgThroughputsOriginal
        allAvgThroughputsRandom = positiveAveThroughputsRandom + negativeAvgThroughputsRandom

        # check whether diff
        ks2dVal, ks2pValOriginalRandom = ks_2samp(allAvgThroughputsOriginal, allAvgThroughputsRandom)
        # If ks2pVal > 1 - alpha, then we check how many samples are greater than 1 - alpha
        # Elif ks2pVal < 1 - alpha, then we check how many samples are smaller than 1 - alpha
        greater = True
        pThreshold = 0.05
        if ks2pValOriginalRandom < pThreshold:
            greater = False

        acceptRatio = sampleKS2(allAvgThroughputsOriginal, allAvgThroughputsRandom, greater=greater,
                                pThreshold=pThreshold)

        diffDetected = False

        if ks2pValOriginalRandom < pThreshold and acceptRatio > 0.95:
            diffDetected = True

        return diffDetected, ks2pValOriginalRandom, acceptRatio

    def getDelayedThrottlingStat(self, replayName, aggregationGroup):
        delayedBytes = []
        delayedTime = []
        # TODO, change point detection takes a long time, skipping it for carriers other than TMobile for now
        # as we checked before only TMUS has this behavior
        if 'TMobile' in self.carrierName:
            delayedBytes, delayedTime = self.delayedThrottlingDetection(
                replayName, aggregationGroup)

        return delayedBytes, delayedTime

    # first construct a list of samples that are from a man-made distribution
    # the distribution is as follows:
    # aggPercentage% of the values are at the middle (i.e., (samplingRight - samplingLeft)/ 2 ) of the spectrum
    # the other (1 - aggPercentage)% are uniformly distributed in [samplingLeft, samplingRight]
    # return the density of the median value (i.e., with aggPercentage% data concentrated)
    # optional:
    # numValues set how many number of values will have aggPercentage% data aggregated,
    # for example, if numValues = 2, there will be two identical spikes, each with aggPercentage% data
    def findDensityThreshold(self, numSamples, samplingLeft, samplingRight, fitBandwidth, plotTitle='', aggPercentage=5,
                             numValues=1):
        randomSamples = []
        aggNumber = int(aggPercentage * numSamples / 100)
        # 1 - aggPercentage % * numValues of the data uniformly distributed on the range
        for i in range(numSamples - aggNumber * numValues):
            randomSamples.append(random.uniform(samplingLeft, samplingRight))
        # aggPercentage % right in the middle
        aggValue = numValues + 1
        count = 1
        for i in range(numValues):
            for i in range(aggNumber):
                randomSamples.append((samplingRight - samplingLeft) * count / aggValue)
            count += 1

        samplingLeft, samplingRight = getRange(randomSamples)

        allProminentValues, allProminentDensities, truePositiveRanges, ylim, y2lim = self.KDEtest(randomSamples,
                                                                                                  samplingLeft,
                                                                                                  samplingRight,
                                                                                                  fitBandwidth,
                                                                                                  plotTitle=plotTitle)

        return allProminentDensities[-numValues:]

    def detectValueWithHighDensity(self, allAvgThroughputsOriginal, allAvgThroughputsRandom, ylim=None, y2lim=None,
                                   plotTitle=''):

        # Only proceed if they are from different distributions
        plotTitle = '{}_{}%_'.format(plotTitle, self.aggPercentage)
        samplingLeft, samplingRight = getRange(allAvgThroughputsOriginal)
        fitBandwidth = 0.1

        # number of thresholds returned = numValues
        # numValues specifies how many values with high density in this distribution
        densityThresholds = self.findDensityThreshold(len(allAvgThroughputsOriginal), samplingLeft, samplingRight,
                                                      fitBandwidth,
                                                      aggPercentage=self.aggPercentage,
                                                      plotTitle=plotTitle + '_threshold_')

        densityThreshold = densityThresholds[-1]

        allProminentValuesOriginal, allProminentDensitiesOriginal, truePositiveRangesOriginal, ylimOriginal, y2limOriginal = self.KDEtest(
            allAvgThroughputsOriginal, samplingLeft, samplingRight, fitBandwidth, xlabel='Average throughput (Mbps)',
            ylim=ylim, y2lim=y2lim,
            threshold=densityThreshold,
            plotTitle=plotTitle + '_original_')

        allProminentValuesRandom, allProminentDensitiesRandom, truePositiveRangesRandom, ylimRandom, y2limRandom = self.KDEtest(
            allAvgThroughputsRandom,
            samplingLeft,
            samplingRight,
            fitBandwidth, ylim=ylimOriginal, y2lim=y2limOriginal,
            xlabel='Average throughput (Mbps)',
            threshold=densityThreshold,
            plotTitle=plotTitle + '_random_')

        # find the value that is unique in original:
        # if there is overlap with the prominent value in random, discard
        uniqueProminentValuesOriginal = []
        uniqueProminentPositiveRangesOriginal = []
        for resultIndex in range(len(allProminentValuesOriginal)):
            unique = True
            for truePositiveRangeRandom in truePositiveRangesRandom:
                if truePositiveRangeRandom[0] < allProminentValuesOriginal[resultIndex] < truePositiveRangeRandom[1]:
                    unique = False
            if unique:
                uniqueProminentValuesOriginal.append(allProminentValuesOriginal[resultIndex])
                uniqueProminentPositiveRangesOriginal.append(truePositiveRangesOriginal[resultIndex])

        return uniqueProminentValuesOriginal, uniqueProminentPositiveRangesOriginal, ylimOriginal, y2limOriginal

    # the true positives, i.e., throttled tests are defined as the ones 1. with differentiation detected individually
    # and 2. the average throughput is limited to one of the detected throttling rates
    def separateTruePositiveTests(self, replayName, aggregationGroup, throttlingTruePositiveRanges, boostBytes,
                                  boostTime):
        if replayName == 'Netflix' and (not boostBytes):
            # For Netflix , sampling only after the first 2 MB since the Netflix replay starts slow
            boostBytes = [2]
        for posTest in self.positiveTests[replayName][aggregationGroup]:
            meanAfterBoostOriginal, samplesAfterBoostOriginal = meanAfterBoosting(posTest['original_xPuts'][0],
                                                                                  posTest['original_xPuts'][1],
                                                                                  boostBytes,
                                                                                  boostTime)
            truePositive = False
            if not meanAfterBoostOriginal:
                continue
            for throttlingRangeDetected in throttlingTruePositiveRanges:
                if throttlingRangeDetected[0] <= meanAfterBoostOriginal <= throttlingRangeDetected[1]:
                    truePositive = True

            if truePositive:
                if replayName not in self.truePositiveTests:
                    self.truePositiveTests[replayName] = {}
                if aggregationGroup not in self.truePositiveTests[replayName]:
                    self.truePositiveTests[replayName][aggregationGroup] = []
                self.truePositiveTests[replayName][aggregationGroup].append(copy.deepcopy(posTest))

    # return [] if no fixed rate throttling is detected
    def detectFixRateThrottling(self, replayName, aggregationGroup, boostBytes, boostTime):
        # get the original/random average throughputs (after boosting period for positive ones) for *all* tests
        positiveAveThroughputsOriginal, positiveAveThroughputsRandom, negativeAvgThroughputsOriginal, negativeAvgThroughputsRandom = self.getAveThroughputsAfterBoost(
            replayName, aggregationGroup, boostBytes, boostTime)

        # get *all* original/random throughput samples (after boosting for positive ones) from *all* tests
        # positiveAveThroughputsOriginal, positiveAveThroughputsRandom, negativeAvgThroughputsOriginal, negativeAvgThroughputsRandom = self.getThroughputsAfterBoost(
        #     replayName, aggregationGroup, boostBytes, boostTime)

        differenceInPolulation = False
        ks2pValOriginalRandom = None
        acceptRatio = None
        # determine whether throttling happens only if there are at least 10 positive tests
        if len(positiveAveThroughputsOriginal) > 10:
            # Step 1, distribution difference in the population
            differenceInPolulation, ks2pValOriginalRandom, acceptRatio = self.checkThrottling(
                positiveAveThroughputsOriginal, positiveAveThroughputsRandom,
                negativeAvgThroughputsOriginal, negativeAvgThroughputsRandom)

        throttlingRatesAll = []
        throttlingRatesDetected = []
        throttlingTruePositiveRanges = []
        # Step 2, check whether there is unique rate(s) with high density for the original replay but not random replay
        # for all tests
        ylimAll = None
        y2limAll = None
        if differenceInPolulation:
            json.dump((positiveAveThroughputsOriginal + negativeAvgThroughputsOriginal,
                       positiveAveThroughputsRandom + negativeAvgThroughputsRandom),
                      open('{}/{}_{}_{}_aveXputsAllTests.json'.format(self.plotDataDir, self.carrierName, replayName,
                                                                      aggregationGroup), 'w'))
            throttlingRatesAll, positiveRangesAll, ylimAll, y2limAll = self.detectValueWithHighDensity(
                positiveAveThroughputsOriginal + negativeAvgThroughputsOriginal,
                positiveAveThroughputsRandom + negativeAvgThroughputsRandom,
                plotTitle='{}_{}_{}_allTests'.format(self.carrierName, replayName, aggregationGroup))
            # print('\r\n step 2', self.carrierName, replayName, aggregationGroup, throttlingRatesAll)

        # Step 3, check whether there is rate(s) with high density detected both with positive tests and all tests
        if throttlingRatesAll:
            json.dump((positiveAveThroughputsOriginal, positiveAveThroughputsRandom),
                      open('{}/{}_{}_{}_aveXputsPosTests.json'.format(self.plotDataDir, self.carrierName, replayName,
                                                                      aggregationGroup), 'w'))
            throttlingRatesPos, positiveRangesPos, ylim, y2lim = self.detectValueWithHighDensity(
                positiveAveThroughputsOriginal, positiveAveThroughputsRandom, ylim=ylimAll, y2lim=y2limAll,
                plotTitle='{}_{}_{}_posTests'.format(self.carrierName, replayName, aggregationGroup))

            for throttlingRateAll in throttlingRatesAll:
                for positiveRangePos in positiveRangesPos:
                    if positiveRangePos[0] < throttlingRateAll < positiveRangePos[1]:
                        throttlingTruePositiveRanges.append(positiveRangePos)
                        throttlingRatesDetected.append(throttlingRateAll)

            # print('\r\n step 3', self.carrierName, replayName, aggregationGroup, throttlingRatesDetected)

        # plotting out the distributions if fixed rate throttling is detected
        if throttlingRatesDetected:
            aveThroughputsOriginal = removeHighest(positiveAveThroughputsOriginal + negativeAvgThroughputsOriginal, 0.5)
            aveThroughputsRandom = removeHighest(positiveAveThroughputsRandom + negativeAvgThroughputsRandom, 0.5)
            avgXputsOriginalSorted = sorted(aveThroughputsOriginal)
            avgXputsRandomSorted = sorted(aveThroughputsRandom)
            # limit the max x to be 15
            avgXputsOriginalSorted = [x for x in avgXputsOriginalSorted if x < 15]
            avgXputsRandomSorted = [x for x in avgXputsRandomSorted if x < 15]

            # Average throughput CDFs for comparing original and random replays
            Xoriginal, Yoriginal = list2CDF(avgXputsOriginalSorted)
            Xrandom, Yrandom = list2CDF(avgXputsRandomSorted)

            twoCDFs = {'Original replay': (Xoriginal, Yoriginal),
                       'Bit-inverted replay': (Xrandom, Yrandom)}

            json.dump((twoCDFs, throttlingRatesDetected),
                      open('{}/{}_{}_{}_CDFThrottlingRates.json'.format(self.plotDataDir, self.carrierName, replayName,
                                                                        aggregationGroup), 'w'))

            plotCDF(twoCDFs, throttlingRatesDetected, self.diffDir,
                    '{}_{}_{}_ks_{}_ar_{}_allTests_'.format(self.carrierName, replayName, aggregationGroup,
                                                            ks2pValOriginalRandom, acceptRatio),
                    xlabel='Average throughput (Mbps)')

            self.separateTruePositiveTests(replayName, aggregationGroup, throttlingTruePositiveRanges, boostBytes,
                                           boostTime)
            self.plotIndividualPositiveTests(replayName, aggregationGroup, throttlingRatesDetected, boostBytes,
                                             boostTime)

            if replayName not in self.throttlingStats:
                self.throttlingStats[replayName] = {}
            self.throttlingStats[replayName][aggregationGroup] = {'Throttling rates': throttlingRatesDetected,
                                                                  'Throttling true positive ranges': throttlingTruePositiveRanges,
                                                                  'Detected Boosting Bytes': boostBytes,
                                                                  'Number of positive tests': len(
                                                                      self.positiveTests[replayName][
                                                                          aggregationGroup]),
                                                                  'Number of total tests': len(
                                                                      self.allTests[replayName][
                                                                          aggregationGroup]),
                                                                  'Number of true positive tests': len(
                                                                      self.truePositiveTests[replayName][
                                                                          aggregationGroup]),
                                                                  'KS p value': ks2pValOriginalRandom,
                                                                  'KS accept ratio': acceptRatio}

        return throttlingRatesDetected

    # Detect delayed throttling/delayed bytes/time
    # The assumption is that boosting based on either attribute is possible
    # Return the delayed bytes/time values determined by KDE
    def delayedThrottlingDetection(self, replayName, aggregationGroup):
        plotTitle = 'delayed_{}_{}_{}'.format(self.carrierName, replayName, aggregationGroup)
        allThroughputsOriginal = []
        allTimestampsOriginal = []

        # detect boosting period (if any) for the tests
        for oneTest in self.positiveTests[replayName][aggregationGroup]:
            allThroughputsOriginal.append(oneTest['original_xPuts'][0])
            allTimestampsOriginal.append(oneTest['original_xPuts'][1])

        # counting the results, used for determining whether a plot is needed
        cntResult = {'Delayed': 0, 'NoDelay': 0, 'Error': 0}
        delayedBytesAggregated = []
        delayedTimeAggregated = []
        avgThroughputsDelay = []
        avgThroughputsAfterDelay = []

        for index in range(len(allThroughputsOriginal)):
            throughputs = allThroughputsOriginal[index]
            timestamps = allTimestampsOriginal[index]
            throughputs, timestamps = condenseSamplesToSize(throughputs, timestamps, 500)
            throughputsSignal = numpy.array(throughputs)
            try:
                algo = rpt.Pelt(model='rbf').fit(throughputsSignal)
                detectionResult = algo.predict(pen=self.delayedThrottlingDetectionPenalty)
            except Exception as e:
                print('\r\n exception in boosting detection', e)
                continue
            # one change point is detected
            if len(detectionResult) == 2:
                MB = countMB(throughputs, timestamps, detectionResult[0])
                boostTime = timestamps[detectionResult[0]]
                avgThroughputBoost = float(numpy.mean(throughputs[:detectionResult[0]]))
                avgThroughputAfterBoost = float(numpy.mean(throughputs[detectionResult[0]:]))
                # check whether the throughput samples before/after boost are statistically different
                # if different, delayed throttling detected in this test
                ks2dVal, ks2pVal = ks_2samp(throughputs[:detectionResult[0]], throughputs[detectionResult[0]:])
                pThreshold = 0.05
                delayedThrottling = False
                if (ks2pVal < pThreshold) and (avgThroughputBoost > avgThroughputAfterBoost):
                    delayedThrottling = True
                if delayedThrottling:
                    avgThroughputsDelay.append(avgThroughputBoost)
                    avgThroughputsAfterDelay.append(avgThroughputAfterBoost)
                    delayedBytesAggregated.append(MB)
                    delayedTimeAggregated.append(boostTime)
                    cntResult['Delayed'] += 1
                    if cntResult['Delayed'] < self.individualPlotCnt:
                        rpt.display(throughputsSignal, detectionResult)
                        plt.savefig(self.boostDir + plotTitle + str(index) + '.png')
                        plt.cla()
                        plt.clf()
                        plt.close('all')
                else:
                    cntResult['NoDelay'] += 1
            elif len(detectionResult) == 1:
                # no change point detected
                cntResult['NoDelay'] += 1
            elif cntResult['Error'] < self.individualPlotCnt:
                # more than one change point detected
                rpt.display(throughputsSignal, detectionResult)
                plt.savefig(self.boostDir + 'error_' + plotTitle + str(index) + '.png')
                plt.cla()
                plt.clf()
                plt.close('all')
                cntResult['Error'] += 1

        delayedBytes = []
        delayedTime = []

        # if boostBytes,
        # check whether the average throughput before and after boost are statistically different
        if delayedBytesAggregated and len(avgThroughputsDelay) > 10 and len(avgThroughputsAfterDelay) > 10:
            ks2dVal, ks2pVal = ks_2samp(avgThroughputsDelay, avgThroughputsAfterDelay)
            greater = True
            pThreshold = 0.05
            if ks2pVal < pThreshold:
                greater = False
            acceptRatio = sampleKS2(avgThroughputsDelay, avgThroughputsAfterDelay, greater=greater,
                                    pThreshold=0.05, sub=0.5, r=100)
            if ks2pVal < 0.05 and acceptRatio > 0.95:
                Xboost, Yboost = list2CDF(avgThroughputsDelay)
                Xafterboost, Yafterboost = list2CDF(avgThroughputsAfterDelay)
                twoCDFs = {'Before change point': (Xboost, Yboost),
                           'After change point': (Xafterboost, Yafterboost)}
                plotCDF(twoCDFs, [], self.diffDir, 'CDF_{}'.format(plotTitle))
                samplingLeft, samplingRight = getRange(delayedBytesAggregated)
                json.dump(delayedBytesAggregated,
                          open('{}/{}_{}_{}_delayedBytes.json'.format(self.plotDataDir, self.carrierName, replayName,
                                                                      aggregationGroup), 'w'))
                delayedBytes, delayedBytesKDEvalues, delayedBytesTruePositiveRanges, ylim, y2lim = self.KDEtest(
                    delayedBytesAggregated,
                    samplingLeft,
                    samplingRight, 0.1,
                    xlabel='Number of bytes transmitted before throttling',
                    plotTitle='bytes_{}'.format(
                        plotTitle))
                # When throttling rate > 2 Mbps, the first ~ 2 MB of Netflix replay might be falsely detected as boost

        return delayedBytes, delayedTime

    # run KDE test for the dataSamples,
    # return a. a list of local maximum samples with densities > the threshold b. the list of their densities
    # c. the true positive ranges (all samples in ranges & with densities > the threshold)
    def KDEtest(self, dataSamples, samplingLeft, samplingRight, fitBandwidth, ylim=None, y2lim=None,
                xlabel='Average throughput (Mbps)',
                threshold=0,
                plotTitle=''):
        dataSamplesNumpyArray = numpy.array(dataSamples).reshape(-1, 1)
        # fit a kernel density function with the input bandwidth
        kde = KernelDensity(kernel='gaussian', bandwidth=fitBandwidth).fit(
            dataSamplesNumpyArray)

        samples = numpy.arange(samplingLeft, samplingRight, fitBandwidth)

        samples = numpy.array(samples).reshape(-1, 1)

        # score_samples returns the log of the probability density
        logProb = kde.score_samples(samples)
        logProbList = list(logProb)
        densities = numpy.array([numpy.exp(x) for x in logProbList])

        # find indexes of all local maximums
        # find true positive ranges and prominent local maximum values
        allProminentValues, allProminentDensities, truePositiveRanges = prominentValuesDetection(
            densities,
            samples,
            samplingLeft,
            samplingRight,
            threshold)

        truePositiveRanges = rationalizeTruePositiveRanges(truePositiveRanges)
        # the following code is for plotting the density out
        binsFrenquency = []

        dataSamples.sort()

        interval = max([10] + dataSamples[: int(98 * len(dataSamples) / 100)]) / float(100)

        for i in range(100):
            binsFrenquency.append(i * interval)

        currentYlim = None
        currentY2lim = None
        # if prominent values are found, plot:
        if len(allProminentValues):
            fig, ax1 = plt.subplots(figsize=(15, 6))

            plotSamples = []
            plotDensities = []

            for index in range(len(samples)):
                if samples[index] > binsFrenquency[-1]:
                    continue
                plotSamples.append(samples[index])
                plotDensities.append(densities[index])
            if 'random' in plotTitle:
                ax1.hist(dataSamples, binsFrenquency, color='#404040')
            else:
                ax1.hist(dataSamples, binsFrenquency)

            currentYlim = ax1.get_ylim()[1]
            if ylim and currentYlim:
                currentYlim = max(ylim, currentYlim)
            ax1.set_ylim(top=currentYlim)

            ax1.set_ylabel('Number of tests')
            ax1.set_xlabel(xlabel)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Normalized Density')
            ax2.plot(plotSamples, plotDensities, c="k", alpha=0.4)
            for m in range(len(allProminentValues)):
                ax2.scatter(allProminentValues[m], allProminentDensities[m], c="seagreen", s=20)
            if threshold:
                plt.axhline(y=threshold, color='r', linestyle='-')

            currentY2lim = ax2.get_ylim()[1]
            if y2lim and currentY2lim:
                currentY2lim = max(y2lim, currentY2lim)
            # ax2.set_ylim(top=3.0, bottom=0.0)
            # set font size
            for ax in [ax1, ax2]:
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(15)

            fig.tight_layout()
            plt.savefig(self.diffDir + '{}_KDE.png'.format(plotTitle), bbox_inches="tight",
                        pad_inches=0)
            plt.cla()
            plt.clf()
            plt.close('all')

        return allProminentValues, allProminentDensities, truePositiveRanges, currentYlim, currentY2lim

    # grouping tests and counting number of tests per day
    # aggregationGroups and replayNames are both lists
    # if aggregationGroup is set, only include tests from specified aggregationGroups
    # otherwise, plot the tests with standard deviation (all aggregation groups)
    # if specifiedReplayNames is set, only include tests from the replayNames
    # otherwise, include all tests (all replay names)
    def groupTestsPerDay(self, replayNames, aggregationGroups):
        # number of positive and negative tests per day
        groupedTests = {}
        for replayName in self.allTests:
            if replayNames and (replayName not in replayNames):
                continue
            for aggregationGroup in self.allTests[replayName]:
                if aggregationGroups and (aggregationGroup not in aggregationGroups):
                    continue
                if (aggregationGroup not in self.positiveTests[replayName]) or (
                        aggregationGroup not in self.negativeTests[replayName]):
                    continue
                for onePositiveTest in self.positiveTests[replayName][aggregationGroup]:
                    incomingTime = onePositiveTest['geoInfo']['localTime']
                    if incomingTime:
                        incomingYMD = incomingTime.split(' ')[0]
                        if incomingYMD not in groupedTests:
                            groupedTests[incomingYMD] = {'positive': 0, 'negative': 0}

                        groupedTests[incomingYMD]['positive'] += 1

                for oneNegativeTest in self.negativeTests[replayName][aggregationGroup]:
                    incomingTime = oneNegativeTest['geoInfo']['localTime']
                    if incomingTime:
                        incomingYMD = incomingTime.split(' ')[0]
                        if incomingYMD not in groupedTests:
                            groupedTests[incomingYMD] = {'positive': 0, 'negative': 0}

                        groupedTests[incomingYMD]['negative'] += 1

        return groupedTests

    # Return True if time of day effect detected
    def checkTimeOfDayEffect(self, replayName, aggregationGroup, throttlingStat):
        testsPerHour, throughputsGroupedByHour = groupTestsPerHour(self.truePositiveTests[replayName][aggregationGroup],
                                                                   self.negativeTests[replayName][aggregationGroup])
        # set value to zero if no test from that hour
        for hour in range(0, 24):
            hourStr = str(hour)
            if hour < 10:
                hourStr = '0' + hourStr
            if hourStr not in testsPerHour:
                testsPerHour[hourStr] = [0, 0]
                throughputsGroupedByHour[hourStr] = [[], []]

        testHours = sorted(testsPerHour.keys())
        # get the fraction during day and night
        totalTestsDay = 0
        totalTestsNight = 0
        throttledTestsDay = 0
        throttledTestsNight = 0

        totalTests = 0
        throttledTests = 0

        avgRandomDay = []
        avgOriDay = []
        avgRandomNight = []
        avgOriNight = []
        numRandomBelowThrottleDay = 0
        numRandomBelowThrottleNight = 0
        for testHour in testHours:
            # list of total tests during this hour
            numTotalTestsThisHour = testsPerHour[testHour][0] + testsPerHour[testHour][1]
            # list of positive tests during this hour
            numPositiveTestsThisHour = testsPerHour[testHour][0]
            # list of positive ratio during this hour

            avgOriThisHour = throughputsGroupedByHour[testHour][0]
            avgRandomThisHour = throughputsGroupedByHour[testHour][1]

            numRandomBelowThrottleThisHour = 0

            for avgRandom in avgRandomThisHour:
                belowTR = False
                for throttlingRate in throttlingStat['Throttling true positive ranges']:
                    if avgRandom <= throttlingRate[1]:
                        belowTR = True
                if belowTR:
                    numRandomBelowThrottleThisHour += 1

            if '00' <= testHour <= '08':
                totalTestsNight += numTotalTestsThisHour
                throttledTestsNight += numPositiveTestsThisHour
                avgOriNight += avgOriThisHour
                avgRandomNight += avgRandomThisHour
                numRandomBelowThrottleNight += numRandomBelowThrottleThisHour

            else:
                totalTestsDay += numTotalTestsThisHour
                throttledTestsDay += numPositiveTestsThisHour
                avgOriDay += avgOriThisHour
                avgRandomDay += avgRandomThisHour
                numRandomBelowThrottleDay += numRandomBelowThrottleThisHour

            totalTests += numTotalTestsThisHour
            throttledTests += numPositiveTestsThisHour

        fractionAll = throttledTests / float(totalTests)
        expectThrottledTestsDay = int(fractionAll * totalTestsDay)
        expectThrottledTestsNight = int(fractionAll * totalTestsNight)

        fractionRandomBelowDay = fractionRandomBelowNight = 0
        if totalTestsNight and totalTestsDay:
            fractionRandomBelowDay = numRandomBelowThrottleDay / float(totalTestsDay)
            fractionRandomBelowNight = numRandomBelowThrottleNight / float(totalTestsNight)

        avgAvgRandomDay = numpy.mean(avgRandomDay)
        avgAvgOriDay = numpy.mean(avgOriDay)
        avgAvgRandomNight = numpy.mean(avgRandomNight)
        avgAvgOriNight = numpy.mean(avgOriNight)

        chisq, p = chisquare([throttledTestsDay, throttledTestsNight],
                             f_exp=[expectThrottledTestsDay, expectThrottledTestsNight])

        # fractionTestsDuringDay = totalTestsDay / float(totalTestsNight + totalTestsDay)

        timeOfDay = False
        if p < 0.05:
            # avg original, avg random during day and night
            # percentage of avg random below throttling rate during day and night
            timeOfDay = True
            print(
                '\r\n all stats', self.carrierName, replayName, fractionRandomBelowDay, fractionRandomBelowNight,
                avgAvgOriDay,
                avgAvgRandomDay, avgAvgOriNight, avgAvgRandomNight, totalTestsDay, totalTestsNight)
        else:
            print('no time of day effect', self.carrierName, replayName)

        return timeOfDay, testsPerHour

    # for each aggregationGroup, count the number of tests/ratio of positive test per hour during that period
    # default grouping all replays together, unless specified
    # plot out the graphs
    def analyzeTestsPerHour(self, replayName, aggregationGroup):
        timeOfDay, testsPerHour = self.checkTimeOfDayEffect(replayName, aggregationGroup,
                                                            self.throttlingStats[replayName][aggregationGroup])
        json.dump(testsPerHour,
                  open('{}/{}_{}_{}_testsPerHour.json'.format(self.plotDataDir, self.carrierName, replayName,
                                                              aggregationGroup), 'w'))
        plotTestsPerHour(testsPerHour, self.timeDir,
                         plotTitle='{}_{}_{}_{}'.format(self.carrierName, replayName, aggregationGroup, timeOfDay))

    # aggregationGroups and replayNames are both lists
    # if aggregationGroups is given, only include tests from specified aggregationGroups
    # otherwise, plot the tests with all aggregation groups
    # if replayNames is given, only include tests from the replayNames
    # otherwise, include all tests (all replayNames)
    def groupTestsPerState(self, replayNames, aggregationGroups):
        # each state has a two elements list, which is
        # [num positive tests, num negative tests] in that state
        # {'MA' : [x1, y1], 'CA': [...], ...}
        testsGroupedByState = {}
        numAllTests = 0
        numPosTests = 0
        for replayName in self.allTests:
            if replayNames and (replayName not in replayNames):
                continue
            for aggregationGroup in self.allTests[replayName]:
                if aggregationGroups and (aggregationGroup not in aggregationGroups):
                    continue
                if (aggregationGroup not in self.positiveTests[replayName]) or (
                        aggregationGroup not in self.negativeTests[replayName]):
                    continue
                for positiveTest in self.truePositiveTests[replayName][aggregationGroup]:
                    latitude = positiveTest['geoInfo']['latitude']
                    longitude = positiveTest['geoInfo']['longitude']
                    if (not latitude) or (latitude == 'null'):
                        continue
                    # get the state based on lon, lat
                    testState = get_US_states(longitude, latitude)
                    if testState not in testsGroupedByState:
                        testsGroupedByState[testState] = [0, 0]
                    testsGroupedByState[testState][0] += 1
                    numPosTests += 1
                    numAllTests += 1

                for negativeTest in self.negativeTests[replayName][aggregationGroup]:
                    latitude = negativeTest['geoInfo']['latitude']
                    longitude = negativeTest['geoInfo']['longitude']
                    if (not latitude) or (latitude == 'null'):
                        continue
                    testState = get_US_states(longitude, latitude)
                    if testState not in testsGroupedByState:
                        testsGroupedByState[testState] = [0, 0]
                    testsGroupedByState[testState][1] += 1
                    numAllTests += 1

        if not numAllTests:
            posRate = 0
        else:
            posRate = numPosTests / float(numAllTests)

        return testsGroupedByState, posRate

    def analyzeTestsPerState(self, replayNames=None, aggregationGroups=None):
        # only include the replays that are in throttlingStats
        if self.throttlingStats and (not replayNames):
            replayNames = []
            for replayName in self.throttlingStats:
                if replayName not in replayNames:
                    replayNames.append(replayName)
        testsPerState, overallPositiveRate = self.groupTestsPerState(replayNames, aggregationGroups)
        json.dump((testsPerState, overallPositiveRate),
                  open('{}/{}_{}_{}_testsPerState.json'.format(self.plotDataDir, self.carrierName, replayNames,
                                                               aggregationGroups), 'w'))

        # Whether there is geo difference: what is variance of throttling ratio among states
        standarDeviationStates = checkVarianceInThrottling(testsPerState)
        plotTestsPerState(testsPerState, overallPositiveRate, self.diffDir,
                          plotTitle='{}_{}_{}_{}_{}_State'.format(self.carrierName, aggregationGroups, replayNames,
                                                                  round(overallPositiveRate, 2),
                                                                  standarDeviationStates))

    # store the identified throttlingStats into a json file
    def resultsBookKeeping(self):
        statsForISP = {'throttlingStats': self.throttlingStats}
        json.dump(statsForISP, open('{}/{}_throttlingStats.json'.format(self.diffDir, self.carrierName), 'w'))
        return

    # plot out the positive and negative tests on map based on GPS locations
    def plotTestsGPS(self, replayName, aggregationGroup):
        latitudes = []
        longitudes = []
        diffDetected = []

        for onePositiveTest in self.positiveTests[replayName][aggregationGroup]:
            if onePositiveTest['geoInfo']['latitude'] and (onePositiveTest['geoInfo']['latitude'] != 'null'):
                latitudes.append(float(onePositiveTest['geoInfo']['latitude']))
                longitudes.append(float(onePositiveTest['geoInfo']['longitude']))
                diffDetected.append('Diff')

        for oneNegativeTest in self.negativeTests[replayName][aggregationGroup]:
            if oneNegativeTest['geoInfo']['latitude'] and (oneNegativeTest['geoInfo']['latitude'] != 'null'):
                latitudes.append(float(oneNegativeTest['geoInfo']['latitude']))
                longitudes.append(float(oneNegativeTest['geoInfo']['longitude']))
                diffDetected.append('NoDiff')

        df = pd.DataFrame(
            {'DiffResult': diffDetected,
             'Longitudes': longitudes,
             'Latitudes': latitudes}
        )
        df['Coordinates'] = list(zip(df.Longitudes, df.Latitudes))
        df['Coordinates'] = df['Coordinates'].apply(Point)
        gdf = gpd.GeoDataFrame(df, geometry='Coordinates')
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        # We restrict the map
        partialWorld = world.cx[min(longitudes): max(longitudes), min(latitudes): max(latitudes)]
        ax = partialWorld.plot(figsize=(12, 9), color='gray', edgecolor='white')
        ax.set_axis_off()
        # We can now plot our GeoDataFrame
        # red for tests showing diff, blue for no diff
        gdf.plot(ax=ax, alpha=0.5, column='DiffResult', cmap='viridis', legend=True, legend_kwds={'loc': 'lower right'},
                 figsize=(12, 9))
        plt.savefig(self.diffDir + '{}_geo.png'.format('{}_{}'.format(replayName, aggregationGroup)),
                    bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close('all')
        return

    # Count tests per os
    def analyzeTestsPerOS(self, replayName, aggregationGroup):
        iOSPositive = 0
        androidPositive = 0
        iOSNegative = 0
        androidNegative = 0
        for onePos in self.truePositiveTests[replayName][aggregationGroup]:
            if onePos['os'] == 'Android':
                androidPositive += 1
            else:
                iOSPositive += 1
        for oneNeg in self.truePositiveTests[replayName][aggregationGroup]:
            if oneNeg['os'] == 'Android':
                androidNegative += 1
            else:
                iOSNegative += 1

        androidTotal = androidNegative + androidPositive
        iOSTotal = iOSNegative + iOSPositive
        androidPositiveRate = androidNegativeRate = iOSPositiveRate = iOSNegativeRate = 0
        if androidTotal:
            androidPositiveRate = androidPositive / float(androidTotal)
            androidNegativeRate = androidNegative / float(androidTotal)
        if iOSTotal:
            iOSPositiveRate = iOSPositive / float(iOSTotal)
            iOSNegativeRate = iOSNegative / float(iOSTotal)

        print('\r\n androidPositive %, androidNegative %, iOSPositive %, iOSNegative % ', replayName, aggregationGroup,
              androidPositiveRate, androidNegativeRate,
              iOSPositiveRate, iOSNegativeRate)
        return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('\r\n Example run: python3 WeheDataParser.py [carriersDir], '
              'where the carriersDir is where the aggregated wehe data is')
        sys.exit(1)
    # Where the test results json files are
    script, allISPDir = sys.argv
    countISP = 0
    # aggregation can be empty (default, all tests together), 'month', 'half', 'quarter'
    aggregation = False
    countThrottling = 0

    allThrottlingCases = {}
    weheDiffStatFile = 'weheDiffStat.json'

    # tests are grouped by ISP
    for ISPDir in os.listdir(allISPDir):
        print('\r\n analyzing {}, total ISPs analyzed {}, ISPs with throttling {}'.format(ISPDir, countISP, countThrottling))
        ISP_full_Dir = allISPDir + ISPDir
        if os.path.isdir(ISP_full_Dir):
            countISP += 1
            oneISP = singleISPstats(ISP_full_Dir, aggregation)
            for replayName in oneISP.allTests:
                for aggregationGroup in oneISP.allTests[replayName]:
                    # run analysis for this ISP-replayName-aggregationGroup
                    oneISP.performAnalysis(replayName, aggregationGroup)
                    # If fixed rate throttling is detected for a given ISP-Replay-AggregationGroup specification
                    # The true positives for this specification will be in the truePositiveTests list
                    # The throttling stat (e.g., throttling rates detected) will be kept in oneISP.throttlingStat

            # dump the throttling info for this ISP into a json file
            oneISP.resultsBookKeeping()
            # Additional analysis
            if oneISP.throttlingStats:
                print('\r\n throttling detected', oneISP.throttlingStats)
                for replayName in oneISP.throttlingStats:
                    for aggregationGroup in oneISP.allTests[replayName]:
                        oneISP.analyzeTestsPerOS(replayName, aggregationGroup)
                        oneISP.plotTestsGPS(replayName, aggregationGroup)
                        oneISP.analyzeTestsPerState([replayName], [aggregationGroup])
                        oneISP.analyzeTestsPerHour(replayName, aggregationGroup)
                    for aggregationGroup in oneISP.throttlingStats[replayName]:
                        carrierReplayName = '{}_{}'.format(oneISP.carrierName, replayName)

                        if carrierReplayName not in allThrottlingCases:
                            allThrottlingCases[carrierReplayName] = {}

                        allThrottlingCases[carrierReplayName][aggregationGroup] = oneISP.throttlingStats[replayName][
                            aggregationGroup]
                    countThrottling += 1
    json.dump(allThrottlingCases, open(weheDiffStatFile, 'w'))