'''
This script aggregates all client tests based on carriers
Tests collected by Wehe is separated by clientID, e.g., /clientID1 contains all tests done by clientID1
To enable further (ISP-app pair based) analysis, this script instead aggregates tests based on ISP,
i.e., separate tests into /carrier1, which contains all tests from carrier1, and the tests are saved in
json files such as carrier1_Youtube_Timestamp.json. Timestamp is from the replay test, for example,
Youtube_12122018 means the test was done using the youtube trace collected on 2018 12 12
'''
import os
import pickle
import json
import reverse_geocode
import sys
import traceback
import logging
import signal
import netaddr as neta
import copy
import time
import subprocess

from datetime import datetime
from timezonefinder import TimezoneFinder
from dateutil import tz
from contextlib import contextmanager
from threading import Timer

logger = logging.getLogger('weheAnalysis')


# French server names are manually verified
def getFrenchCarrierName(wifiCarrierName):
    carrierName = wifiCarrierName
    if 'proxad' in wifiCarrierName:
        carrierName = 'Free (WiFi)'
    elif 'SFR' in wifiCarrierName:
        carrierName = 'SFR (WiFi)'
    elif 'orange' in wifiCarrierName:
        carrierName = 'Orange (WiFi)'
    elif 'bouyguestelecom' in wifiCarrierName:
        carrierName = 'BouyguesTelecom (WiFi)'
    elif 'gaoland' in wifiCarrierName:
        carrierName = 'Free (WiFi)'
    elif 'ORANGEFRANCEHSIAB' in wifiCarrierName:
        carrierName = 'Orange (WiFi)'
    elif 'BouyguesTelecomSA' in wifiCarrierName:
        carrierName = 'BouyguesTelecom (WiFi)'
    elif 'OrangeSA' in wifiCarrierName:
        carrierName = 'Orange (WiFi)'
    elif 'FreeSAS' in wifiCarrierName:
        carrierName = 'Free (WiFi)'
    elif 'BOUYGTEL' in wifiCarrierName:
        carrierName = 'BouyguesTelecom (WiFi)'

    return carrierName


class currentTestsInfo(object):
    def __init__(self, currentDir):
        self.testsPerDay = {'WiFi': {}, 'cellular': {}}
        self.testsPerISP = {}
        self.numTests = 0
        self.parseAllTests(currentDir)

    '''
    update testsPerDay, testsPerCountry, testsPerISP, numTests
    with tests from this client
    Do not filter any test, add them as is
    '''

    def parseAllTests(self, currentDir):

        decisionDir = currentDir + '/decisions'
        # if no test was ever finished
        if not os.path.isdir(decisionDir):
            return

        # For each decision, find its replayInfo
        for decision in os.listdir(decisionDir):

            # get test metaInfo (in the file name)
            metaInfo = decision.split('.')[0].split('_')

            # there should be 5 different info in the file name
            if len(metaInfo) != 5:
                # logger.warn('\r\n NOT 5 TUPLES METAINFO {}'.format(metaInfo))
                continue

            userID = metaInfo[1]
            side = metaInfo[2]
            historyCount = metaInfo[3]
            testID = metaInfo[4]

            # ignore the DPI tests for now
            if testID not in ['0', '1']:
                continue

            uniqueTestID = userID + '_' + historyCount

            if side != 'Client':
                # logger.info('\r\n Ignore Server side analysis for now {} '.format(decisionFile))
                continue

            replayInfoFileName = 'replayInfo_{}_{}_{}'.format(userID, historyCount, testID)
            # The client throughputs for the original replay and the randomized replay for each test
            originalXputsFileName = 'Xput_{}_{}_{}'.format(userID, historyCount, 0)
            randomXputsFileName = 'Xput_{}_{}_{}'.format(userID, historyCount, 1)
            mobileStatsFileName = 'mobileStats_{}_{}_{}'.format(userID, historyCount, testID)

            replayInfoFileName = currentDir + '/replayInfo/' + replayInfoFileName
            originalXputInfoFile = currentDir + '/clientXputs/' + originalXputsFileName
            randomXputInfoFile = currentDir + '/clientXputs/' + randomXputsFileName
            mobileStatsFile = currentDir + '/mobileStats/' + mobileStatsFileName

            replayInfo = loadReplayInfo(replayInfoFileName)
            original_xputs, original_ts = loadClientXputs(originalXputInfoFile)
            random_xputs, random_ts = loadClientXputs(randomXputInfoFile)

            # if any of these fails
            if (not replayInfo) or (not original_xputs) or (not random_xputs):
                continue
            incomingTime = replayInfo[0]
            incomingDate = incomingTime.split(' ')[0]
            clientIP = replayInfo[2]
            # anonymize client by modifying the ip to only first three octets, e.g., v4: 1.2.3.4 -> 1.2.3.0,
            # v6 : 1:2:3:4:5:6 -> 1:2:3:4:5:0000
            if '.' in clientIP:
                v4ExceptLast = clientIP.rsplit('.', 1)[0]
                clientIP = v4ExceptLast + '.0'
            else:
                v6ExceptLast = clientIP.rsplit(':', 1)[0]
                clientIP = v6ExceptLast + ':0000'

            # if no mobile stat in replayInfo[14]
            if not replayInfo[14]:
                mobileStats = loadMobileStatsFile(mobileStatsFile)
            else:
                mobileStats = json.loads(replayInfo[14])
            if not mobileStats:
                continue
            lat, lon, country, countryCode, city = loadMobileStats(mobileStats)

            appName, replayName = self.updateReplayName(replayInfo[4])

            networkType = mobileStats['networkType']

            if "updatedCarrierName" in mobileStats:
                carrierName = mobileStats["updatedCarrierName"]
            elif networkType == 'WIFI':
                carrierName = self.getCarrierNameByIP(clientIP)
            else:
                carrierName = ''.join(e for e in mobileStats['carrierName'] if e.isalnum())
                carrierName = carrierName + ' (cellular)'

            # Special case for French ISPs, manually verified the transforming from whois results to provider names
            if country == 'France' and networkType == 'WIFI':
                carrierName = getFrenchCarrierName(carrierName)

            # combine the tests with carrierName variance
            if ' ' in carrierName:
                networkPortion = carrierName.split(' ')[1]
                # combine carriernames
                if 'VZW' in carrierName:
                    carrierName = 'Verizon ' + networkPortion
                elif 'VzW' in carrierName:
                    carrierName = 'Verizon ' + networkPortion
                elif 'Verizon' in carrierName:
                    carrierName = 'Verizon ' + networkPortion
                elif 'O2UK' in carrierName:
                    carrierName = 'O2 ' + networkPortion
                elif 'ATT' in carrierName:
                    carrierName = 'ATT ' + networkPortion
                elif 'TMobile' in carrierName:
                    carrierName = 'TMobile ' + networkPortion
                elif 'IowaWireless' in carrierName:
                    carrierName = 'iWireless ' + networkPortion
            elif networkType == 'WIFI':
                carrierName = carrierName + ' (WiFi)'
            else:
                carrierName = carrierName + ' (cellular)'

            # update testsPerDay
            if networkType == 'WIFI':
                addOrUpdate(self.testsPerDay['WiFi'], incomingDate, 1)
            else:
                addOrUpdate(self.testsPerDay['cellular'], incomingDate, 1)

            if carrierName not in self.testsPerISP:
                self.testsPerISP[carrierName] = {replayName: []}

            if replayName not in self.testsPerISP[carrierName]:
                self.testsPerISP[carrierName][replayName] = []

            manufacture = mobileStats['manufacturer']
            model = mobileStats['model']
            osVersion = mobileStats['os']
            if manufacture == 'Apple':
                mobileOS = 'iOS'
            else:
                mobileOS = 'Android'

            if 'localTime' in mobileStats["locationInfo"]:
                localTime = mobileStats["locationInfo"]['localTime']
            else:
                localTime = getLocalTime(incomingTime, lon, lat)
            self.numTests += 1

            self.testsPerISP[carrierName][replayName].append({'uniqueTestID': uniqueTestID,
                                                              'original_xPuts': (
                                                                  original_xputs, original_ts),
                                                              'random_xPuts': (random_xputs, random_ts),
                                                              'geoInfo': {'latitude': lat,
                                                                          'longitude': lon,
                                                                          'country': country,
                                                                          'countryCode': countryCode,
                                                                          'city': city,
                                                                          'localTime': localTime},
                                                              'timestamp': incomingTime,
                                                              'carrierName': carrierName,
                                                              'replayName': replayName,
                                                              'os': mobileOS,
                                                              'osVersion': osVersion,
                                                              'model': model,
                                                              'clientIP': clientIP})

            mobileStats["updatedCarrierName"] = carrierName
            mobileStats["locationInfo"] = {'latitude': lat,
                                           'longitude': lon,
                                           'country': country,
                                           'countryCode': countryCode,
                                           'city': city,
                                           'localTime': localTime}
            replayInfo[14] = json.dumps(mobileStats)
            self.updateReplayInfo(replayInfoFileName, replayInfo)

    def updateReplayName(self, replayName):
        replayTimeStamp = ''
        if '-' in replayName:
            replayTimeStamp = replayName.split('-')[1]

        if 'Random' in replayName:
            appName = replayName.split('Random')[0]
        else:
            appName = replayName.split('-')[0]

        # make the replayName Skype, instead of SkypeUDP
        if 'UDP' in appName:
            appName = appName.split('UDP')[0] + appName.split('UDP')[1]

        replayName = appName + '_' + replayTimeStamp

        return appName, replayName

    def updateReplayInfo(self, replayInfoFileName, updatedReplayInfo):
        replayInfoJson = replayInfoFileName + '.json'
        json.dump(updatedReplayInfo, open(replayInfoJson, 'w'))

    def getCarrierNameByIP(self, clientIP):
        # get WiFi network carrierName
        try:
            IPrange, org = getRangeAndOrg(clientIP)
            if not org:
                carrierName = ' (WiFi)'
                logger.warn('NO ORG Failed at getting carrierName for {}'.format(clientIP))
            else:
                # Remove special characters in carrierName to merge test results together
                carrierName = ''.join(e for e in org if e.isalnum()) + ' (WiFi)'
        except Exception as e:
            logger.warn('EXCEPTION Failed at getting carrierName for {}, {}'.format(clientIP, e))
            carrierName = ' (WiFi)'

        return carrierName


def timedRun(cmd, timeout_sec):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = Timer(timeout_sec, proc.kill)
    out = ''
    try:
        timer.start()
        out, stderr = proc.communicate()
    finally:
        timer.cancel()
    return out


def getRangeAndOrg(ip):
    out = timedRun(['whois', ip], 3)
    # try:
    #     out = out.decode("utf-8")
    # except:
    #     logger.warn('\r\n not utf-8 format')

    IPRange = None
    orgName = None
    netRange = None

    # Get IP Range
    if 'NetRange:' in out:
        netRange = out.split('NetRange:')[1].split('\n')[0]
        netRange = netRange.split()
        IPRange = neta.IPRange(netRange[0], netRange[2])

    # LACNIC/RIPE format
    elif 'inetnum:' in out:
        netRange = out.split('inetnum:')[1].split('\n')[0]
        if '/' in netRange:
            netRange = netRange.split()[0]
            IPRange = neta.IPSet(neta.IPNetwork(netRange))
        else:
            netRange = netRange.split()
            IPRange = neta.IPRange(netRange[0], netRange[2])

    # Get Organization
    if 'OrgName:' in out:
        orgName = out.split('OrgName:')[1].split('\n')[0]
    elif 'Organization:' in out:
        orgName = out.split('Organization:')[1].split('\n')[0]
    elif 'owner:' in out:
        orgName = out.split('owner:')[1].split('\n')[0]
    elif 'org-name:' in out:
        orgName = out.split('org-name:')[1].split('\n')[0]
    elif 'abuse-mailbox:' in out:
        orgName = out.split('abuse-mailbox:')[1].split('@')[1].split('.')[0]
    elif 'netname:' in out:
        orgName = out.split('netname:')[1].split('\n')[0]

    if orgName and netRange:
        return IPRange, orgName
    else:
        return None, None


def addOrUpdate(targetDic, keyToCheck, addValue):
    if keyToCheck not in targetDic:
        targetDic[keyToCheck] = 0
    targetDic[keyToCheck] += addValue


def loadMobileStats(mobileStats):
    # use mobile stats to locate the geoInfo

    try:
        lat = mobileStats['locationInfo']['latitude']
        lon = mobileStats['locationInfo']['longitude']
        # later version of the replay server stores location info in replayInfo file
        if 'country' in mobileStats['locationInfo'] and 'countryCode' in mobileStats['locationInfo'] and lat:
            lat = float("{0:.1f}".format(float(lat)))
            lon = float("{0:.1f}".format(float(lon)))
            country = mobileStats['locationInfo']['country']
            city = mobileStats['locationInfo']['city']
            countryCode = mobileStats['locationInfo']['countryCode']
        elif (lat == lon == '0.0') or (lat == lon == 0.0) or (lat == 'nil') or (lat == 'null'):
            lat = lon = ''
            country = ''
            city = ''
            countryCode = ''
        elif lat:
            coordinates = [(float(lat), float(lon))]
            geoInfo = reverse_geocode.search(coordinates)[0]
            country = geoInfo['country']
            city = geoInfo['city']
            countryCode = geoInfo['country_code'].lower()
            lat = float("{0:.1f}".format(float(lat)))
            lon = float("{0:.1f}".format(float(lon)))
        else:
            lat = lon = country = countryCode = city = ''

    except Exception as e:
        logger.warn('\r\n fail at loading mobileStats from JSON for {}'.format(mobileStats))
        traceback.print_exc(file=sys.stdout)
        country = ''
        city = ''
        countryCode = ''
        lat = lon = ''

    return lat, lon, country, countryCode, city


# Load results and replayInfo from files
# If format is pickle, create a json file
def loadReplayInfo(replayInfoFileName):
    try:
        # Check whether replayInfo is in json or pickle
        replayInfoPickle = replayInfoFileName + '.pickle'
        replayInfoJson = replayInfoFileName + '.json'
        if os.path.exists(replayInfoPickle):
            replayInfo = pickle.load(open(replayInfoPickle, 'r'))
        elif os.path.exists(replayInfoJson):
            replayInfo = json.load(open(replayInfoJson, 'r'))
        else:
            return False
    except:
        traceback.print_exc(file=sys.stdout)
        return False

    return replayInfo


# Load client side throughputs from xput file
def loadClientXputs(xputInfofile):
    try:
        xputInfoPickle = xputInfofile + '.pickle'
        xputInfoJson = xputInfofile + '.json'
        if os.path.exists(xputInfoPickle):
            (xPuts, ts) = pickle.load(open(xputInfoPickle, 'r'))
        elif os.path.exists(xputInfoJson):
            (xPuts, ts) = json.load(open(xputInfoJson, 'r'))
        else:
            logger.warn('Failed at finding xputs for {} '.format(xputInfofile))
            return False, False
    except:
        traceback.print_exc(file=sys.stdout)
        logger.warn('Failed at Loading xputs Info or client for {}'.format(xputInfofile))
        return False, False

    return xPuts, ts


# Load client side throughputs from xput file
def loadMobileStatsFile(mobileStatsFile):
    try:
        mobileStatsJson = mobileStatsFile + '.json'
        if os.path.exists(mobileStatsJson):
            mobileStatsString = json.load(open(mobileStatsJson, 'r'))
            mobileStats = json.loads(mobileStatsString)
        else:
            logger.warn('Failed at finding mobileStats file for {} '.format(mobileStatsJson))
            return False
    except:
        traceback.print_exc(file=sys.stdout)
        logger.warn('Failed at Loading mobileStats Info for {}'.format(mobileStatsFile))
        return False

    return mobileStats


def getLocalTime(utcTime, lon, lat):
    if (lat == lon == '0.0') or (lat == lon == 0.0) or lat == 'null' or (not lat):
        return str(datetime.strptime(utcTime, '%Y-%m-%d %H:%M:%S'))

    utcTime = datetime.strptime(utcTime, '%Y-%m-%d %H:%M:%S')

    tf = TimezoneFinder()
    from_zone = tz.gettz('UTC')

    to_zone = tf.timezone_at(lng=lon, lat=lat)

    to_zone = tz.gettz(to_zone)

    utc = utcTime.replace(tzinfo=from_zone)

    # Convert time zone
    convertedTime = str(utc.astimezone(to_zone))

    return convertedTime


# aggregate tests based on ISPs and create JSON files for all tests
def createFileForEachISP(allTestsPerISP, parentDir=''):
    if not os.path.isdir(parentDir):
        os.mkdir(parentDir)
    for carrierName in allTestsPerISP:
        carrierDir = parentDir + carrierName
        if not os.path.isdir(carrierDir):
            os.mkdir(carrierDir)

        for replayName in allTestsPerISP[carrierName]:
            fileName = carrierDir + '/' + carrierName + '_' + replayName + '.json'
            json.dump(allTestsPerISP[carrierName][replayName], open(fileName, 'w'))


'''
This method creates one JSON files for each ISP: (each ISP name should be in the format : 'carriername (network type)')
Each JSON file contains dictionaries for each replay that the user tested.

testsForOneISP = {'replayName1' : [{test_1}, {test_2}, ...],
                  ... }

each test contains the following fields:
{ 'uniqueTestID' : userID_historyCount, 'original_xPuts': (original_xputs, original_ts), 'random_xPuts' : (random_xputs, random_ts), 
'GeoInfo' : {'lat': ,'lon': ,'country': ,'city': ,}, 'carrierName': ,'timestamp': ,'replayName': , 'clientIP': , 'os': }
'''


def createJSONForallISP(testsDir, resultsDir):
    countClientsWithTest = 0
    countClientsNoTest = 0
    countTotalTests = 0
    countClientsWiFi = 0
    countClientsCellular = 0
    countClientsBoth = 0

    allTestsPerDay = {'WiFi': {}, 'cellular': {}}
    allTestsPerISP = {}

    countMultiplier = 0
    for oneDir in os.listdir(testsDir):
        if countTotalTests >= 10000 * countMultiplier:
            logger.info(
                '\r\n new tests {}, all tests so far {}, all users with test so far {}'.format(oneDir, countTotalTests,
                                                                                                countClientsWithTest))
            countMultiplier += 1
        currentDir = testsDir + '/' + oneDir
        currentInfo = currentTestsInfo(currentDir)

        if currentInfo.numTests:
            countClientsWithTest += 1
        else:
            countClientsNoTest += 1
            continue

        wifiTested = False
        cellularTested = False
        for networkType in currentInfo.testsPerDay:
            clientTestsNetwork = 0
            for testDate in currentInfo.testsPerDay[networkType]:
                countTotalTests += currentInfo.testsPerDay[networkType][testDate]
                clientTestsNetwork += currentInfo.testsPerDay[networkType][testDate]
                if testDate not in allTestsPerDay[networkType]:
                    allTestsPerDay[networkType][testDate] = currentInfo.testsPerDay[networkType][testDate]
                else:
                    allTestsPerDay[networkType][testDate] += currentInfo.testsPerDay[networkType][testDate]
            # if the client has tests for this network
            if clientTestsNetwork:
                if networkType == 'WiFi':
                    countClientsWiFi += 1
                    wifiTested = True
                    if cellularTested:
                        countClientsBoth += 1
                else:
                    cellularTested = True
                    countClientsCellular += 1
                    if wifiTested:
                        countClientsBoth += 1

        # count testfromthisISP
        clientTestsNumber = 0

        for testISP in currentInfo.testsPerISP:
            for replayName in currentInfo.testsPerISP[testISP]:
                clientTestsNumber += len(currentInfo.testsPerISP[testISP][replayName])

        for testISP in currentInfo.testsPerISP:
            if testISP not in allTestsPerISP:
                allTestsPerISP[testISP] = copy.deepcopy(currentInfo.testsPerISP[testISP])
            else:
                for replayName in currentInfo.testsPerISP[testISP]:
                    if replayName not in allTestsPerISP[testISP]:
                        allTestsPerISP[testISP][replayName] = copy.deepcopy(currentInfo.testsPerISP[testISP][replayName])
                    else:
                        allTestsPerISP[testISP][replayName] += copy.deepcopy(
                            currentInfo.testsPerISP[testISP][replayName])

    json.dump(allTestsPerDay, open('allTestsPerDay.json', 'w'))

    createFileForEachISP(allTestsPerISP, parentDir=resultsDir)
    logger.info(
        '\r\n all tests {}, all users with test {}, user no test {}, user cellular {}, user wifi {}, user both {}'.format(
            countTotalTests, countClientsWithTest, countClientsNoTest,
            countClientsCellular, countClientsWiFi, countClientsBoth))


def main():
    logging.getLogger().setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s--%(name)s--%(levelname)s\t%(message)s', datefmt='%m/%d/%Y--%H:%M:%S')
    handler = logging.FileHandler('weheParser.log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    analysisStarts = time.time()

    if len(sys.argv) != 3:
        print('\r\n Example run: python3 WeheDataParser.py [testsDir] [resultsDir], '
              'where the testsDir is where the tests are, resultsDir is where to put the results')
        sys.exit(1)

    script, testsDir, resultsDir = sys.argv

    createJSONForallISP(testsDir, resultsDir)

    analysisEnds = time.time()

    print('\r\n analysis time ', analysisEnds - analysisStarts)


if __name__ == "__main__":
    main()
