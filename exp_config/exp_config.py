"""
time
"""

import time
import math

# lowletter use here, could be change
# upperletter link outside, don't change
# orig_xxx is origin, no orig_xxx is map to pcap file, so "must check" mapping

SCHEDULER_TYPE = "MAX"  #0,1,"random","MAX","min","algo",

timestring = "2021-09-17 08:09:00"
structtime = time.strptime(timestring, "%Y-%m-%d %H:%M:%S")
timestamp = float(time.mktime(structtime))

currtime = time.time()
if timestamp - currtime < 1 * 60:
    print("gg: please config your start time ")

print(timestamp)
GOGO_TIME = timestamp

#unit is second
TOTAL_TIME = 4 * 60
#unit is second, monitor period, controller get budget and scheduler distribute
SLEEP_PERIOD = 1

# custom by your experimnet
# if user > user_threshold will congestion
user_threshold = 1
# depand on your computer
# if send interval < time_threshold will lag
# ref orig time smallest value would be time_threshold
FIRST_TIME_SLEEP = 10
time_threshold = 0.2

print_ctrl = 0

"""
file
"""

#latency, not throughput monitor
CSV_OUTPUTPATH = './pktgen/timerecord/'

####################check####################
PKT_FILE_LIST = {
    #0chat #1email #2file #3stream #4p2p #5voip #6browser
    '0': './pktgen/pcap/aim_chat_3a.pcap',  #0chat
    '1': './pktgen/pcap/aim_chat_3a.pcap',  #0chat
    '2': './pktgen/pcap/aim_chat_3a.pcap',  #0chat
    '3': './pktgen/pcap/aim_chat_3a.pcap',  #0chat
    '4': './pktgen/pcap/aim_chat_3a.pcap',  #0chat
    '5': './pktgen/pcap/aim_chat_3a.pcap',  #0chat
    '6': './pktgen/pcap/aim_chat_3a.pcap',  #0chat
}
"""
#0chat #1email #2file #3stream #4p2p #5voip #6browser
'0' : './pktgen/pcap/aim_chat_3a.pcap',        #0chat
'1' : './pktgen/pcap/email1a.pcap',            #1email
'2' : './pktgen/pcap/ftps_up_2a.pcap',         #2file
'3' : './pktgen/pcap/youtube1.pcap',           #3stream
'4' : './pktgen/pcap/Torrent01.pcapng',        #4p2p
'5' : './pktgen/pcap/skype_chat1a.pcap',       #5voip
'6' : './pktgen/pcap/facebook_chat_4b.pcap',   #6browser
"""

PKT_FILE_MAP = {
    #0chat #1email #2file #3stream #4p2p #5voip #6browser
    0: 0,  #0chat
    1: 0,  #0chat
    2: 0,  #0chat
    3: 0,  #0chat
    4: 0,  #0chat
    5: 0,  #0chat
    6: 0,  #0chat
}
####################check####################

"""
median
"""

#dict statistics from pcap
orig_BW = {
    #avg allbyte/alltime
    #0chat #1email #2file #3stream #4p2p #5voip #6browser
    0: 624,
    1: 4485,
    2: 1029862,
    3: 245670,
    4: 238271,
    5: 14385,
    6: 19673,
}

orig_AVG_INTERVAL = {
    #avg alltime/allcount
    0: 0.514901,
    1: 0.510331,
    2: 0.001012,
    3: 0.017945,
    4: 0.003790,
    5: 0.011784,
    6: 0.058626,
}

orig_ONE_PKT_SIZE = {
    #median
    #0chat #1email #2file #3stream #4p2p #5voip #6browser
    0: 92,
    1: 63,
    2: 1514,
    3: 2742,
    4: 1404,
    5: 149,
    6: 146,
}

#avg MUST CHECK you want avg or median
for i, v in orig_ONE_PKT_SIZE.items():
    orig_ONE_PKT_SIZE[i] = orig_BW[i] * orig_AVG_INTERVAL[i]
print(orig_ONE_PKT_SIZE)
#MUST CHECK you want avg or median
ONE_PKT_SIZE = {i: orig_ONE_PKT_SIZE[PKT_FILE_MAP[i]] for i in orig_ONE_PKT_SIZE}

orig_INNTER_ARRIVAL_TIME = {
    #median
    #0chat #1email #2file #3stream #4p2p #5voip #6browser
    0: 0.000001,
    1: 0.000009,
    2: 0.000002,
    3: 0.000005,
    4: 0.000026,
    5: 0.000003,
    6: 0.000008,
}
INNTER_ARRIVAL_TIME = {i: orig_INNTER_ARRIVAL_TIME[PKT_FILE_MAP[i]] for i in orig_INNTER_ARRIVAL_TIME}

"""
calculator
"""

orig_PKT_FILE_INTERVAL = {}
"""
t=[]
for i,v in orig_ONE_PKT_SIZE.items():
    t.append((time_threshold+0.1)/orig_ONE_PKT_SIZE[i]*orig_BW[i])
t=sorted(t)
wx=t[-2]

for i,v in orig_ONE_PKT_SIZE.items():
    orig_PKT_FILE_INTERVAL[i]=orig_ONE_PKT_SIZE[i]/orig_BW[i]*wx
    if orig_PKT_FILE_INTERVAL[i]<time_threshold:
        orig_PKT_FILE_INTERVAL[i]=time_threshold
    orig_PKT_FILE_INTERVAL[i]=orig_PKT_FILE_INTERVAL[i]-(orig_PKT_FILE_INTERVAL[i]%0.001)

    if orig_ONE_PKT_SIZE[i]>1500:
        orig_PKT_FILE_INTERVAL[i]=orig_PKT_FILE_INTERVAL[i]/(ONE_PKT_SIZE[i]/1500)
        orig_ONE_PKT_SIZE[i]=ONE_PKT_SIZE[i]/math.ceil(ONE_PKT_SIZE[i]/1500)
        ONE_PKT_SIZE = {i:orig_ONE_PKT_SIZE[PKT_FILE_MAP[i]] for i in orig_ONE_PKT_SIZE}
"""

orig_PKT_FILE_INTERVAL = orig_AVG_INTERVAL

PKT_FILE_INTERVAL = {i: orig_PKT_FILE_INTERVAL[PKT_FILE_MAP[i]] for i in orig_PKT_FILE_INTERVAL}
if print_ctrl == 1:
    print(orig_PKT_FILE_INTERVAL)
    print(PKT_FILE_INTERVAL)

orig_NUM_PKT = {}
for i, v in ONE_PKT_SIZE.items():
    orig_NUM_PKT[i] = int(TOTAL_TIME / orig_PKT_FILE_INTERVAL[i])
NUM_PKT = {i: orig_NUM_PKT[PKT_FILE_MAP[i]] for i in orig_NUM_PKT}
if print_ctrl == 1:
    print(NUM_PKT)

#60 is monitor latency packet, mininet unit is MBytes /1000000, 1Byte=8bit
MININET_BW = {}
for i in orig_ONE_PKT_SIZE:
    c = float(1 / orig_PKT_FILE_INTERVAL[i])
    MININET_BW[i] = float(user_threshold * orig_ONE_PKT_SIZE[i] * c / 1000000 * 8)
if print_ctrl == 1:
    print(MININET_BW)

BUDGET_BW = {}
for i in orig_ONE_PKT_SIZE:
    c = float(SLEEP_PERIOD / orig_PKT_FILE_INTERVAL[i])
    BUDGET_BW[i] = int(user_threshold * orig_ONE_PKT_SIZE[i] * c)
if print_ctrl == 1:
    print(BUDGET_BW)

orig_BUDGET_PKT_SIZE = {}
for i in orig_ONE_PKT_SIZE:
    c = float(SLEEP_PERIOD / orig_PKT_FILE_INTERVAL[i])
    orig_BUDGET_PKT_SIZE[i] = int(orig_ONE_PKT_SIZE[i] * c)
BUDGET_PKT_SIZE = {i: orig_BUDGET_PKT_SIZE[PKT_FILE_MAP[i]]for i in orig_BUDGET_PKT_SIZE}
if print_ctrl == 1:
    print(BUDGET_PKT_SIZE)

orig_RESULT = {}
for i in orig_ONE_PKT_SIZE:
    orig_RESULT[i] = int(orig_ONE_PKT_SIZE[i] * orig_NUM_PKT[i])
RESULT = {i: orig_RESULT[PKT_FILE_MAP[i]] for i in orig_RESULT}
if print_ctrl == 1:
    print(RESULT)
