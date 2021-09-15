"""
time
"""
import time
import math

# lowletter use here, could be change
# upperletter link outside, don't change
# LIST_xxx is origin, no LIST_xxx is map to pcap file, so "must check" mapping

CSV_OUTPUTPATH = './pg/timerecord/'

SCHEDULER_TYPE = "random" #0,1,"random","MAX","min","algo",

timestring= "2021-09-15 16:03:00"
structtime=time.strptime(timestring, "%Y-%m-%d %H:%M:%S")
timestamp=float(time.mktime(structtime))
print(timestamp)

GOGO_TIME = timestamp

#unit is second
TOTAL_TIME = 4*60
#unit is second, monitor period, controller get budget and scheduler distribute
SLEEP_PERIOD = 1

# custom by your experimnet
# if user > user_threshold will congestion
user_threshold = 1.1
# depand on your computer
# if send interval < time_threshold will lag
# ref orig time smallest value would be time_threshold
FIRST_TIME_SLEEP = 10
time_threshold = 0.2

print_ctrl=1

"""
file
"""
####################check####################
PKT_FILE_LIST = {
'0' : './pg/pcap/aim_chat_3a.pcap',        #0chat
'1' : './pg/pcap/aim_chat_3a.pcap',        #0chat
'2' : './pg/pcap/aim_chat_3a.pcap',        #0chat
'3' : './pg/pcap/aim_chat_3a.pcap',        #0chat
'4' : './pg/pcap/aim_chat_3a.pcap',        #0chat
'5' : './pg/pcap/aim_chat_3a.pcap',        #0chat
'6' : './pg/pcap/aim_chat_3a.pcap',        #0chat
}

"""
'0' : './pg/pcap/facebook_audio1a.pcap',   #6browser
'1' : './pg/pcap/aim_chat_3a.pcap',        #0chat
'2' : './pg/pcap/aim_chat_3a.pcap',        #0chat
'3' : './pg/pcap/aim_chat_3a.pcap',        #0chat
'4' : './pg/pcap/youtube1.pcap',           #3stream
'5' : './pg/pcap/youtube1.pcap',           #3stream
'6' : './pg/pcap/youtube1.pcap',           #3stream
"""
PKT_FILE_MAP = {
0 : 0,        #0chat
1 : 0,        #0chat
2 : 0,        #0chat
3 : 0,        #0chat
4 : 0,        #0chat
5 : 0,        #0chat
6 : 0,        #0chat
}
####################check####################
"""
'0' : './pg/pcap/aim_chat_3a.pcap',        #0chat
'1' : './pg/pcap/email1a.pcap',            #1email
'2' : './pg/pcap/ftps_up_2a.pcap',         #2file
'3' : './pg/pcap/youtube1.pcap',           #3stream
'4' : './pg/pcap/Torrent01.pcapng',        #4p2p
'5' : './pg/pcap/skype_chat1a.pcap',       #5voip
'6' : './pg/pcap/facebook_chat_4b.pcap',   #6browser
"""

"""
#0chat
'0' : './pg/pcap/25.pcapng',  
#1email      
'1' : './pg/pcap/26.pcapng',
#2file        
'2' : './pg/pcap/25.pcapng',
#3stream        
'3' : './pg/pcap/26.pcapng',
#4p2p        
'4' : './pg/pcap/25.pcapng',
#5voip        
'5' : './pg/pcap/26.pcapng',
#6browser        
'6' : './pg/pcap/25.pcapng',    
"""  


"""
median
"""

#dict statistics from pcap
ORIG_BW = {
    #avg allbyte/alltime
    #0chat #1email #2file #3stream #4p2p #5voip #6browser
    0 : 624,     
    1 : 4485,     
    2 : 1029862,  
    3 : 245670,   
    4 : 238271,   
    5 : 14385,    
    6 : 19673,    
}

LIST_AVG_INTERVAL={
    #avg alltime/allcount
    0:0.514901,
    1:0.510331,
    2:0.001012,
    3:0.017945,
    4:0.003790,
    5:0.011784,
    6:0.058626,
}

LIST_ONE_PKT_SIZE = {
    #median
    #0chat #1email #2file #3stream #4p2p #5voip #6browser
    0:92,
    1:63,
    2:1514,
    3:2742,
    4:1404,
    5:149,
    6:146,
    }

#avg MUST CHECK you want avg or median
for i,v in LIST_ONE_PKT_SIZE.items():
    LIST_ONE_PKT_SIZE[i] = ORIG_BW[i] * LIST_AVG_INTERVAL[i]
print(LIST_ONE_PKT_SIZE)
#MUST CHECK you want avg or median
ONE_PKT_SIZE = {i:LIST_ONE_PKT_SIZE[PKT_FILE_MAP[i]] for i in LIST_ONE_PKT_SIZE}


LIST_INNTER_ARRIVAL_TIME={
    #median
    #0chat #1email #2file #3stream #4p2p #5voip #6browser
    0:0.000001,
    1:0.000009,
    2:0.000002,
    3:0.000005,
    4:0.000026,
    5:0.000003,
    6:0.000008,
}
INNTER_ARRIVAL_TIME = {i:LIST_INNTER_ARRIVAL_TIME[PKT_FILE_MAP[i]] for i in LIST_INNTER_ARRIVAL_TIME}


"""
calculator
"""

LIST_PKT_FILE_INTERVAL={}

"""
t=[]
for i,v in LIST_ONE_PKT_SIZE.items():
    t.append((time_threshold+0.1)/LIST_ONE_PKT_SIZE[i]*ORIG_BW[i])
t=sorted(t)
wx=t[-2]

for i,v in LIST_ONE_PKT_SIZE.items():
    LIST_PKT_FILE_INTERVAL[i]=LIST_ONE_PKT_SIZE[i]/ORIG_BW[i]*wx
    if LIST_PKT_FILE_INTERVAL[i]<time_threshold:
        LIST_PKT_FILE_INTERVAL[i]=time_threshold
    LIST_PKT_FILE_INTERVAL[i]=LIST_PKT_FILE_INTERVAL[i]-(LIST_PKT_FILE_INTERVAL[i]%0.001)

    if LIST_ONE_PKT_SIZE[i]>1500:
        LIST_PKT_FILE_INTERVAL[i]=LIST_PKT_FILE_INTERVAL[i]/(ONE_PKT_SIZE[i]/1500)
        LIST_ONE_PKT_SIZE[i]=ONE_PKT_SIZE[i]/math.ceil(ONE_PKT_SIZE[i]/1500)
        ONE_PKT_SIZE = {i:LIST_ONE_PKT_SIZE[PKT_FILE_MAP[i]] for i in LIST_ONE_PKT_SIZE}
"""

LIST_PKT_FILE_INTERVAL=LIST_AVG_INTERVAL

PKT_FILE_INTERVAL= {i:LIST_PKT_FILE_INTERVAL[PKT_FILE_MAP[i]] for i in LIST_PKT_FILE_INTERVAL}
if print_ctrl==1:
    print(LIST_PKT_FILE_INTERVAL)
    print(PKT_FILE_INTERVAL)

LIST_NUM_PKT = {}
for i,v in ONE_PKT_SIZE.items():
    LIST_NUM_PKT[i]=int(TOTAL_TIME/LIST_PKT_FILE_INTERVAL[i])
NUM_PKT= {i:LIST_NUM_PKT[PKT_FILE_MAP[i]] for i in LIST_NUM_PKT}
if print_ctrl==1:
    print(NUM_PKT)

#60 is monitor latency packet, mininet unit is MBytes /1000000, 1Byte=8bit
BW = {}
for i in LIST_ONE_PKT_SIZE:
    c=float(1/LIST_PKT_FILE_INTERVAL[i])
    BW[i]=float(user_threshold*LIST_ONE_PKT_SIZE[i]*c/1000000*8)
if print_ctrl==1:
    print(BW)

BUDGET_BW = {}
for i in LIST_ONE_PKT_SIZE:
    c=float(SLEEP_PERIOD/LIST_PKT_FILE_INTERVAL[i])
    BUDGET_BW[i]=int(user_threshold*LIST_ONE_PKT_SIZE[i]*c)
if print_ctrl==1:
    print(BUDGET_BW)

LIST_BUDGET_PKT_SIZE ={}
for i in LIST_ONE_PKT_SIZE:
    c=float(SLEEP_PERIOD/LIST_PKT_FILE_INTERVAL[i])
    LIST_BUDGET_PKT_SIZE[i]=int(LIST_ONE_PKT_SIZE[i]*c)
BUDGET_PKT_SIZE= {i:LIST_BUDGET_PKT_SIZE[PKT_FILE_MAP[i]] for i in LIST_BUDGET_PKT_SIZE}
if print_ctrl==1:
    print(BUDGET_PKT_SIZE)

LIST_RESULT={}
for i in LIST_ONE_PKT_SIZE:    
    LIST_RESULT[i]=int(LIST_ONE_PKT_SIZE[i]*LIST_NUM_PKT[i])
RESULT= {i:LIST_RESULT[PKT_FILE_MAP[i]] for i in LIST_RESULT}
if print_ctrl==1:
    print(RESULT)
