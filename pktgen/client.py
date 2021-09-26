from scapy.all import *
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.inet6 import IPv6

import os
import inspect
import time
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(1,'./')
from exp_config.exp_config import PKT_FILE_LIST,NUM_PKT,PKT_FILE_INTERVAL,CSV_OUTPUTPATH,GOGO_TIME,FIRST_TIME_SLEEP

import numpy as np

start=(1*(len('client')))
end=(-1*len('.py'))
hostid=os.path.basename(inspect.getfile(inspect.currentframe()))
hostid=int(hostid[start:end])

def hostid_to_slice(hostid):
    slice=hostid-1
    return slice

def slice_to_client(slice):
    hostid=slice+1
    return hostid

def slice_to_server(slice):
    hostid=slice+8
    return hostid

flow=[0]*7
timerecord=[0]*7
i=hostid_to_slice(hostid)
p=str(Path(PKT_FILE_LIST.get(str(i))))
flow[i] = rdpcap(p)
timerecord[i] = [0]*len(flow[i])
print('ready')

class Counter(dict):
    def __missing__(self,key):
        return 0
        
L3_ctrl=0
L4_ctrl=0

pkt_ctrl={}
for i in range(7):
    pkt_ctrl[i]= Counter()


def pc_flow(pkt,i):
    L3_ctrl=1
    L4_ctrl=1
    
    while L3_ctrl==1:
        try:
            src_ip=pkt[IP].src
            dst_ip=pkt[IP].dst
            
            L3_ctrl=0
            break
        except:
            src_ip='0.0.0.0'
            dst_ip='0.0.0.0'
        try:
            src_ip=pkt[IPv6].src
            dst_ip=pkt[IPv6].dst
            
            L3_ctrl=0
            break
        except:
            src_ip='0:0:0:0:0:0:0:0'
            dst_ip='0:0:0:0:0:0:0:0'
        break
    
    while L4_ctrl==1:
        try:
            src_port=pkt[TCP].sport
            dst_port=pkt[TCP].dport
            L4_ctrl=0
            break
        except:
            src_port=0
            dst_port=0
        try:
            src_port=pkt[UDP].sport
            dst_port=pkt[UDP].dport
            L4_ctrl=0
            break
        except:
            src_port=0
            dst_port=0
        break

    src_socket=str(src_ip)+':'+str(src_port)+'-'+str(dst_ip)+':'+str(dst_port)
    dst_socket=str(dst_ip)+':'+str(dst_port)+'-'+str(src_ip)+':'+str(src_port)
    chk_socket=('10.0.0.'+str(slice_to_server(i)))+':'+str(src_port)+'-'+('10.0.0.'+str(slice_to_client(i)))+':'+str(dst_port)

    if pkt_ctrl[i][src_socket]==0 and pkt_ctrl[i][dst_socket]==0:
        pkt_ctrl[i][src_socket]=1
        pkt_ctrl[i][dst_socket]=0
    return src_socket,chk_socket

def sendp_flow(i,sendp_count):
    timer=time.time()
    waittime=PKT_FILE_INTERVAL[i]

    j=sendp_count
    pkt=flow[i][j]
    pkt_socket,chk_socket=pc_flow(pkt,i)
    if pkt_ctrl[i][pkt_socket]==1:
        try:
            flow[i][j][IP].src='10.0.0.'+str(slice_to_client(i))
            flow[i][j][IP].dst='10.0.0.'+str(slice_to_server(i))
            print(f'send={pkt_socket}')
            sendp(flow[i][j],verbose=False)
        except:
            print('pass')
            pass
        
        timerecord[i][j]=(time.time())
        sendp_count+=1
        
        waittime=waittime-(time.time()-timer)
        if waittime>0:
            time.sleep(waittime)
    elif pkt_ctrl[i][pkt_socket]==0:
        print(f'wait={pkt_socket}')
        waittime=sniff_flow(i,chk_socket,timer,waittime-(1e-05))
        timerecord[i][j]=(time.time())        
        sendp_count+=1
        
        if waittime>0:
                time.sleep(waittime)
    return sendp_count

def sniff_flow(i,wait_socket,timer,waittime):
    t='h'+str(slice_to_client(i))+'-eth0'
    while waittime>0:
        try:
            pkt=sniff(iface=t,count=1,timeout=waittime)
            pkt_socket,chk_socket=pc_flow(pkt[0],i)
        except:
            waittime=waittime-(time.time()-timer)
            continue            
        if pkt_socket==wait_socket:
            print(f'{pkt_socket}=={wait_socket}')
            break
        else:
            print(f'{pkt_socket}!={wait_socket}')
            waittime=waittime-(time.time()-timer) 
            continue
    return waittime
    
def cnt_flow(i):
    sendp_count=0
    time.sleep(FIRST_TIME_SLEEP*i)
    while sendp_count<NUM_PKT[i]:
        sendp_count=sendp_flow(i,sendp_count)
        
    timerecord[i]=np.array(timerecord[i])
    csv_name=CSV_OUTPUTPATH+str(i)+'_client.csv'
    np.savetxt(csv_name,timerecord[i],fmt="%lf",delimiter=",")

def main():
    cnt_flow(hostid_to_slice(hostid))

if __name__ =='__main__':
    timestamp=float(time.time())
    time.sleep(GOGO_TIME-timestamp-10)
    while (timestamp := float(time.time()) ) < GOGO_TIME:
        pass
    main()