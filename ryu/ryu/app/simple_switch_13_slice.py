# Copyright (C) 2011 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ryu.base import app_manager

from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import MAIN_DISPATCHER, HANDSHAKE_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls

from ryu.ofproto import ofproto_v1_3
from ryu.ofproto import ether
from ryu.ofproto import inet

from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib.packet import ipv4, ipv6
from ryu.lib.packet import udp, tcp
from ryu.lib.packet import icmp

from ryu.lib import hub
from ryu.lib import ofctl_v1_3
from operator import attrgetter

from ryu_customapp import ryu_preprocessing, ryu_scheduler

#sklearn
from ryu.lib import pcaplib
import joblib

#innerdelay
import time
import netaddr
import csv

import sys

sys.path.insert(1, './')
from exp_config.exp_config import SLEEP_PERIOD, orig_BUDGET_PKT_SIZE, BUDGET_BW, GOGO_TIME, TOTAL_TIME, SCHEDULER_TYPE, PKT_FILE_MAP


class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.ofctl = ofctl_v1_3

        self.datapaths = {}

        csv_outputfile = str(time.time()) + "_" + str(SCHEDULER_TYPE) + ".csv"
        self.csv_throughput_record_file = csv_outputfile

        with open(self.csv_throughput_record_file, 'w') as csv_file:
            row = [GOGO_TIME]
            writer = csv.writer(csv_file)
            writer.writerow(row)

        #manual
        self.hard_timeout = ryu_scheduler.scheduler_hard_timeout(0, 0)
        self.hard_timeout[-1] = 1
        self.duration_sec = 0

        #print_ctrl
        self.AllPacketInfo_ctrl = 0
        self.ClassPrint_ctrl = 0
        self.ScheudulerPrint_ctrl = 0
        self.ActionPrint_ctrl = 0
        self.MonitorPrint_ctrl = 0
        self.LatencyPrint_ctrl = 0

        #function_ctrl only scheduler and monitor =1
        self.Classifier_ctrl = 0  # allright by ip=0, classification by model=1
        self.Scheuduler_ctrl = SCHEDULER_TYPE  # 0, 1, "random", "MAX", "min", "algo",
        self.FlowMatch_ctrl = 0
        self.Monitor_ctrl = 1
        self.Latency_ctrl = 0
        self.UpdateBudget_ctrl = 0

        #manual
        #topo check it same with mininet
        self.SliceNum = 7
        self.Host1half = 7
        self.Host2half = 7
        self.HostTotal = self.Host1half + self.Host2half
        self.SwitchTotal = 1 + self.SliceNum + 1

        #for classifier
        self.loaded_model = joblib.load(
            './ryu/ryu/app/ryu_customapp/models/b255v6 RandomForest choice_random=0.004 train_size=0.8 test_size=0.2 choice_split=3 choice_train=2 1630563027.216647.pkl'
        )
        self.packet_count = 0
        self.class_count = {i: 0 for i in range(self.SliceNum)}
        #[-1]=unknown class
        self.class_count[-1] = 0
        self.pcap_writer = pcaplib.Writer(open('mypcap.pcap', 'wb'), snaplen=80)

        #for scheduler
        self.total_length = 0

        #class mapping
        self.app_to_port = {1: {}, 2: {}}
        self.app_to_service = {
            0: 0,
            1: 1,
            2: 6,
            3: 2,
            4: 1,
            5: 5,
            6: 0,
            7: 3,
            8: 2,
            9: 2,
            10: 5,
            11: 3,
            12: 4,
            13: 6,
            14: 3,
            15: 5,
            16: 3,
        }
        self.service_to_string = {
            -1: '-1 Unknown',  # no L3 or L4 or error###
            0: '0 Chat',  # 0AIM, 6ICQ
            1: '1 Email',  # 1Email, 4Gmail
            2: '2 File Transfer',  # 3FTPS, 8SCP, 9SFTP
            3: '3 Streaming',  # 7Netflix, 14Vimeo, 16YouTube
            4: '4 P2P',  # 12Torrent
            5: '5 VoIP',  # 5Hangouts, 10Skype, 11Spotify, 15Voipbuster
            6: '6 Browser',  # 2Facebook, 13 Tor
        }
        """
        0 AIM        :0 Chat
        1 Email      :1 Email
        2 Facebook   :6 Browser
        3 FTPS       :2 File Transfer
        4 Gmail      :1 Email
        5 Hangouts   :5 VoIP
        6 ICQ        :0 Chat
        7 Netflix    :3 Streaming
        8 SCP        :2 File Transfer
        9 SFTP       :2 File Transfer
        10 Skype     :5 VoIP
        11 Spotify   :5 VoIP
        12 Torrent   :4 P2P
        13 Tor       :6 Browser
        14 Vimeo     :3 Streaming
        15 Voipbuster:5 VoIP
        16 YouTube   :3 Streaming
        """

        #manual
        #slice number transfer to switch or outport
        self.SliceNum_to_s1_Slice0Port = self.Host1half + 1
        self.SliceNum_to_s2_Slice0Port = self.Host2half + 1
        for k, v in self.app_to_service.items():
            self.app_to_port[1][k] = v + self.SliceNum_to_s1_Slice0Port
            self.app_to_port[2][k] = v + self.SliceNum_to_s2_Slice0Port
        self.slice_to_dstport = {1: {}, 2: {}}
        self.slice_to_srcport = {1: {}, 2: {}}
        self.port_to_slice = {1: {}, 2: {}}
        self.slice_to_dpid = {}
        self.dpid_to_slice = {}
        for i in range(self.SliceNum):
            self.slice_to_dstport[1][i] = i + self.SliceNum_to_s1_Slice0Port
            self.slice_to_srcport[1][i] = i + 1
            self.slice_to_dstport[2][i] = i + 1
            self.slice_to_srcport[2][i] = i + self.SliceNum_to_s2_Slice0Port
            self.port_to_slice[1][i + 1] = i
            self.port_to_slice[2][i + 1] = i
            self.port_to_slice[1][i + self.SliceNum_to_s1_Slice0Port] = i
            self.port_to_slice[2][i + self.SliceNum_to_s2_Slice0Port] = i
            self.slice_to_dpid[i] = i + 3
            self.dpid_to_slice[i + 3] = i

        #monitor
        self.sleep_period = SLEEP_PERIOD
        if self.Monitor_ctrl == 1:
            self.monitor_thread = hub.spawn(self._monitor)
        self.moniter_record = {
            'prev_tx_bytes': {1: {}, 2: {}},
            'prev_rx_bytes': {1: {}, 2: {}},
            'prev_tx_packets': {1: {}, 2: {}},
            'prev_rx_packets': {1: {}, 2: {}},
            'Tx_flow': {1: {4294967294: 0}, 2: {4294967294: 0}, },
            'Rx_flow': {1: {4294967294: 0}, 2: {4294967294: 0}, },
        }
        for dpid in range(1, 2 + 1):
            for i in range(2 * self.SliceNum + 1):
                self.moniter_record['prev_tx_bytes'][dpid][i] = 0
                self.moniter_record['prev_rx_bytes'][dpid][i] = 0
                self.moniter_record['prev_tx_packets'][dpid][i] = 0
                self.moniter_record['prev_rx_packets'][dpid][i] = 0
                self.moniter_record['Tx_flow'][dpid][i] = 0
                self.moniter_record['Rx_flow'][dpid][i] = 0

        self.bandwidth = BUDGET_BW
        self.bandwidth[-1] = 1
        self.bandfree = {1: {}, 2: {}}
        self.bandfree[1] = {i: BUDGET_BW[i] for i in range(self.SliceNum)}
        self.bandfree[2] = {i: BUDGET_BW[i] for i in range(self.SliceNum)}
        self.bandpktsize = orig_BUDGET_PKT_SIZE
        self.bandpktsize[-1] = 60

        #latency
        self.innerdelay = {i: 0 for i in range(self.SwitchTotal + 1)}
        self.ping_monitor_timestamp = {}
        self.ping_req_timestamp = {}
        self.ping_reqin_timestamp = {}
        self.ping_rly_timestamp = {}
        self.ping_rlyin_timestamp = {}
        self.reqecho_timestamp = {}
        self.latency = {1: {}, 2: {}}
        self.latency[1] = {i: 0 for i in range(self.SliceNum)}
        self.latency[2] = {i: 0 for i in range(self.SliceNum)}

        self.flow_dynamic = {}

        # dict_to_outport
        # MAC
        self.dstmac_to_port = {}
        self.srcmac_to_port = {}
        #s1
        self.dstmac_to_port[1] = {}
        self.srcmac_to_port[1] = {}
        for i in range(1, int(self.Host1half + 1)):
            port_s1_left = i
            port_s1_right = i + self.Host1half
            host_1half_i = i
            host_1half_MAC = "00:00:00:00:00:" + str(
                hex(host_1half_i).lstrip("0x")).zfill(2)
            self.dstmac_to_port[1][host_1half_MAC] = port_s1_left
            self.srcmac_to_port[1][host_1half_MAC] = port_s1_right
        for i in range(self.Host1half + 1, int(self.HostTotal + 1)):
            port_s1_left = i - self.Host1half
            port_s1_right = i
            host_2half_i = i
            host_2half_MAC = "00:00:00:00:00:" + str(
                hex(host_2half_i).lstrip("0x")).zfill(2)
            self.dstmac_to_port[1][host_2half_MAC] = port_s1_right
            self.srcmac_to_port[1][host_2half_MAC] = port_s1_left
        #s2
        self.dstmac_to_port[2] = {}
        self.srcmac_to_port[2] = {}
        for i in range(1, int(self.Host1half + 1)):
            port_s2_left = i + self.Host2half
            port_s2_right = i
            host_1half_i = i
            host_1half_MAC = "00:00:00:00:00:" + str(
                hex(host_1half_i).lstrip("0x")).zfill(2)
            self.dstmac_to_port[2][host_1half_MAC] = port_s2_left
            self.srcmac_to_port[2][host_1half_MAC] = port_s2_right
        for i in range(self.Host1half + 1, int(self.HostTotal + 1)):
            port_s2_left = i
            port_s2_right = i - self.Host2half
            host_2half_i = i
            host_2half_MAC = "00:00:00:00:00:" + str(
                hex(host_2half_i).lstrip("0x")).zfill(2)
            self.dstmac_to_port[2][host_2half_MAC] = port_s2_right
            self.srcmac_to_port[2][host_2half_MAC] = port_s2_left
        #switch link switch
        for j in range(1 + 2, self.SliceNum + 2 + 1):
            self.dstmac_to_port[j] = {}
            self.srcmac_to_port[j] = {}
            for i in range(1, int(self.Host1half + 1)):
                host_1half_i = i
                host_1half_MAC = "00:00:00:00:00:" + str(
                    hex(host_1half_i).lstrip("0x")).zfill(2)
                self.dstmac_to_port[j][host_1half_MAC] = 1
                self.srcmac_to_port[j][host_1half_MAC] = 2
            for i in range(self.Host1half + 1, int(self.HostTotal + 1)):
                host_2half_i = i
                host_2half_MAC = "00:00:00:00:00:" + str(
                    hex(host_2half_i).lstrip("0x")).zfill(2)
                self.dstmac_to_port[j][host_2half_MAC] = 2
                self.srcmac_to_port[j][host_2half_MAC] = 1

        # dict_to_outport
        # IP
        self.dstipv4_to_port = {}
        self.srcipv4_to_port = {}
        #s1
        self.dstipv4_to_port[1] = {}
        self.srcipv4_to_port[1] = {}
        for i in range(1, int(self.Host1half + 1)):
            port_s1_left = i
            port_s1_right = i + self.Host1half
            host_1half_i = i
            host_1half_IP = "10.0.0." + str(host_1half_i).zfill(1)
            self.dstipv4_to_port[1][host_1half_IP] = port_s1_left
            self.srcipv4_to_port[1][host_1half_IP] = port_s1_right
        for i in range(self.Host1half + 1, int(self.HostTotal + 1)):
            port_s1_left = i - self.Host1half
            port_s1_right = i
            host_2half_i = i
            host_2half_IP = "10.0.0." + str(host_2half_i).zfill(1)
            self.dstipv4_to_port[1][host_2half_IP] = port_s1_right
            self.srcipv4_to_port[1][host_2half_IP] = port_s1_left
        #s2
        self.dstipv4_to_port[2] = {}
        self.srcipv4_to_port[2] = {}
        for i in range(1, int(self.Host1half + 1)):
            port_s2_left = i + self.Host2half
            port_s2_right = i
            host_1half_i = i
            host_1half_IP = "10.0.0." + str(host_1half_i).zfill(1)
            self.dstipv4_to_port[2][host_1half_IP] = port_s2_left
            self.srcipv4_to_port[2][host_1half_IP] = port_s2_right
        for i in range(self.Host1half + 1, int(self.HostTotal + 1)):
            port_s2_left = i
            port_s2_right = i - self.Host2half
            host_2half_i = i
            host_2half_IP = "10.0.0." + str(host_2half_i).zfill(1)
            self.dstipv4_to_port[2][host_2half_IP] = port_s2_right
            self.srcipv4_to_port[2][host_2half_IP] = port_s2_left
        #switch link switch
        for j in range(1 + 2, self.SliceNum + 2 + 1):
            self.dstipv4_to_port[j] = {}
            self.srcipv4_to_port[j] = {}
            for i in range(1, int(self.Host1half + 1)):
                host_1half_i = i
                host_1half_IP = "10.0.0." + str(host_1half_i).zfill(1)
                self.dstipv4_to_port[j][host_1half_IP] = 1
                self.srcipv4_to_port[j][host_1half_IP] = 2
            for i in range(self.Host1half + 1, int(self.HostTotal + 1)):
                host_2half_i = i
                host_2half_IP = "10.0.0." + str(host_2half_i).zfill(1)
                self.dstipv4_to_port[j][host_2half_IP] = 2
                self.srcipv4_to_port[j][host_2half_IP] = 1

        # dict_to_outport
        # PORT
        self.outport_to_port = {}
        self.inport_to_port = {}
        #s1
        self.outport_to_port[1] = {}
        self.inport_to_port[1] = {}
        for i in range(1, int(self.Host1half + 1)):
            port_s1_left = i
            port_s1_right = i + self.Host1half
            host_1half_i = i
            host_1half_PORT = str(host_1half_i).zfill(1)
            self.outport_to_port[1][host_1half_PORT] = port_s1_left
            self.inport_to_port[1][host_1half_PORT] = port_s1_right
        for i in range(self.Host1half + 1, int(self.HostTotal + 1)):
            port_s1_left = i - self.Host1half
            port_s1_right = i
            host_2half_i = i
            host_2half_PORT = str(host_2half_i).zfill(1)
            self.outport_to_port[1][host_2half_PORT] = port_s1_right
            self.inport_to_port[1][host_2half_PORT] = port_s1_left
        #s2
        self.outport_to_port[2] = {}
        self.inport_to_port[2] = {}
        for i in range(1, int(self.Host1half + 1)):
            port_s2_left = i + self.Host2half
            port_s2_right = i
            host_1half_i = i
            host_1half_PORT = str(host_1half_i).zfill(1)
            self.outport_to_port[2][host_1half_PORT] = port_s2_left
            self.inport_to_port[2][host_1half_PORT] = port_s2_right
        for i in range(self.Host1half + 1, int(self.HostTotal + 1)):
            port_s2_left = i
            port_s2_right = i - self.Host2half
            host_2half_i = i
            host_2half_PORT = str(host_2half_i).zfill(1)
            self.outport_to_port[2][host_2half_PORT] = port_s2_right
            self.inport_to_port[2][host_2half_PORT] = port_s2_left
        #switch link switch
        for j in range(1 + 2, self.SliceNum + 2 + 1):
            self.outport_to_port[j] = {}
            self.inport_to_port[j] = {}
            for i in range(1, int(self.Host1half + 1)):
                host_1half_i = i
                host_1half_PORT = str(host_1half_i).zfill(1)
                self.outport_to_port[j][host_1half_PORT] = 1
                self.inport_to_port[j][host_1half_PORT] = 2
            for i in range(self.Host1half + 1, int(self.HostTotal + 1)):
                host_2half_i = i
                host_2half_PORT = str(host_2half_i).zfill(1)
                self.outport_to_port[j][host_2half_PORT] = 2
                self.inport_to_port[j][host_2half_PORT] = 1

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        hard_timeout = 0
        self.add_flow(datapath=datapath,
                      priority=0,
                      match=match,
                      actions=actions,
                      hard_timeout=hard_timeout)

    def add_flow(self, datapath, priority, match, actions, hard_timeout,
                 *buffer_id):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath,
                                    command=datapath.ofproto.OFPFC_ADD,
                                    hard_timeout=hard_timeout,
                                    priority=priority,
                                    buffer_id=buffer_id,
                                    match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath,
                                    hard_timeout=hard_timeout,
                                    priority=priority,
                                    match=match,
                                    instructions=inst)
        datapath.send_msg(mod)

    def _send_package(self, msg, datapath, in_port, actions):
        data = None
        ofproto = datapath.ofproto

        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        out = datapath.ofproto_parser.OFPPacketOut(datapath=datapath,
                                                   buffer_id=msg.buffer_id,
                                                   in_port=in_port,
                                                   actions=actions,
                                                   data=data)
        datapath.send_msg(out)

    #different switch to lowload slice
    def _out_port_group(self, dpid, out_port, class_result):
        slice_num = class_result
        if self.Scheuduler_ctrl == 0:
            return out_port
        elif class_result == -1:
            return out_port

        elif dpid == 1 and out_port >= self.slice_to_dstport[dpid][0] and out_port <= self.slice_to_dstport[dpid][self.SliceNum - 1]:
            flow = {i: self.class_count[i] for i in self.class_count if i != -1}
            latency = self.latency[dpid]
            bandfree = self.bandfree[dpid]

            print(f'bandfree s{dpid}: {bandfree}')

            if self.Scheuduler_ctrl == 1 or self.bandfree[dpid][class_result] > self.bandpktsize[class_result]:
                slice_num = class_result
            elif self.Scheuduler_ctrl == "random":
                slice_num = ryu_scheduler.random_algo(class_result, latency, bandfree, flow)
            elif self.Scheuduler_ctrl == "MAX":
                slice_num = ryu_scheduler.MAX_algo(class_result, latency, bandfree, flow)
            elif self.Scheuduler_ctrl == "min":
                slice_num = ryu_scheduler.min_algo(class_result, latency, bandfree, flow)
            elif self.Scheuduler_ctrl == "algo":
                slice_num = ryu_scheduler.scheduler_algo(class_result, latency, bandfree, flow)
            else:
                slice_num = class_result

            out_port = self.slice_to_dstport[dpid][slice_num]

        #1src
        elif dpid == 1 and out_port >= self.slice_to_srcport[dpid][0] and out_port <= self.slice_to_srcport[dpid][self.SliceNum - 1]:
            return out_port

        #2dst
        elif dpid == 2 and out_port >= self.slice_to_dstport[dpid][0] and out_port <= self.slice_to_dstport[dpid][self.SliceNum - 1]:
            return out_port

        elif dpid == 2 and out_port >= self.slice_to_srcport[dpid][0] and out_port <= self.slice_to_srcport[dpid][self.SliceNum - 1]:
            flow = {i: self.class_count[i] for i in self.class_count if i != -1}
            latency = self.latency[dpid]
            bandfree = self.bandfree[dpid]

            print(f'bandfree s{dpid}: {bandfree}')

            if self.Scheuduler_ctrl == 1 or self.bandfree[dpid][class_result] > self.bandpktsize[class_result]:
                slice_num = class_result
            elif self.Scheuduler_ctrl == "random":
                slice_num = ryu_scheduler.random_algo(class_result, latency, bandfree, flow)
            elif self.Scheuduler_ctrl == "MAX":
                slice_num = ryu_scheduler.MAX_algo(class_result, latency, bandfree, flow)
            elif self.Scheuduler_ctrl == "min":
                slice_num = ryu_scheduler.min_algo(class_result, latency, bandfree, flow)
            elif self.Scheuduler_ctrl == "algo":
                slice_num = ryu_scheduler.scheduler_algo(class_result, latency, bandfree, flow)
            else:
                slice_num = class_result

            out_port = self.slice_to_srcport[dpid][slice_num]

        #concume avaliable slice
        self.bandfree[dpid][slice_num] -= self.bandpktsize[class_result]
        print(f'concume {self.bandpktsize[class_result]}')
        print(f'bandfree s{dpid}: {self.bandfree[dpid]}')

        if self.ScheudulerPrint_ctrl == 1:
            print(f'controller s{dpid} outport={out_port}')

        return out_port

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # If you hit this you might want to increase
        # the "miss_send_length" of your switch
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only {} of {} bytes",
                              ev.msg.msg_len, ev.msg.total_len)
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id
        pkt = packet.Packet(msg.data)

        #need L3 L4 to classifier if not will valueerror
        L3_ctrl = 0
        L4_ctrl = 0
        ICMP_ctrl = 1

        try:
            id_eth = pkt.get_protocols(ethernet.ethernet)[0]
            eth_dst = id_eth.dst
            eth_src = id_eth.src
        except:
            eth_dst = 'ff:ff:ff:ff:ff:ff'
            eth_src = 'ff:ff:ff:ff:ff:ff'

        while L3_ctrl == 0:
            try:
                id_ipv6 = pkt.get_protocols(ipv6.ipv6)[0]
                ipv6_dst = id_ipv6.dst
                ipv6_src = id_ipv6.src
                #for scheduler
                self.total_length = id_ipv6.total_length
                ip_dst = ipv6_dst
                ip_src = ipv6_src
                #for classifier
                L3_ctrl = 1
                break
            except:
                ipv6_dst = '0:0:0:0:0:0:0:0'
                ipv6_src = '0:0:0:0:0:0:0:0'
                ip_dst = ipv6_dst
                ip_src = ipv6_src
            try:
                id_ipv4 = pkt.get_protocols(ipv4.ipv4)[0]
                ipv4_dst = id_ipv4.dst
                ipv4_src = id_ipv4.src
                #for scheduler
                self.total_length = id_ipv4.total_length
                ip_dst = ipv4_dst
                ip_src = ipv4_src
                arp_dst = ipv4_dst
                arp_src = ipv4_src

                #L4 protocol for mapping
                ip_proto = id_ipv4.proto

                #for classifier
                L3_ctrl = 1
                break
            except:
                ipv4_dst = '0.0.0.0'
                ipv4_src = '0.0.0.0'
                ip_dst = ipv4_dst
                ip_src = ipv4_src
                arp_dst = ipv4_dst
                arp_src = ipv4_src
            break

        while L4_ctrl == 0:
            try:
                id_tcp = pkt.get_protocols(tcp.tcp)[0]
                tcp_src = id_tcp.src_port
                tcp_dst = id_tcp.dst_port
                #for scheduler
                src_port = tcp_src
                dst_port = tcp_dst
                #for classifier
                L4_ctrl = 1
                break
            except:
                tcp_src = 0
                tcp_dst = 0
                src_port = tcp_src
                dst_port = tcp_dst
            try:
                id_udp = pkt.get_protocols(udp.udp)[0]
                udp_src = id_udp.src_port
                udp_dst = id_udp.dst_port
                #for scheduler
                src_port = udp_src
                dst_port = udp_dst
                #for classifier
                L4_ctrl = 1
                break
            except:
                udp_src = 0
                udp_dst = 0
                src_port = udp_src
                dst_port = udp_dst
            break

        self.packet_count += 1
        if self.AllPacketInfo_ctrl == 1:
            self.logger.info("---------------------------------------------------------------------------------------")
            self.logger.info(
                'Count switch in_port eth_src           eth_dst           ip_src         ip_dst'
            )
            self.logger.info(
                '{0:>5} {1:>6} {2:>7} {3:>17} {4:>17} {5:<8}:{6:>5} {7:<8}:{8:>5}'
                .format(self.packet_count, dpid, in_port, eth_src, eth_dst, ip_src, src_port, ip_dst, dst_port))
            self.logger.info("---------------------------------------------------------------------------------------")

        #monitor ping
        if self.Latency_ctrl == 1:
            try:
                id_icmp = pkt.get_protocols(icmp.icmp)[0]
                if id_icmp.type == icmp.ICMP_ECHO_REQUEST:
                    if dpid >= self.slice_to_dpid[0] and dpid <= self.slice_to_dpid[self.SliceNum - 1]:
                        ping_id = id_icmp.data.id
                        self.ping_reqin_timestamp[dpid] = time.time()

                        #reply
                        echo = id_icmp.data
                        echo.data = bytearray(echo.data)
                        data = self._ping_reply(src_dpid=dpid, echo=echo)
                        self._send_ping(src_dpid=dpid, dst_dpid=1, data=data)
                        self.ping_rly_timestamp[dpid] = time.time()

                        innerdelay = self.innerdelay[1] + self.innerdelay[ping_id]
                        monitor = self.ping_monitor_timestamp[ping_id]

                        req = self.ping_req_timestamp[ping_id]
                        reqin = self.ping_reqin_timestamp[ping_id]
                        latency_req = reqin - req - innerdelay
                        self.latency[1][self.dpid_to_slice[ping_id]] = latency_req

                        #for classifier
                        ICMP_ctrl = 0

                elif id_icmp.type == icmp.ICMP_ECHO_REPLY:
                    ping_id = id_icmp.data.id
                    self.ping_rlyin_timestamp[ping_id] = time.time()

                    innerdelay = self.innerdelay[1] + self.innerdelay[ping_id]
                    monitor = self.ping_monitor_timestamp[ping_id]

                    rly = self.ping_rly_timestamp[ping_id]
                    rlyin = self.ping_rlyin_timestamp[ping_id]
                    latency_rly = rlyin - rly - innerdelay
                    self.latency[2][self.dpid_to_slice[ping_id]] = latency_rly

                    avg = (self.latency[1][self.dpid_to_slice[ping_id]] +
                           self.latency[2][self.dpid_to_slice[ping_id]]) / 2

                    #for classifier
                    ICMP_ctrl = 0

                    if self.LatencyPrint_ctrl == 1:
                        loadbar = str('')
                        for i in range(int(avg) * 5):
                            loadbar = loadbar + '■'
                        self.logger.info("---------------------------------------------------------------------------------------")
                        self.logger.info(f'{id} avg latency   request    reply     ')
                        self.logger.info(
                            '{0:<4.3f}{1:<10} {2:<10.3f} {3:<10.3f}'.format(
                                avg, loadbar,
                                self.latency[1][self.dpid_to_slice[ping_id]],
                                self.latency[2][self.dpid_to_slice[ping_id]]))
                        self.logger.info("---------------------------------------------------------------------------------------")
                    """
                    self.innerdelay[ping_id]=0
                    self.ping_monitor_timestamp[ping_id]=0
                    self.ping_req_timestamp[ping_id]=0
                    self.ping_reqin_timestamp[ping_id]=0
                    self.ping_rly_timestamp[ping_id]=0
                    self.ping_rlyin_timestamp[ping_id]=0
                    """
            except:
                pass

        #preprocess
        open('mypcap.pcap', 'wb').close()
        self.pcap_writer = pcaplib.Writer(open('mypcap.pcap', 'wb'),
                                          snaplen=80)
        self.pcap_writer.write_pkt(ev.msg.data)
        X_test = ryu_preprocessing.transform_pcap('mypcap.pcap')

        #classifier
        class_result = -1
        if L3_ctrl == 1 and L4_ctrl == 1 and ICMP_ctrl == 1:
            try:
                #for classifier
                if self.Classifier_ctrl == 1:
                    app_result = self.loaded_model.predict(X_test)
                    class_result = self.app_to_service[app_result]
                else:
                    if ip_src.startswith('10.0.0.'):
                        start = (1 * (len('10.0.0.')))
                        i = int(ip_src[start:])
                        if i > self.SliceNum:
                            i = i - self.SliceNum
                        slicei = i - 1
                        class_result = int(PKT_FILE_MAP[slicei])
                #for scheduler
                self.class_count[class_result] += 1
                if self.ClassPrint_ctrl == 1:
                    print(f'class={self.service_to_string[class_result]} count={self.class_count[class_result]}')
            except:
                class_result = -1
                self.class_count[class_result] += 1

        #avoid mistake for next time classifier
        L3_ctrl = 0
        L4_ctrl = 0
        ICMP_ctrl = 1

        #mapping dict_to_port dst src

        if dpid in self.dstipv4_to_port and ipv4_dst in self.dstipv4_to_port[dpid]:
            #out_port
            out_port = self.dstipv4_to_port[dpid][ipv4_dst]
            if dpid == 1 or dpid == 2:
                out_port = self._out_port_group(dpid, out_port, class_result)
            if self.ActionPrint_ctrl == 1:
                self.logger.info(f"dst ip    s{dpid:<2}(out={out_port:>2})")
            #match
            if self.FlowMatch_ctrl == 1:
                if tcp_dst:
                    match = datapath.ofproto_parser.OFPMatch(eth_type=0x0800,
                                                             ipv4_dst=ipv4_dst,
                                                             ip_proto=ip_proto,
                                                             tcp_dst=tcp_dst)
                elif udp_dst:
                    match = datapath.ofproto_parser.OFPMatch(eth_type=0x0800,
                                                             ipv4_dst=ipv4_dst,
                                                             ip_proto=ip_proto,
                                                             udp_dst=udp_dst)
                else:
                    match = datapath.ofproto_parser.OFPMatch(eth_type=0x0800,
                                                             ipv4_dst=ipv4_dst)
            else:
                match = datapath.ofproto_parser.OFPMatch(eth_type=0x0800,
                                                         ipv4_dst=ipv4_dst)

            actions = [datapath.ofproto_parser.OFPActionOutput(port=out_port)]
            self.add_flow(datapath=datapath,
                          priority=1,
                          match=match,
                          actions=actions,
                          hard_timeout=self.hard_timeout[class_result])
            self._send_package(msg, datapath, in_port, actions)
        elif dpid in self.dstmac_to_port and eth_dst in self.dstmac_to_port[dpid]:
            out_port = self.dstmac_to_port[dpid][eth_dst]
            if dpid == 1 or dpid == 2:
                out_port = self._out_port_group(dpid, out_port, class_result)
            if self.ActionPrint_ctrl == 1:
                self.logger.info(f"dst mac    s{dpid:<2}(out={out_port:>2})")
            match = datapath.ofproto_parser.OFPMatch(eth_dst=eth_dst)
            actions = [datapath.ofproto_parser.OFPActionOutput(port=out_port)]
            self.add_flow(datapath=datapath,
                          priority=1,
                          match=match,
                          actions=actions,
                          hard_timeout=self.hard_timeout[class_result])
            self._send_package(msg, datapath, in_port, actions)
        elif dpid in self.srcipv4_to_port and ipv4_src in self.srcipv4_to_port[dpid]:
            #out_port
            out_port = self.srcipv4_to_port[dpid][ipv4_src]
            if dpid == 1 or dpid == 2:
                out_port = self._out_port_group(dpid, out_port, class_result)
            if self.ActionPrint_ctrl == 1:
                self.logger.info(f"src ip    s{dpid:<2}(out={out_port:>2})")
            #match
            if self.FlowMatch_ctrl == 1:
                if tcp_src:
                    match = datapath.ofproto_parser.OFPMatch(eth_type=0x0800,
                                                             ipv4_src=ipv4_src,
                                                             ip_proto=ip_proto,
                                                             tcp_src=tcp_src)
                elif udp_src:
                    match = datapath.ofproto_parser.OFPMatch(eth_type=0x0800,
                                                             ipv4_src=ipv4_src,
                                                             ip_proto=ip_proto,
                                                             udp_src=udp_src)
                else:
                    match = datapath.ofproto_parser.OFPMatch(eth_type=0x0800,
                                                             ipv4_src=ipv4_src)
            else:
                match = datapath.ofproto_parser.OFPMatch(eth_type=0x0800,
                                                         ipv4_src=ipv4_src)

            actions = [datapath.ofproto_parser.OFPActionOutput(port=out_port)]
            self.add_flow(datapath=datapath,
                          priority=1,
                          match=match,
                          actions=actions,
                          hard_timeout=self.hard_timeout[class_result])
            self._send_package(msg, datapath, in_port, actions)
        elif dpid in self.srcmac_to_port and eth_src in self.srcmac_to_port[dpid]:
            out_port = self.srcmac_to_port[dpid][eth_src]
            if dpid == 1 or dpid == 2:
                out_port = self._out_port_group(dpid, out_port, class_result)
            if self.ActionPrint_ctrl == 1:
                self.logger.info(f"src mac    s{dpid:<2}(out={out_port:>2})")
            match = datapath.ofproto_parser.OFPMatch(eth_src=eth_src)
            actions = [datapath.ofproto_parser.OFPActionOutput(port=out_port)]
            self.add_flow(datapath=datapath,
                          priority=1,
                          match=match,
                          actions=actions,
                          hard_timeout=self.hard_timeout[class_result])
            self._send_package(msg, datapath, in_port, actions)
        elif dpid in self.inport_to_port and in_port in self.inport_to_port[dpid]:
            out_port = self.inport_to_port[dpid][in_port]
            if dpid == 1 or dpid == 2:
                out_port = self._out_port_group(dpid, out_port, class_result)
            if self.ActionPrint_ctrl == 1:
                self.logger.info(f"src port    s{dpid:<2}(out={out_port:>2})")
            match = datapath.ofproto_parser.OFPMatch(in_port=in_port)
            actions = [datapath.ofproto_parser.OFPActionOutput(port=out_port)]
            self.add_flow(datapath=datapath,
                          priority=1,
                          match=match,
                          actions=actions,
                          hard_timeout=self.hard_timeout[class_result])
            self._send_package(msg, datapath, in_port, actions)

    #monitor
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    """
    -controller-
    |          |
    D1         D2
    |          |
    S1-latency-S3

    #echo = controller -> Sa;Sb -> controller
    #innerdelay D1;D2 = echo /2
    #ping = Controller -> Sa -> Sb -> Controller
    #latency = ping - D1 - D2
    """

    #all monitor request here
    def _monitor(self):
        while True:
            for dpid, datapath in self.datapaths.items():
                self._request_stats(datapath)

                if self.Latency_ctrl == 1:
                    #for innerdelay
                    self._echo_request(datapath)

                    #for latency
                    self.ping_monitor_timestamp[dpid] = time.time()
                    if dpid >= self.slice_to_dpid[0] and dpid <= self.slice_to_dpid[self.SliceNum - 1]:
                        echo = icmp.echo(id_=dpid, seq=1)
                        data = self._ping_request(dst_dpid=dpid, echo=echo)
                        self._send_ping(src_dpid=1, dst_dpid=dpid, data=data)
                        self.ping_req_timestamp[dpid] = time.time()
            hub.sleep(self.sleep_period)

    #stats request
    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    #innerdelay request
    def _echo_request(self, datapath):
        ofp_parser = datapath.ofproto_parser
        dpid = datapath.id
        req = ofp_parser.OFPEchoRequest(datapath)
        datapath.send_msg(req)
        self.reqecho_timestamp[dpid] = time.time()

    #innerdelay end
    @set_ev_cls(ofp_event.EventOFPEchoReply, [HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER, MAIN_DISPATCHER])
    def _echo_reply_handler(self, ev):
        timestamp_reply = time.time()
        dpid = ev.msg.datapath.id
        self.innerdelay[dpid] = (timestamp_reply - self.reqecho_timestamp[dpid]) / 2

    #ping
    def _build_ping(self, dpid, _type, echo):
        if _type == icmp.ICMP_ECHO_REQUEST:
            src_port = dpid - 1
            dst_switch = dpid
            eth_src = str(self.datapaths[1].ports[src_port].hw_addr)
            eth_dst = str(self.datapaths[dst_switch].ports[1].hw_addr)
            ip_src = int(netaddr.IPAddress('0.0.0.0'))
            ip_dst = int(netaddr.IPAddress('0.0.0.0'))
        elif _type == icmp.ICMP_ECHO_REPLY:
            src_switch = dpid
            dst_port = dpid - 1
            eth_src = str(self.datapaths[src_switch].ports[1].hw_addr)
            eth_dst = str(self.datapaths[1].ports[dst_port].hw_addr)
            ip_src = int(netaddr.IPAddress('0.0.0.0'))
            ip_dst = int(netaddr.IPAddress('0.0.0.0'))
        e = ethernet.ethernet(dst=eth_src,
                              src=eth_dst,
                              ethertype=ether.ETH_TYPE_IP)
        ip = ipv4.ipv4(version=4,
                       header_length=5,
                       tos=0,
                       total_length=84,
                       identification=0,
                       flags=0,
                       offset=0,
                       ttl=64,
                       proto=inet.IPPROTO_ICMP,
                       csum=0,
                       src=ip_src,
                       dst=ip_dst)
        i = icmp.icmp(type_=_type, code=0, csum=0, data=echo)
        p = packet.Packet()
        p.add_protocol(e)
        p.add_protocol(ip)
        p.add_protocol(i)
        p.serialize()
        return p

    def _send_ping(self, src_dpid, dst_dpid, data):
        datapath = self.datapaths[src_dpid]
        if src_dpid == 1:
            out_port = self.slice_to_dstport[1][self.dpid_to_slice[dst_dpid]]
        elif src_dpid >= self.slice_to_dpid[0] and src_dpid <= self.slice_to_dpid[self.SliceNum - 1]:
            out_port = 1
        buffer_id = 0xffffffff
        in_port = datapath.ofproto.OFPP_CONTROLLER
        actions = [datapath.ofproto_parser.OFPActionOutput(port=out_port, max_len=0)]
        msg = datapath.ofproto_parser.OFPPacketOut(datapath=datapath,
                                                   buffer_id=buffer_id,
                                                   in_port=in_port,
                                                   actions=actions,
                                                   data=data)
        datapath.send_msg(msg)

    def _ping_request(self, dst_dpid, echo):
        p = self._build_ping(dst_dpid, icmp.ICMP_ECHO_REQUEST, echo)
        return p.data

    def _ping_reply(self, src_dpid, echo):
        p = self._build_ping(src_dpid, icmp.ICMP_ECHO_REPLY, echo)
        return p.data

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        #monitor slice update
        if dpid == 1 or dpid == 2:
            for stat in sorted(body, key=attrgetter('port_no')):
                if stat.port_no <= 2 * self.SliceNum:
                    currtx = stat.tx_bytes
                    prevtx = self.moniter_record['prev_tx_bytes'][dpid][stat.port_no]
                    tx_bytes = currtx - prevtx

                    self.moniter_record['prev_tx_bytes'][dpid][stat.port_no] = currtx
                    self.moniter_record['Tx_flow'][dpid][stat.port_no] = tx_bytes

                    currrx = stat.rx_bytes
                    prevrx = self.moniter_record['prev_rx_bytes'][dpid][stat.port_no]
                    rx_bytes = currrx - prevrx

                    self.moniter_record['prev_rx_bytes'][dpid][stat.port_no] = currrx
                    self.moniter_record['Rx_flow'][dpid][stat.port_no] = rx_bytes

                    if self.UpdateBudget_ctrl == 1:
                        self.bandfree[dpid] = {
                            i: self.bandwidth[i] - self.moniter_record['Tx_flow'][dpid][self.slice_to_dstport[dpid][i]] -
                            self.moniter_record['Rx_flow'][dpid][self.slice_to_dstport[dpid][i]]
                            for i in range(self.SliceNum)
                        }

        #monitor pkts bytes
        if self.MonitorPrint_ctrl == 1:
            if dpid == 1 or dpid == 2:
                self.logger.info('datapath         port     '
                                 'rx-pkts  rx-bytes rx-prev  '
                                 'tx-pkts  tx-bytes tx-prev  '
                                 'latency  est.free est.load')
                self.logger.info('---------------- -------- '
                                 '-------- -------- -------- '
                                 '-------- -------- -------- '
                                 '-------- -------- -------- ')
                for stat in sorted(body, key=attrgetter('port_no')):
                    self.duration_sec = stat.duration_sec
                    if dpid == 1 or dpid == 2:
                        if stat.port_no <= 2 * self.SliceNum:
                            latency = self.latency[dpid][self.port_to_slice[dpid][stat.port_no]]
                            bandfree = self.bandfree[dpid][self.port_to_slice[dpid][stat.port_no]]
                            bandwidth = self.bandwidth[self.port_to_slice[dpid][stat.port_no]]
                            bandload = self.moniter_record['Rx_flow'][dpid][stat.port_no] + self.moniter_record['Tx_flow'][dpid][stat.port_no]
                        else:
                            latency = 123.567
                            bandfree = 1
                            bandwidth = 1
                            bandload = bandwidth

                    bar = str('')
                    barlen = int(bandload / bandwidth * 11)
                    if barlen > 11:
                        barlen = 11
                    for i in range(barlen):
                        bar = bar + '■'

                    self.logger.info(
                        '%016x %8x %8d %8d %8d %8d %8d %8d %8.3f %8d %8d %s',
                        ev.msg.datapath.id, stat.port_no, 
                        stat.rx_packets, stat.rx_bytes, self.moniter_record['Rx_flow'][dpid][stat.port_no],
                        stat.tx_packets, stat.tx_bytes, self.moniter_record['Tx_flow'][dpid][stat.port_no], 
                        latency, bandfree, bandload, bar)

        #monitor record csv file
        csvtime = time.time()
        if dpid == 2 and csvtime >= GOGO_TIME and csvtime <= GOGO_TIME + TOTAL_TIME:
            with open(self.csv_throughput_record_file, 'a') as csv_file:
                row = [csvtime]
                for csvid in range(1, 2 + 1):
                    for csvno in range(1, self.SliceNum*2 + 1):
                        row.append(self.moniter_record['Rx_flow'][csvid][csvno])
                        row.append(self.moniter_record['Tx_flow'][csvid][csvno])
                writer = csv.writer(csv_file)
                writer.writerow(row)