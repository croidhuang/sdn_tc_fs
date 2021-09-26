#sudo python3 mininet/custom/custom_example.py
#http://mininet.org/api/annotated.html


from exp_config.exp_config import MININET_BW
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.clean import Cleanup
from mininet.log import setLogLevel, info, error
from mininet.util import dumpNodeConnections

import sys
sys.path.insert(1, './')

SliceNum = 7
HostTotal = SliceNum*2
SwitchTotal = 1+SliceNum+1

bandwidth = MININET_BW

dpid_to_slice = {1: {}, 2: {}, }
dpid_to_slice[1] = {i: i-3 for i in range(3, 9+1)}
dpid_to_slice[2] = {i: i-3 for i in range(3, 9+1)}


def myNetwork():
    net = Mininet(
        topo=None,
        autoSetMacs=True,
        autoStaticArp=True,
        build=False)

    info('*** Add controller\n\"')
    net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)

    info('*** Add switches\n\"')
    SwitchList = [0]
    SwitchNum = [0]
    for i in range(1, SwitchTotal+1):
        SwitchList.append(i)
        SwitchNum.append('s'+str(i))
        SwitchList[i] = net.addSwitch(SwitchNum[i])

    info('*** Add hostes\n\"')
    HostList = [0]
    HostNum = [0]
    for i in range(1, HostTotal+1):
        HostList.append(i)
        HostNum.append('h'+str(i))
        HostList[i] = net.addHost(HostNum[i])

    info('*** Add linkes\n\"')
    #host link switch
    for i in range(1, int(HostTotal/2)+1):
        Host1half = i
        net.addLink(HostList[Host1half], SwitchList[1])
        Host2half = i+int(HostTotal/2)
        net.addLink(SwitchList[2], HostList[Host2half])

    #switch link switch
    for i in range(1+2, SliceNum+2+1):
        net.addLink(SwitchList[1], SwitchList[i], cls=TCLink, bw=bandwidth[dpid_to_slice[1][i]])
        net.addLink(SwitchList[i], SwitchList[2], cls=TCLink, bw=bandwidth[dpid_to_slice[2][i]])

    info('*** Start Network\n\"')
    net.start()

    info('*** OVS command\n\"')
    #net.get
    for i in range(1, SwitchTotal+1):
        SwitchList[i] = net.get(SwitchNum[i])

    for i in range(1, SwitchTotal+1):
        SwitchList[i].cmdPrint('ovs-ofctl dump-flows s'+str(i))

    info("***Dumping host connections\n")
    dumpNodeConnections(net.hosts)

    """
    #not work, need controller identify iperf
    info( "***Testing bandwidth\n" )
    testlist=[0]
    for i in range(1, int(HostTotal)+1):
    	testlist.append(i)

    for i in range(1, int(HostTotal/2)+1):        
        testlist[i],testlist[i+7] = net.get(('h%s' % (i+0)), ('h%s' % (i+7)))
    """

    """
    #not work, it will wait at h1
    for i in range(1, HostTotal+1):
        HostList[i] = net.get( HostNum[i])

    for i in range(1, HostTotal+1):
        if i <=HostTotal/2:
            HostList[i].cmdPrint("python3 pktgen/gen/client"+str(i)+".py")
        elif i>HostTotal/2:
            HostList[i].cmdPrint("python3 pktgen/gen/server"+str(i)+".py")
        else:
            break
    """
    CLI(net)

    Cleanup.cleanup()


if __name__ == '__main__':
    setLogLevel('info')
    myNetwork()
