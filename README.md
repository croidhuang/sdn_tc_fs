# sdn_tc_fs

安裝mininet和ryu  
安裝python及所需的套件  

修改設定  
pg/gen/pkt_config/pkt_config.py  
修改產生封包的開始時間：timestring  
修改pcap路徑：PKT_FILE_LIST  
修改pcap對應的傳送間隔：PKT_FILE_MAP  
修改select函數：SCHEDULER_TYPE  
修改其他設定如執行時間及頻寬相關參數  
  
執行mininet  
sudo python3 mininet/custom/custom_example_7to7.py  
  
執行ryu
ryu-manager ryu/ryu/app/simple_switch_13_slice.py  
  
在mininet輸入  
mininet> xterm h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14  
  
在h1執行  
python3 pg/gen/client1.py  
在h2執行  
python3 pg/gen/client2.py  
在h3執行  
python3 pg/gen/client3.py  
...  
在h8執行  
python3 pg/gen/server8.py  
在h9執行  
python3 pg/gen/server9.py  
在h10執行  
python3 pg/gen/server10.py  
...  
  
等待讀取pcap直到出現ready  
  

執行完成  
monitor檔案儲存在home目錄  
