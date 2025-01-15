import socket
import json
from typing import Dict

def send_udp_packet(data):
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    MESSAGE = f'{data[0]},{data[1]}'
    print(MESSAGE)
    sock = socket.socket(socket.AF_INET, # Internet
                 socket.SOCK_DGRAM) # UDP
    sock.sendto(bytes(MESSAGE, 'utf-8'), (UDP_IP, UDP_PORT))

def send_udp_packet_json(data):
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    MESSAGE = data
    
    sock = socket.socket(socket.AF_INET, # Internet
                 socket.SOCK_DGRAM) # UDP
    sock.sendto( bytes(MESSAGE, 'utf-8'), (UDP_IP, UDP_PORT))
# dict ={'number_of_cars': '8', 'positions': [[0.0, 2.816136712885105e-05], [0.00021833517999675836, 7.868069364220294e-11], [6.0791264094130416e-05, 1.6942665670287513e-11], [0.00017715581866326827, 5.7210744935098316e-11], [0.00013396707537970056, 3.643827629827223e-11], [0.0002340300087734201, 5.940103179173772e-11], [8.563005018287996e-05, 0.0002825511293819986], [0.00023641090909543738, 7.994269358503107e-11]], 'time': '12:55:01.591919'}
# dict = {'number_of_cars': '3', 'positions': [[0.0, 5.482727561088148e-05], [0.00020959967317411496, 6.380761361823622e-11], [8.468999040145916e-05, 0.0002737605976860233]], 'time': '12:50:10.866507'}


# while True:
#    send_udp_packet_json(dict)
