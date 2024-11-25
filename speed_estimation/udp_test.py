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


def send_udp_packet_json(data : Dict):
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    MESSAGE = json.dumps(dict)
    
    print(MESSAGE)
    
    sock = socket.socket(socket.AF_INET, # Internet
                 socket.SOCK_DGRAM) # UDP
    sock.sendto( bytes(MESSAGE, 'utf-8'), (UDP_IP, UDP_PORT))

dict = {
        "number_of_cars": "10",
        "positions": [],
        "time" : "14:00:00"
}
while True:
    send_udp_packet_json(dict)
