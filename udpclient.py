import socket
import json

class UDPClient():
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.host = host
        self.port = port
    
    def send(self, dat):
        js  = json.dumps(dat)
        print("Length: %d, %s" % (len(js), js))
        self.sock.sendto(js.encode("UTF-8"), (self.host, self.port))