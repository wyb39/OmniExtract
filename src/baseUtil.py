import os
import socket
import logging

logging.basicConfig(level=logging.INFO)
def get_root_path():
    return os.path.join(os.path.dirname(__file__), '..')

def find_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1",port))
            return port



