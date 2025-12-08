from threading import Thread
import uvicorn
from loguru import logger
from app import app as fastApp
import baseUtil
import random

def find_port():
    global port
    while True:
        try:
            port = baseUtil.find_port(port)
            logger.info(f"Uvicorn server port: {port}")
            break
        except Exception as e:
            logger.error(f"Exception Uvicorn server error repeat port: {port}")
            port = random.randint(18000, 20000)

def uvicornserver():
    global port
    uvicorn.run(fastApp, host="localhost", port=port)

def uvicornserverThread():
    uvth = Thread(target=uvicornserver)
    uvth.setDaemon(True)
    uvth.start()


port = 9000

if __name__ == '__main__':
    find_port()
    # uvicornserverThread()
    uvicornserver()