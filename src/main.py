import argparse
import os
from threading import Thread
import random

from loguru import logger
import baseUtil


def parse_args():
    parser = argparse.ArgumentParser(description="OmniExtract service")
    parser.add_argument(
        "--cache-for-optimization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache identical model calls during prompt optimization "
        "(default: enabled). Use --no-cache-for-optimization to disable.",
    )
    parser.add_argument(
        "--cache-for-other",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache identical model calls outside prompt optimization "
        "(default: disabled). Use --cache-for-other to enable.",
    )
    return parser.parse_args()


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
    args = parse_args()
    # DSPy response cache is configured via service startup flags. Expose them
    # through environment variables so model.py picks them up when building LMs.
    os.environ["OMNIEXTRACT_CACHE_FOR_OPTIMIZATION"] = str(args.cache_for_optimization).lower()
    os.environ["OMNIEXTRACT_CACHE_FOR_OTHER"] = str(args.cache_for_other).lower()
    logger.info(
        f"DSPy cache settings: cache_for_optimization={args.cache_for_optimization}, "
        f"cache_for_other={args.cache_for_other}"
    )

    # Import after the cache environment variables are set.
    import uvicorn
    from app import app as fastApp

    find_port()
    # uvicornserverThread()
    uvicornserver()
