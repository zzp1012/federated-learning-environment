# ************************************************************************************************************ #
# import libraries
import re
import socket
import argparse
import numpy as np
import pandas as pd
import os
import logging

# set the logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('schedule')

# set the data dir & read the data
channel_data_dir = "../data"
channel_data = pd.concat([pd.read_csv(os.path.join(channel_data_dir, csv_name)) for csv_name in os.listdir(channel_data_dir)], ignore_index = True)

# used for round robin 
queue = []

def sch_random(round_idx, time_counter):
    # set the seed
    np.random.seed(round_idx)

    # random sample clients
    cars = list(channel_data[channel_data['Time'] == time_counter]['Car'])
    client_indexes = []
    if cars:
        client_indexes = list(np.random.choice(cars, max(int(len(cars) / 2), 1), replace=False).ravel())
    logger.info("client_indexes = " + str(client_indexes) + "; time_counter = " + str(time_counter))

    # random local iterations
    local_itr = np.random.randint(2) + 1

    s_client_indexes = str(list(client_indexes))[1:-1].replace(',', '')
    s_local_itr = str(local_itr)

    return s_client_indexes + "," + s_local_itr

def sch_channel(round_idx, time_counter):
    # set the seed
    np.random.seed(round_idx)

    # sample only based on channel condition
    curr_channel = channel_data[channel_data['Time'] == time_counter]
    curr_channel = curr_channel.sort_values(by=['Distance to BS(4982,905)'], ascending=True) # sort by the channel condition
    client_indexes = (curr_channel.loc[:int((len(curr_channel) + 1)/ 2), 'Car']).tolist()
    
    # random local iterations
    local_itr = np.random.randint(2) + 1
    logger.info("client_indexes = " + str(client_indexes) + "; time_counter = " + str(time_counter) + "; local_itr = " + str(local_itr))

    s_client_indexes = str(list(client_indexes))[1:-1].replace(',', '')
    s_local_itr = str(local_itr)
    return s_client_indexes + "," + s_local_itr


def sch_rrobin(round_idx, time_counter): 
    # set the seed
    np.random.seed(round_idx)
   
    cars = list(channel_data[channel_data['Time'] == time_counter]['Car']) 
    queue.extend(cars)
    client_indexes = []
    num_client = int(len(cars) / 2) + 1

    while len(client_indexes) < num_client:
        car_temp = queue.pop(0)
        if car_temp in cars:  # add the car exist in the current time
            client_indexes.append(car_temp)

    logger.info("client_indexes = " + str(client_indexes) + "; time_counter = " + str(time_counter))

    # random local iterations
    local_itr = np.random.randint(2) + 1

    s_client_indexes = str(list(client_indexes))[1:-1].replace(',', '')
    s_local_itr = str(local_itr)
    return s_client_indexes + "," + s_local_itr

def main():
    parser = argparse.ArgumentParser(description= "scheduler")
    parser.add_argument("-m", "--method",
                        type= str,
                        default="sch_random",
                        help="declare the benchmark methods you use")
    parser.add_argument("-v", "--verbose", 
                        action= "store_true", 
                        dest= "verbose", 
                        help= "enable debug info output")
    args = parser.parse_args()

    if not args.verbose:
        logger.setLevel(logging.INFO)
    logger.debug("--------DEBUG enviroment start--------")

    # set the scheduler method
    if args.method == "sch_random":
        scheduler = sch_random
    elif args.method == "sch_channel":
        scheduler = sch_channel
    elif args.method == "sch_rrobin":
        scheduler = sch_rrobin
    else:
        scheduler = sch_random
    logger.info("current scheduler method: {}".format(args.method))

    # setup the socket server
    host = socket.gethostname()
    server = socket.socket()
    server.bind((host, 8999)) # bind the server to port 8999. Before running the code, make sure there is no program running under this port.

    server.listen(5) # maximum connections 5.

    round_idx = 0

    # set the global time_counter
    time_counter = 0

    while True:
        client, addr = server.accept() # connected to client.
        logger.info("round " + str(round_idx) + " connected address: " + str(addr)) # print connection message.

        response = client.recv(1000000).decode()

        # initialization
        loss_locals = []
        FPF1_idx_lst = []
        
        if response != "nothing":
            response = response.split(',')
            if response[0] != '':
                time_counter = int(float(response[0]))
            if response[1] != '':
                loss_locals = [float(i) for i in response[1].split(' ')]
            if response[2] != '':
                FPF1_idx_lst = [float(i) for i in response[2].split(' ')]
        
        mes = scheduler(round_idx, time_counter)
        client.send(mes.encode()) # send the message to the connected client.

        client.close() # close the connection

        round_idx = round_idx + 1 # record the current round of aggregation.



if __name__ == "__main__":
    main()
# ************************************************************************************************************ #
