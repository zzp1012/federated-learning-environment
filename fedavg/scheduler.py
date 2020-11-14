# ************************************************************************************************************ #
# import libraries
import socket
import sys
import numpy as np
import pandas as pd
import os
import logging

# set the logger
logging.basicConfig(level=logging.DEBUG)

# set the data dir & read the data
channel_data_dir = "../data"
channel_data = pd.concat([pd.read_csv(os.path.join(channel_data_dir, csv_name)) for csv_name in os.listdir(channel_data_dir)], ignore_index = True)

# set the global time_counter
time_counter = channel_data['Time'][0]

def scheduler(round_idx):
    # random sample clients
    cars = list(channel_data[channel_data['Time'] == time_counter]['Car'])
    client_indexes = list(np.random.choice(cars, max(int(len(cars) / 2), 1), replace=False))
    logging.info("client_indexes = " + str(client_indexes) + "; time_counter = " + str(time_counter))

    # random local iterations
    local_itr = np.random.randint(2) + 1

    s_client_indexes = str(list(client_indexes))[1:-1].replace(',', '')
    s_local_itr = str(local_itr)

    return s_client_indexes + "," + s_local_itr

def main():
    # setup the socket server
    host = socket.gethostname()
    server = socket.socket()
    server.bind((host, 8999)) # bind the server to port 8999. Before running the code, make sure there is no program running under this port.

    server.listen(5) # maximum connections 5.

    round_idx = 0

    while True:
        client, addr = server.accept() # connected to client.
        print("round " + str(round_idx) + " connected address: " + str(addr)) # print connection message.

        response = client.recv(1024).decode()
        
        if response != "nothing":
            time_counter = int(response)

        mes = scheduler(round_idx)
        client.send(mes.encode()) # send the message to the connected client.

        client.close() # close the connection

        round_idx = round_idx + 1 # record the current round of aggregation.



if __name__ == "__main__":
    main()
# ************************************************************************************************************ #
