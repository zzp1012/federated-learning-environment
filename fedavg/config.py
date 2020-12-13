# ************************************************************************************************************ #
import logging
import pandas as pd
import os
from tensorboardX import SummaryWriter

# set the requirement.
bandwith = 1
res_weight = 0.5
res_ratio = 0.1 # the ratio of radio_res

# the ratio of standalone training time over real distributed learning training time.
timing_ratio = 10

# set the data dir
channel_data_dir = "../data"
# read channel data at once.
channel_data = pd.concat([pd.read_csv(os.path.join(channel_data_dir, csv_name)) for csv_name in os.listdir(channel_data_dir)], ignore_index = True)

# restrict the number of client_num_in_total to maxmium car ID + 1
client_num_in_total = channel_data['Car'].max() + 1
client_num_per_round = 100 # number of local clients

# set the logger
logging.basicConfig(
                    # filename = "logfile",
                    # filemode = "w+",
                    format='%(name)s %(levelname)s %(message)s',
                    datefmt = "%H:%M:%S",
                    level=logging.DEBUG)
logger = logging.getLogger("training")

# setup the tensorboardX

boardX = SummaryWriter(comment="-fedavg")

# set hyperparameter for calculating FPF2 index
G1 = 2
G2 = 2
# ************************************************************************************************************ #