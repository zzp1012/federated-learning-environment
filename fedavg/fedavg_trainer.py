# ************************************************************************************************************ # newly added libraries
import copy
import socket
import torch
import numpy as np
import wandb

from client import Client
from config import *
# ************************************************************************************************************ #

class FedAvgTrainer(object):
    def __init__(self, dataset, model, device, args):
        self.device = device
        self.args = args

# ************************************************************************************************************ #
        [client_num, train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.client_num = client_num # record the client number of the dataset
# ************************************************************************************************************ #
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.model_global = model
        self.model_global.train()

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict)
# ************************************************************************************************************ #
        # time counter starts from the first line
        self.time_counter = channel_data['Time'][0]
# ************************************************************************************************************ #

    def setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logger.debug("############setup_clients (START)#############")
        for client_idx in range(client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device)
            self.client_list.append(c)
        logger.debug("############setup_clients (END)#############")

# ************************************************************************************************************ # Simply setup a client to recieve the message from server.
    def client_sampling(self):
        # setup client
        hostname = socket.gethostname()
        client_socket = socket.socket()
        client_socket.connect((hostname, 8999))

        # send nothing
        client_socket.sendall("nothing".encode())

        # get response
        response = client_socket.recv(1024).decode().split(',')

        # close connection
        client_socket.close()

        # decode the message (string) into numbers and arraies.
        client_indexes = []
        local_itr = 1
        if response[0] != '':
            client_indexes = [int(float(i)) for i in response[0].split(' ')]
        if response[1] != '':
            local_itr = [int(float(i)) for i in response[1].split(' ')][0]

        return client_indexes, local_itr
# ************************************************************************************************************ #

# ************************************************************************************************************ #
    def feedback(self, loss_locals, FPF1_idx_lst, FPF2_idx_lst):
        # setup client
        hostname = socket.gethostname()
        client_socket = socket.socket()
        client_socket.connect((hostname, 8999))

        # send time_counter & local_losses & FPF1_index & FPF2_index
        message = str(self.time_counter) + ',' + str(loss_locals)[1:-1].replace(',', '') + ',' +  str(list(FPF1_idx_lst.ravel()))[1:-1].replace(',', '') + ',' +  str(list(FPF2_idx_lst.ravel()))[1:-1].replace(',', '')
        client_socket.sendall(message.encode())

        # get response
        client_socket.recv(1024)

        # close connection
        client_socket.close()
# ************************************************************************************************************ #

# ************************************************************************************************************ #
    def tx_time(self, client_indexes):
        if not client_indexes:
            return
        # read the channel condition for corresponding cars.
        channel_res = np.reshape(np.array(channel_data[channel_data['Time'] == self.time_counter * channel_data['Car'].isin(client_indexes)]["Distance to BS(4982,905)"]), (1, -1))
        logger.debug("channel_res: {}".format(channel_res))

        # linearly resolve the optimazation problem
        tmp_t = 1
        while np.sum(res_weight * channel_res * res_ratio / tmp_t) > 1:
            tmp_t += 1

        self.time_counter += tmp_t
        logger.debug("time_counter after tx_time: {}".format(self.time_counter))
# ************************************************************************************************************ #

    def train(self):
# ************************************************************************************************************ #
        # Initialize values
        local_itr_lst = np.zeros((1, self.args.comm_round)) # historical local iterations.
        client_selec_lst = np.zeros((self.args.comm_round, client_num_in_total)) # historical client selections.
        local_w_lst = [self.model_global.cpu().state_dict()] * client_num_in_total # maintain a lst for all clients to store local weights
        loss_locals = [] # initial a lst to store loss values
        FPF1_idx_lst = np.zeros((1, client_num_in_total)) # maintain a lst for FPF1 indexes
        FPF2_idx_lst = np.zeros((1, client_num_in_total)) # maintain a lst for FPF2 indexes
        # Initialize A for calculating FPF2 index
        A_mat = dict()
        for para in self.model_global.state_dict().keys():
            weight_shape = self.model_global.state_dict()[para].numpy().ravel().shape[0]
            A_mat[para] = np.ones(weight_shape) # initial the value of A with zero.
        G_mat = np.zeros((1, client_num_in_total)) # initial the value of G with zero
# ************************************************************************************************************ #
        for round_idx in range(self.args.comm_round):
            logger.info("################Communication round : {}".format(round_idx))
            logger.info("time_counter: {}".format(self.time_counter))

            self.model_global.train()
# ************************************************************************************************************ #
            # update time counter on scheduler
            self.feedback(loss_locals, FPF1_idx_lst, FPF2_idx_lst)
            # change to the newly defined client_sampling function.
            client_indexes, local_itr = self.client_sampling()
            # contribute to time counter
            self.tx_time(client_indexes)
            logger.info("client_indexes = " + str(client_indexes))
# ************************************************************************************************************ #

# ************************************************************************************************************ #
            # store the last model's training parameters.
            last_w = self.model_global.cpu().state_dict() 
             # Initialization
            w_locals, loss_locals, time_interval_lst = [], [], []
            # Update local_itr_lst & client_selec_lst
            local_itr_lst[0, round_idx] = local_itr
            client_selec_lst[round_idx, client_indexes] = 1
# ************************************************************************************************************ #
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            for idx in range(len(client_indexes)):
                # update dataset
                client = self.client_list[idx]
                client_idx = client_indexes[idx]

# ************************************************************************************************************ #
                dataset_idx = client_idx % self.client_num
                client.update_local_dataset(dataset_idx, self.train_data_local_dict[dataset_idx],
                                            self.test_data_local_dict[dataset_idx],
                                            self.train_data_local_num_dict[dataset_idx])
# ************************************************************************************************************ #

# ************************************************************************************************************ #
                # train on new dataset
                # add a new parameter "local_itr" to the funciton "client.train()"
                # add a new return value "time_interval" which is the time consumed for training model in client.
                w, loss, time_interval = client.train(net=copy.deepcopy(self.model_global).to(self.device), local_iteration = local_itr)
                # record current time interval into time_interval_lst
                time_interval_lst.append(time_interval)
                # update the local weights
                local_w_lst[client_idx] = copy.deepcopy(w)
# ************************************************************************************************************ #

                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                loss_locals.append(copy.deepcopy(loss))
                logger.info('Client {:3d}, loss {:.3f}'.format(client_idx, loss))

# ************************************************************************************************************ #
            # calculate FPF1 index.
            def FPF1_cal(local_w_lst):
                def FPF1_norm_sum(w):
                    norm_sum = 0
                    for para in w.keys():
                        norm_sum += torch.norm(w[para] - last_w[para])
                    return norm_sum
                
                norm_sum = np.reshape(np.array(list(map(FPF1_norm_sum, local_w_lst))), (1, -1))
                # return FPF1 index list
                res = norm_sum / np.dot(local_itr_lst, client_selec_lst)
                res[np.bitwise_or(np.isnan(res), np.isinf(res))] = 0
                return res

            # update FPF index list
            FPF1_idx_lst = FPF1_cal(local_w_lst)

            # calculate FPF2 index.
            def FPF2_cal(local_w_lst):
                def FPF2_numerator(w):
                    res = 0
                    for para in w.keys():
                        res += np.linalg.norm(w[para].numpy().ravel() * A_mat[para])
                    return res
                
                numerators = np.reshape(np.array(list(map(FPF2_numerator, local_w_lst))), (1, -1))
                res = numerators / G_mat
                res[np.bitwise_or(np.isnan(res), np.isinf(res))] = 0
                return res

            # update FPF index list
            FPF2_idx_lst = FPF2_cal(local_w_lst)
# ************************************************************************************************************ #

# ************************************************************************************************************ #
            # update the time counter
            if time_interval_lst:
                self.time_counter += sum(time_interval_lst) * timing_ratio / len(time_interval_lst)
            logger.debug("time_counter after training: {}".format(self.time_counter))
            # if current time_counter has exceed the channel table, I will simply stop early
            if self.time_counter >= channel_data['Time'].max(): 
                logger.info("++++++++++++++training process ends early++++++++++++++")
                break
            self.time_counter = np.array(channel_data['Time'][channel_data['Time'] > self.time_counter])[0]
# ************************************************************************************************************ #

            # update global weights
            w_glob = self.aggregate(w_locals)
            # logger.info("global weights = " + str(w_glob))

            # copy weight to net_glob
            self.model_global.load_state_dict(w_glob)

# ************************************************************************************************************ #
            # update A_mat
            for para in w_glob.keys():
                A_mat[para] = A_mat[para] * (1 - 1/G2) + (w_glob[para].numpy().ravel() - last_w[para].numpy().ravel()) / G2

            # update G_mat
            G_mat = G_mat * (1 - 1 / G1) + np.dot(local_itr_lst, client_selec_lst) / G1
            
            # print loss
            if not loss_locals:
                logger.info('Round {:3d}, Average loss None'.format(round_idx))
            else:
                loss_avg = sum(loss_locals) / len(loss_locals)
                logger.info('Round {:3d}, Average loss {:.3f}'.format(round_idx, loss_avg))
# ************************************************************************************************************ #

            if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
                self.local_test_on_all_clients(self.model_global, round_idx)

    def aggregate(self, w_locals):
# ************************************************************************************************************ #
        if not w_locals:
            return self.model_global.cpu().state_dict()
# ************************************************************************************************************ #

        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def local_test_on_all_clients(self, model_global, round_idx):
        logger.info("################local_test_on_all_clients : {}".format(round_idx))
        train_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : []
        }
        
        test_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : []
        }

        client = self.client_list[0]
# ************************************************************************************************************ #
        for client_idx in range(min(client_num_in_total, self.client_num)):
# ************************************************************************************************************ #
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(model_global, False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(model_global, True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            if self.args.dataset == "stackoverflow_lr":
                train_metrics['precisions'].append(copy.deepcopy(train_local_metrics['test_precision']))
                train_metrics['recalls'].append(copy.deepcopy(train_local_metrics['test_recall']))
                test_metrics['precisions'].append(copy.deepcopy(test_local_metrics['test_precision']))
                test_metrics['recalls'].append(copy.deepcopy(test_local_metrics['test_recall']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])
        train_precision = sum(train_metrics['precisions']) / sum(train_metrics['num_samples'])
        train_recall = sum(train_metrics['recalls']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        test_precision = sum(test_metrics['precisions']) / sum(test_metrics['num_samples'])
        test_recall = sum(test_metrics['recalls']) / sum(test_metrics['num_samples'])

        if self.args.dataset == "stackoverflow_lr":
            stats = {'training_acc': train_acc, 'training_precision': train_precision, 'training_recall': train_recall, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Pre": train_precision, "round": round_idx})
            wandb.log({"Train/Rec": train_recall, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})

            boardX.add_scalar("Train/Acc", train_acc, round_idx)
            boardX.add_scalar("Train/Pre", train_precision, round_idx)
            boardX.add_scalar("Train/Rec", train_recall, round_idx)
            boardX.add_scalar("Train/Loss", train_loss, round_idx)

            logger.info(stats)

            stats = {'test_acc': test_acc, 'test_precision': test_precision, 'test_recall': test_recall, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_precision, "round": round_idx})
            wandb.log({"Test/Rec": test_recall, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

            boardX.add_scalar("Test/Acc", test_acc, round_idx)
            boardX.add_scalar("Test/Pre", test_precision, round_idx)
            boardX.add_scalar("Test/Rec", test_recall, round_idx)
            boardX.add_scalar("Test/Loss", test_loss, round_idx)

            logger.info(stats)

        else:
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})

            boardX.add_scalar("Train/Acc", train_acc, round_idx)
            boardX.add_scalar("Train/Loss", train_loss, round_idx)

            logger.info(stats)

            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

            boardX.add_scalar("Test/Acc", test_acc, round_idx)
            boardX.add_scalar("Test/Loss", test_loss, round_idx)

            logger.info(stats)
