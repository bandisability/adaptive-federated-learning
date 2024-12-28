import socket
import time
import json
import os
import numpy as np
from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauServer
from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.utils import send_msg, recv_msg, get_indices_each_node_case
from torch.utils.tensorboard import SummaryWriter

# Configurations
from config import *

# Initialize TensorBoard logging
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

model = get_model(model_name)
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)

if time_gen is not None:
    use_fixed_averaging_slots = True
else:
    use_fixed_averaging_slots = False

if batch_size < total_data:  # Read all data once when using stochastic gradient descent
    train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path)

    # Precompute indices for all nodes
    indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind((SERVER_ADDR, SERVER_PORT))
client_sock_all = []

# Establish connections with clients
while len(client_sock_all) < n_nodes:
    listening_sock.listen(5)
    print("Waiting for incoming connections...")
    client_sock, (ip, port) = listening_sock.accept()
    print('Connected to client:', (ip, port))
    client_sock_all.append(client_sock)

if single_run:
    stat = CollectStatistics(results_file_name=single_run_results_file_path, is_single_run=True)
else:
    stat = CollectStatistics(results_file_name=multi_run_results_file_path, is_single_run=False)

# Broadcast model and data to clients
for sim in sim_runs:
    if batch_size >= total_data:
        train_image, train_label, test_image, test_label, train_label_orig = get_data(
            dataset, total_data, dataset_file_path, sim_round=sim)

        indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

    for case in case_range:
        for tau_setup in tau_setup_all:
            stat.init_stat_new_global_round()

            dim_w = model.get_weight_dimension(train_image, train_label)
            w_global_init = model.get_init_weight(dim_w, rand_seed=sim)
            w_global = w_global_init
            w_global_min_loss = None
            loss_min = np.inf
            prev_loss_is_min = False

            # Adaptive control algorithm
            if tau_setup < 0:
                is_adapt_local = True
                tau_config = 1
            else:
                is_adapt_local = False
                tau_config = tau_setup

            if is_adapt_local or estimate_beta_delta_in_all_runs:
                if tau_setup == -1:
                    control_alg = ControlAlgAdaptiveTauServer(is_adapt_local, dim_w, client_sock_all, n_nodes,
                                                              control_param_phi, moving_average_holding_param)
                else:
                    raise Exception('Invalid tau setup.')
            else:
                control_alg = None

            for n in range(n_nodes):
                indices_this_node = indices_each_node_case[case][n]
                msg = ['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset,
                       num_iterations_with_same_minibatch_for_tau_equals_one, step_size,
                       batch_size, total_data, control_alg, indices_this_node, read_all_data_for_stochastic,
                       use_min_loss, sim]
                send_msg(client_sock_all[n], msg)

            print('All clients connected')

            # Wait for data preparation to complete
            for n in range(n_nodes):
                recv_msg(client_sock_all[n], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER')

            print('Starting federated learning process')

            total_time = 0
            total_time_recomputed = 0
            it_each_local = None
            it_each_global = None
            is_last_round = False
            is_eval_only = False
            tau_new_resume = None

            while True:
                print('---------------------------------------------')

                time_total_all_start = time.time()
                for n in range(n_nodes):
                    msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_config, is_last_round, prev_loss_is_min]
                    send_msg(client_sock_all[n], msg)

                w_global_prev = w_global
                w_global = np.zeros(dim_w)
                loss_last_global = 0.0
                loss_w_prev_min_loss = 0.0
                received_loss_local_w_prev_min_loss = False
                data_size_total = 0
                time_all_local_all = 0
                data_size_local_all = []

                tau_actual = 0
                for n in range(n_nodes):
                    msg = recv_msg(client_sock_all[n], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
                    w_local = msg[1]
                    time_all_local = msg[2]
                    tau_actual = max(tau_actual, msg[3])
                    data_size_local = msg[4]
                    loss_local_last_global = msg[5]
                    loss_local_w_prev_min_loss = msg[6]

                    w_global += w_local * data_size_local
                    data_size_total += data_size_local
                    time_all_local_all = max(time_all_local_all, time_all_local)

                    if use_min_loss:
                        loss_last_global += loss_local_last_global * data_size_local
                        if loss_local_w_prev_min_loss is not None:
                            loss_w_prev_min_loss += loss_local_w_prev_min_loss * data_size_local
                            received_loss_local_w_prev_min_loss = True

                w_global /= data_size_total

                # Log loss and performance metrics
                writer.add_scalar("Loss/Last_Global", loss_last_global, stat.current_round)
                writer.add_scalar("Time/Local", time_all_local_all, stat.current_round)

                if use_min_loss:
                    loss_last_global /= data_size_total
                    if received_loss_local_w_prev_min_loss:
                        loss_w_prev_min_loss /= data_size_total
                        loss_min = loss_w_prev_min_loss

                    if loss_last_global < loss_min:
                        loss_min = loss_last_global
                        w_global_min_loss = w_global_prev
                        prev_loss_is_min = True
                    else:
                        prev_loss_is_min = False

                time_total_all_end = time.time()
                time_total_all = time_total_all_end - time_total_all_start
                time_global_aggregation_all = max(0.0, time_total_all - time_all_local_all)

                if use_fixed_averaging_slots:
                    it_each_local = np.sum(time_gen.get_local(tau_actual)) / tau_actual
                    it_each_global = time_gen.get_global(1)[0]
                else:
                    it_each_local = time_all_local_all / tau_actual
                    it_each_global = time_global_aggregation_all

                total_time += time_total_all
                total_time_recomputed += it_each_local * tau_actual + it_each_global

                stat.collect_stat_end_local_round(case, tau_actual, it_each_local, it_each_global, control_alg, model,
                                                  train_image, train_label, test_image, test_label, w_global,
                                                  total_time_recomputed)

                # Determine if it is the last round
                if total_time_recomputed > max_time:
                    is_last_round = True

                if is_last_round:
                    break

            if use_min_loss:
                w_eval = w_global_min_loss
            else:
                w_eval = w_global

            stat.collect_stat_end_global_round(sim, case, tau_setup, total_time, model, train_image, train_label,
                                               test_image, test_label, w_eval, total_time_recomputed)

writer.close()

