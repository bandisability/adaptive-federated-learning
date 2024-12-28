import socket
import time
import struct
import logging
import json
from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauClient, ControlAlgAdaptiveTauServer
from data_reader.data_reader import get_data, get_data_train_samples
from models.get_model import get_model
from util.sampling import MinibatchSampling
from util.utils import send_msg, recv_msg, log_metrics_to_tensorboard

# Configurations are in a separate config.py file
from config import SERVER_ADDR, SERVER_PORT, dataset_file_path, TENSORBOARD_LOG_DIR
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TensorBoard writer for logging metrics
writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

# Initialize socket for client-server communication
sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))

print('---------------------------------------------------------------------------')

# Variables to track state across iterations
batch_size_prev = None
total_data_prev = None
sim_prev = None

# Function to save weights and metrics periodically
def save_model_weights(model_weights, round_number, save_path="./model_weights/"):
    """Save the model weights to a file."""
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = f"{save_path}/weights_round_{round_number}.pth"
    with open(filename, 'wb') as f:
        f.write(model_weights)
    logger.info(f"Model weights saved to {filename}")

# Function to log detailed metrics
def log_detailed_metrics(metrics, round_number):
    """Log detailed metrics to a JSON file."""
    metrics_file = f"./logs/metrics_round_{round_number}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_file}")

try:
    while True:
        # Receive initialization message from server
        msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
        model_name, dataset, num_iterations_with_same_minibatch_for_tau_equals_one, step_size, batch_size, total_data, control_alg_server_instance, indices_this_node, read_all_data_for_stochastic, use_min_loss, sim = msg[1:]

        # Initialize model
        model = get_model(model_name)
        if hasattr(model, 'create_graph'):
            model.create_graph(learning_rate=step_size)

        # Load dataset if necessary
        if read_all_data_for_stochastic or batch_size >= total_data:
            if batch_size_prev != batch_size or total_data_prev != total_data or (batch_size >= total_data and sim_prev != sim):
                logger.info('Loading all data samples...')
                train_image, train_label, _, _, _ = get_data(dataset, total_data, dataset_file_path, sim_round=sim)

        batch_size_prev = batch_size
        total_data_prev = total_data
        sim_prev = sim

        if batch_size >= total_data:
            sampler = None
            train_indices = indices_this_node
        else:
            sampler = MinibatchSampling(indices_this_node, batch_size, sim)
            train_indices = None

        # Notify server that data preparation is complete
        send_msg(sock, ['MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER'])

        while True:
            # Receive weights and tau configuration from server
            msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
            w, tau_config, is_last_round = msg[1:4]

            time_local_start = time.time()
            local_metrics = {'loss_history': [], 'gradient_norms': [], 'round_time': 0}

            # Local training
            for i in range(tau_config):
                if batch_size < total_data:
                    sample_indices = sampler.get_next_batch()
                    train_image, train_label = get_data_train_samples(dataset, sample_indices, dataset_file_path)
                    train_indices = range(0, len(train_label))

                # Compute gradient and update weights
                grad = model.gradient(train_image, train_label, w, train_indices)
                grad_norm = (grad ** 2).sum() ** 0.5
                local_metrics['gradient_norms'].append(grad_norm)
                logger.info(f"Gradient norm at iteration {i}: {grad_norm}")

                if i == 0:
                    loss = model.loss(train_image, train_label, w, train_indices)
                    local_metrics['loss_history'].append(loss)
                    writer.add_scalar("Loss/Local", loss, i)
                    logger.info(f"Initial loss at iteration {i}: {loss}")

                w = w - step_size * grad

            # End of local training
            time_local_end = time.time()
            time_all_local = time_local_end - time_local_start
            local_metrics['round_time'] = time_all_local
            logger.info(f"Local computation time: {time_all_local:.4f} seconds")

            # Save model weights and log metrics
            save_model_weights(w, tau_config)
            log_detailed_metrics(local_metrics, tau_config)

            # Send updated weights and timing information back to server
            send_msg(sock, ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_all_local])

            if is_last_round:
                break

except (struct.error, socket.error) as e:
    logger.error(f"Server has stopped. Error: {e}")
finally:
    writer.close()
    logger.info("Client process terminated.")
