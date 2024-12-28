import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.sampling import MinibatchSampling
from config import *

# TensorBoard setup
writer = SummaryWriter(log_dir="./logs")

if use_min_loss:
    raise Exception("use_min_loss should be disabled in centralized case.")

# Initialize the model
model = get_model(model_name)
if hasattr(model, "create_graph"):
    model.create_graph(learning_rate=step_size)

# Fixed averaging slots configuration
use_fixed_averaging_slots = time_gen is not None

# Statistics collection
stat = CollectStatistics(
    results_file_name=single_run_results_file_path
    if single_run
    else multi_run_results_file_path,
    is_single_run=single_run,
)

# Enhanced visualization: client performance tracking
def log_to_tensorboard(writer, step, loss, accuracy, time_taken=None, tag_prefix=""):
    """
    Logs metrics to TensorBoard.
    Args:
        writer: TensorBoard SummaryWriter object.
        step: Current training step or epoch.
        loss: Loss value to log.
        accuracy: Accuracy value to log.
        time_taken: (Optional) Time taken for the iteration.
        tag_prefix: Prefix for TensorBoard tags.
    """
    writer.add_scalar(f"{tag_prefix}/Loss", loss, step)
    writer.add_scalar(f"{tag_prefix}/Accuracy", accuracy, step)
    if time_taken is not None:
        writer.add_scalar(f"{tag_prefix}/Time", time_taken, step)

# Adaptive dynamic client sampling
def adaptive_client_sampling(batch_size, total_data, dynamic_factor=1.0):
    """
    Adjusts the sampling logic based on resource constraints and training dynamics.
    Args:
        batch_size: Current batch size.
        total_data: Total available training data.
        dynamic_factor: Scaling factor for batch adjustments.
    Returns:
        Updated batch indices.
    """
    adjusted_batch_size = int(batch_size * dynamic_factor)
    adjusted_batch_size = max(1, min(adjusted_batch_size, total_data))
    return np.arange(adjusted_batch_size)

# Training loop
for sim in sim_runs:
    # Data loading with dynamic adjustments
    train_image, train_label, test_image, test_label, train_label_orig = get_data(
        dataset, total_data, dataset_file_path, sim_round=sim if batch_size >= total_data else None
    )
    train_indices = np.arange(len(train_label))

    sampler = (
        MinibatchSampling(train_indices, batch_size, sim)
        if batch_size < total_data
        else None
    )

    stat.init_stat_new_global_round()

    dim_w = model.get_weight_dimension(train_image, train_label)
    w_init = model.get_init_weight(dim_w, rand_seed=sim)
    w = w_init
    w_min_loss = None
    loss_min = np.inf

    total_time = 0
    total_time_recomputed = 0

    iteration = 0

    while True:
        time_start = time.time()

        w_prev = w
        if sampler:
            train_indices = sampler.get_next_batch()

        # Adaptive sampling
        if iteration % 5 == 0:
            train_indices = adaptive_client_sampling(batch_size, total_data, dynamic_factor=0.9 + 0.1 * np.random.rand())

        grad = model.gradient(train_image, train_label, w, train_indices)
        w = w - step_size * grad

        if np.isnan(w).any():
            print("*** NaN encountered in weights, reverting to previous values")
            w = w_prev
        else:
            loss_latest = model.loss(train_image, train_label, w, train_indices)

            if use_min_loss and loss_latest < loss_min:
                loss_min = loss_latest
                w_min_loss = w

        time_end = time.time()
        iteration_time = time_end - time_start
        total_time += iteration_time

        stat.collect_stat_end_local_round(
            None, np.nan, iteration_time, np.nan, None, model, train_image, train_label,
            test_image, test_label, w, total_time
        )

        # Log to TensorBoard
        if iteration % 10 == 0:
            train_loss = model.loss(train_image, train_label, w, train_indices)
            train_accuracy = model.evaluate(train_image, train_label, w)
            log_to_tensorboard(writer, iteration, train_loss, train_accuracy, iteration_time, tag_prefix="Train")

        iteration += 1

        if total_time >= max_time:
            break

    w_eval = w_min_loss if use_min_loss else w
    test_loss = model.loss(test_image, test_label, w_eval, None)
    test_accuracy = model.evaluate(test_image, test_label, w_eval)

    # Log final statistics
    log_to_tensorboard(writer, sim, test_loss, test_accuracy, tag_prefix="Test")
    stat.collect_stat_end_global_round(
        sim, None, np.nan, total_time, model, train_image, train_label,
        test_image, test_label, w_eval, total_time
    )

writer.close()

