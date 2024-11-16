#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=5):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Args:
            patience  (int):  Number of epochs with no improvement after which training will be stopped.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.save_ckpt = False
        self.stop_flag = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        if np.isnan(val_loss) or np.isinf(val_loss):
            raise RuntimeError("Invalid loss, terminating training")

        score = -val_loss

        if self.best_score is None:
            self.save_ckpt = True
            self.best_score = score
        elif score < self.best_score:
            self.save_ckpt = False
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_flag = True
        else:
            self.save_ckpt = True
            self.best_score = score
            self.counter = 0

        return self.save_ckpt, self.stop_flag


class LossScheduler:
    def __init__(self, decay_end=1000, curve_name="s"):
        decay_start = 0
        self.counter = -1
        self.decay_end = decay_end
        self.interpolated_values = self.curve_interpolation(
            decay_start, decay_end, decay_end, curve_name
        )

    def linear_interpolation(self, start, end, num_points):
        x = np.linspace(start, end, num_points)
        return x

    def s_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (t - np.sin(2 * np.pi * t) / (2 * np.pi))
        return x

    def inverse_s_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (t + np.sin(2 * np.pi * t) / (2 * np.pi))
        return x

    def deceleration_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (1 - np.cos(np.pi * t / 2))
        return x

    def acceleration_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (np.sin(np.pi * t / 2))
        return x

    def curve_interpolation(self, start, end, num_points, curve_name):
        if curve_name == "linear":
            interpolated_values = self.linear_interpolation(start, end, num_points)
        elif curve_name == "s":
            interpolated_values = self.s_curve_interpolation(start, end, num_points)
        elif curve_name == "inverse_s":
            interpolated_values = self.inverse_s_curve_interpolation(
                start, end, num_points
            )
        elif curve_name == "deceleration":
            interpolated_values = self.deceleration_curve_interpolation(
                start, end, num_points
            )
        elif curve_name == "acceleration":
            interpolated_values = self.acceleration_curve_interpolation(
                start, end, num_points
            )
        else:
            assert False, "Invalid curve name. {}".format(curve_name)

        return interpolated_values / num_points

    def __call__(self, loss_weight):
        self.counter += 1
        if self.counter >= self.decay_end:
            return loss_weight
        else:
            return self.interpolated_values[self.counter] * loss_weight
