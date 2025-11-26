# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Iterable, Sequence, Type, Union
import warnings

import torch
from torch import Tensor

import os
import xarray as xr
from datetime import datetime
import numpy as np

try:
    from apex.optimizers import FusedAdam
except ImportError:
    warnings.warn("Apex is not installed, defaulting to PyTorch optimizers.")

from physicsnemo import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger, PythonLogger
# from physicsnemo.launch.utils import load_checkpoint #, save_checkpoint
from model.checkpoint import load_checkpoint, save_checkpoint
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
import glob
import re


class Trainer:
    """Training loop for diagnostic models."""

    def __init__(
        self,
        model: Module,
        dist_manager: DistributedManager,
        loss: Callable,
        train_datapipe: Sequence,
        valid_datapipe: Sequence,
        input_output_from_batch_data: Union[Callable, None] = None,
        optimizer: Union[Type[torch.optim.Optimizer], None] = None,
        optimizer_params: Union[dict, None] = None,
        scheduler: Union[Type[torch.optim.lr_scheduler.LRScheduler], None] = None,
        scheduler_params: Union[dict, None] = None,
        max_epoch: int = 1,
        patience: int = 50,
        save_best_checkpoint: bool = True,
        load_epoch: Union[int, str, None] = None,
        inference_on_epoch: Union[str, int] = "best",
        checkpoint_every: int = 1,
        checkpoint_dir: Union[str, None] = None,
        validation_callbacks: Iterable[Callable] = (),
    ):
        self.model = model
        self.dist_manager = dist_manager
        self.loss = loss
        self.train_datapipe = train_datapipe
        self.valid_datapipe = valid_datapipe
        self.max_epoch = max_epoch
        if input_output_from_batch_data is None:
            input_output_from_batch_data = lambda x: x
        self.input_output_from_batch_data = input_output_from_batch_data
        self.optimizer = self._setup_optimizer(
            opt_cls=optimizer, opt_params=optimizer_params
        )
        self.lr_scheduler = self._setup_lr_scheduler(
            scheduler_cls=scheduler, scheduler_params=scheduler_params
        )
        self.validation_callbacks = list(validation_callbacks)
        self.inference_on_epoch = inference_on_epoch
        self.device = self.dist_manager.device
        self.logger = PythonLogger()

        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self.epoch = 1
        if load_epoch is not None:
            epoch = None if load_epoch == "latest" else load_epoch
            self.load_checkpoint(epoch=epoch)

        # Early stopping
        self.save_best_checkpoint = save_best_checkpoint
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.best_model_state_dict = None
        self.patience = patience 
        self.bad_epochs = 0


        # wrap capture here instead of using decorator so it'll still be wrapped if
        # overridden by a subclass
        self.train_step_forward = StaticCaptureTraining(
            model=self.model,
            optim=self.optimizer,
            logger=self.logger,
            use_graphs=False,  # for some reason use_graphs=True causes a crash
        )(self.train_step_forward)

        self.eval_step = StaticCaptureEvaluateNoGrad(
            model=self.model, logger=self.logger, use_graphs=False
        )(self.eval_step)

    def eval_step(self, invar: Tensor) -> Tensor:
        """Perform one step of model evaluation."""
        return self.model(invar[0], invar[1])

    def train_step_forward(self, invar: Tensor, outvar_true: Tensor) -> Tensor:
        """Train model on one batch."""
        outvar_pred = self.model(invar[0], invar[1])  # invar[0] is the input, invar[1] is the ratio
      
        return self.loss(outvar_pred, outvar_true)

    def fit(self):
        """Main function for training loop."""
        for self.epoch in range(self.epoch, self.max_epoch + 1):
            self.train_on_epoch()

            # Early stopping check
            if self.bad_epochs >= self.patience:
                print(f"Early stopping triggered at epoch {self.epoch}")
                break

        if self.dist_manager.rank == 0:
            self.logger.info("Finished training!")

            # Save best model using PhysicsNemo logic
            if self.save_best_checkpoint and self.best_model_state_dict is not None:
                self.model.load_state_dict(self.best_model_state_dict)
                self.epoch = self.best_epoch
                self.save_checkpoint(base_name="best")
                print(
                    f"Saved best model from epoch {self.best_epoch} "
                    f"with val_loss={self.best_val_loss:.5f}"
                )

    def train_on_epoch(self):
        """Train for one epoch."""
        with LaunchLogger(
            "train",
            epoch=self.epoch,
            num_mini_batch=len(self.train_datapipe),
            epoch_alert_freq=10,
        ) as log:
            for batch in self.train_datapipe:
                inputs, target, *_ = self.input_output_from_batch_data(batch)    
                loss = self.train_step_forward(
                    inputs, target)
            
                log.log_minibatch({"loss": loss.detach()})

            log.log_epoch({"Learning Rate": self.optimizer.param_groups[0]["lr"]})

        # Validation
        if self.dist_manager.rank == 0:
            with LaunchLogger("valid", epoch=self.epoch) as log:
                val_loss = self.validate_on_epoch()
                log.log_epoch({"Validation error": val_loss})

            # Track best model (do not save yet)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = self.epoch
                self.best_model_state_dict = {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                }
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1

        if self.dist_manager.world_size > 1:
            torch.distributed.barrier()

        self.lr_scheduler.step()

        checkpoint_epoch = (self.checkpoint_dir is not None) and (
            (self.epoch % self.checkpoint_every == 0) or (self.epoch == self.max_epoch)
        )
        if checkpoint_epoch and self.dist_manager.rank == 0:
            # Save PhysicsNeMo Launch checkpoint
            self.save_checkpoint(base_name="checkpoint")

    @torch.no_grad()
    def validate_on_epoch(
        self,
        perform_inference: bool = False, 
        save_dir: str = "./results/maps",
        prediction_var_name: str = "predicted",
        mean = None,
        std = None,
    ) -> Tensor:
        """Return average loss over one validation epoch."""
        loss_epoch = 0
        num_examples = 0  # Number of validation examples
        # Dealing with DDP wrapper
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model

        preds = []
        times = []
        if mean is not None and std is not None:
            mean = torch.from_numpy(mean).to(device=self.device)
            std = torch.from_numpy(std).to(device=self.device)
            # reverse normalization
            if mean.ndim == 1:  # (C,)
                mean = mean.view(1, -1, 1, 1)
                std = std.view(1, -1, 1, 1)
            elif mean.ndim == 3:  # (C, H, W)
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected shape for mean: {mean.shape}")
            
        try:
            model.eval()
            for (i, batch) in enumerate(self.valid_datapipe):
                (invar, outvar_true, coords_dict) = self.input_output_from_batch_data(batch)
                invar = (invar[0].detach(), invar[1].detach())
                outvar_true = outvar_true.detach()
                outvar_pred = self.eval_step(invar)

                loss_epoch += self.loss(outvar_pred, outvar_true)
                num_examples += 1

                for callback in self.validation_callbacks:
                    callback(outvar_true, outvar_pred, epoch=self.epoch, batch_idx=i)
                if perform_inference:
                    if mean is not None and std is not None:
                        # reverse normalization
                        outvar_pred_unnorm = outvar_pred * std + mean
 
                    preds.append(outvar_pred_unnorm.cpu())
                    times.extend([np.datetime64(t) for t in coords_dict["time"]])
        finally:  # restore train state even if exception occurs
            model.train()

        if perform_inference and self.dist_manager.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            
            preds_np = torch.cat(preds, dim=0).numpy()
            times_np = np.array(times, dtype="datetime64[ns]")

            # Since coords_dict is drawn from the data loader, it comes in batches, e.g. coords_dict["lat"] has shape (batchs_size, lat)
            lat = coords_dict["lat"][0]
            lon = coords_dict["lon"][0]

            coords = {
                "time": times_np,
                "latitude": lat,  # static
                "longitude": lon,
            }

            # Only add "channel" if not splitting per-variable
            if not isinstance(prediction_var_name, list):
                coords["channel"] = np.arange(preds_np.shape[1])

            # If multiple variable names are provided, split channels and assign each
            if isinstance(prediction_var_name, list):
                if len(prediction_var_name) != preds_np.shape[1]:
                    raise ValueError(f"Number of variable names ({len(prediction_var_name)}) "
                                     f"does not match number of channels ({preds_np.shape[1]}) in predictions.")

                data_vars = {
                    var_name: (["time", "latitude", "longitude"], preds_np[:, i])
                    for i, var_name in enumerate(prediction_var_name)
                }
            else:
                # Fallback: use a single variable name
                data_vars = {
                    prediction_var_name: (["time", "channel", "latitude", "longitude"], preds_np)
                }

            ds = xr.Dataset(data_vars, coords=coords)

            if isinstance(prediction_var_name, list):
                var_str = "_".join(prediction_var_name)
            else:
                var_str = prediction_var_name

            timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            netcdf_path = os.path.join(save_dir, f"{var_str}_{timestamp}.nc")
            ds.to_netcdf(netcdf_path)
            print(f"✅ Saved predictions to NetCDF: {netcdf_path}")

        return loss_epoch / num_examples

    def _setup_optimizer(self, opt_cls=None, opt_params=None):
        """Initialize optimizer."""
        opt_kwargs = {"lr": 0.0005}
        if opt_params is not None:
            opt_kwargs.update(opt_params)

        if opt_cls is None:
            try:
                opt_cls = FusedAdam
            except NameError:  # in case we don't have apex
                opt_cls = torch.optim.AdamW

        return opt_cls(self.model.parameters(), **opt_kwargs)

    def _setup_lr_scheduler(self, scheduler_cls=None, scheduler_params=None):
        """Initialize learning rate scheduler."""
        scheduler_kwargs = {}
        if scheduler_cls is None:
            scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
            scheduler_kwargs["T_max"] = self.max_epoch
        if scheduler_params is not None:
            scheduler_kwargs.update(scheduler_params)

        return scheduler_cls(self.optimizer, **scheduler_kwargs)

    def load_checkpoint(self, epoch: Union[int, None] = None) -> int:
        """Load training state from checkpoint.

        Parameters
        ----------
        epoch: int or None, optional
            The epoch for which the state is loaded. If None, will load the
            latest epoch.
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set in order to load checkpoints.")
        self.epoch = load_checkpoint(
            self.checkpoint_dir,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            device=self.device,
            epoch=epoch,
        )
        return self.epoch

    def save_checkpoint(self, base_name: str = "checkpoint"):
        """Save training state from checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set in order to save checkpoints.")
        save_checkpoint(
            self.checkpoint_dir,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            epoch=self.epoch,
            base_name=base_name
        )

    def load_model_for_inference(self):
        """Loads the best model saved with base_name='best'."""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set to load checkpoints.")
        if self.inference_on_epoch == "best":
            # Obtain epoch corresponding to best
            # Find all checkpoint files with 'best' prefix
            best_checkpoints = glob.glob(f"{self.checkpoint_dir}/best.*.pt")
            if not best_checkpoints:
                self.logger.warning("No best checkpoint found. Loading latest checkpoint instead.")
                epoch = None
            else:
                # Extract epoch number from filenames
                epochs = []
                for ckpt in best_checkpoints:
                    match = re.search(r'best\.\d+\.(\d+)\.pt', ckpt)
                    if match:
                        epochs.append(int(match.group(1)))
                
                if epochs:
                    epoch = max(epochs)  # Get the highest epoch number
                    self.logger.info(f"Loading best model from epoch {epoch}")
                else:
                    epoch = None
                    self.logger.warning("Could not parse epoch from best checkpoint filename. Loading latest.")
            load_checkpoint(
                path=self.checkpoint_dir,
                models=self.model,
                optimizer=None,
                scheduler=None,
                epoch=epoch,  # or None if you just want latest
                device=self.device,
                base_name="best"
            )
        else:
            load_checkpoint(
                path=self.checkpoint_dir,
                models=self.model,
                optimizer=None,
                scheduler=None,
                epoch=self.inference_on_epoch,
                device=self.device,
            )

        
        

