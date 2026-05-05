import torch
import numpy as np
import gpytorch
from math import log
import math
import time
from .models import SVGPModel, ExactGPModel, ArdGPModel, CustomGPModel
from .acquisition import (optimize_acquisition, optimize_acquisition_for_context,
                          optimize_scalarized_acquisition_for_context, ucb_acquisition_group)
from typing import Callable, Optional, Tuple, List, Dict
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
from .LBFGS import FullBatchLBFGS
from .gen_models import ParetoVAETrainer
from .simple_ddpm_model import SimpleParetoTrainer
from scipy.stats.qmc import LatinHypercube

IF_GLOBAL = True
SCALAR = "HV"
NOISE = False


class HypervolumeScalarization:
    """
    Hypervolume scalarization for minimization using a nadir point,
    where the per-coordinate ratio is exponentiated first and then the
    minimum over objectives is taken.

    Given an objective vector y (to be minimized), a nadir point (an upper bound),
    and a weight vector, we define the scalarization as:

        s_λ(y) = min_i [ max(0, (nadir_i - y_i) / weights_i) ^ exponent ]

    We then return its negative so that a lower scalarized value corresponds to a better candidate.

    Args:
        nadir_point (torch.Tensor): The nadir point vector (upper bounds) for each objective.
        exponent (float): The exponent to apply to each coordinate ratio before minimization.
    """

    def __init__(self, nadir_point: torch.Tensor, exponent: float = 2.0):
        self.nadir_point = nadir_point
        self.exponent = exponent

    def __call__(self, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Compute the improvement relative to the nadir point.
        # Assuming y <= nadir_point element-wise, diff = nadir_point - y.
        diff = self.nadir_point - y
        # Compute the per-objective ratio.
        ratio = diff / (weights + 1e-8)
        # Ensure non-negativity.
        ratio = torch.clamp(ratio, min=0.0)
        # First, raise each element to the specified exponent.
        exp_ratio = ratio ** self.exponent
        # Then, take the minimum over the objective dimensions.
        min_val, _ = torch.min(exp_ratio, dim=-1)
        # Return the negative so that minimizing the scalarized value corresponds to better (lower) objective values.
        return -min_val


class AugmentedTchebycheff:
    """Augmented Tchebycheff scalarization"""

    def __init__(self, reference_point: torch.Tensor, rho: float = 0.05):
        self.reference_point = reference_point
        self.rho = rho

    def __call__(self, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_diff = weights * (y - self.reference_point)
        weighted_y = weights * y
        max_term = torch.max(weighted_diff, dim=-1)[0]
        sum_term = self.rho * torch.sum(weighted_y, dim=-1)
        return max_term + sum_term


class PseudoObjectiveFunction:
    """Wrapper for objective functions"""
    def __init__(self,
                 func: Callable,
                 dim: int = 0,
                 context_dim: int = 0,
                 output_dim: int = 0,
                 nadir_point: torch.Tensor = None):
        self.func = func
        self.input_dim = dim
        self.dim = dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.nadir_point = nadir_point

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


class BayesianOptimization:
    def __init__(self,
                 objective_func,
                 inducing_points=None,
                 train_steps=500,
                 model_type='SVGP',
                 optimizer_type='adam'):
        self.objective_func = objective_func
        self.inducing_points = inducing_points
        self.train_steps = train_steps
        self.model_type = model_type
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.optimizer_type = optimizer_type.lower()

        self.dim = objective_func.dim
        self.model = None

        # Placeholders for normalization parameters
        self.x_mean, self.x_std = None, None
        self.y_mean, self.y_std = None, None

        # ------------------------------
        # Z-score Normalization Function
        # ------------------------------
    def normalize_data(self, X, Y):
        """
        Perform Z-score normalization for train_X and train_Y.

        Parameters:
        - X: Training input data (train_X).
        - Y: Training output data (train_Y).

        Returns:
        - Normalized X and Y tensors.
        """
        # Compute mean and std for X and Y
        x_mean, x_std = X.mean(dim=0), X.std(dim=0)
        y_mean, y_std = Y.mean(dim=0), Y.std(dim=0)

        # Handle zero variance explicitly
        zero_var_x = (x_std == 0)
        zero_var_y = (y_std == 0)

        if zero_var_x.any():
            print(f"Warning: {zero_var_x.sum()} input features have zero variance")
            x_std[zero_var_x] = 1.0  # Don't normalize constant features

        if zero_var_y.any():
            print(f"Warning: {zero_var_y.sum()} output objectives have zero variance")
            y_std[zero_var_y] = 1.0  # Don't normalize constant objectives

        # Z-score normalization
        X_normalized = (X - x_mean) / x_std
        Y_normalized = (Y - y_mean) / y_std

        # Store the current normalization parameters
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

        return X_normalized, Y_normalized

    def normalize_minmax_data(self, X, Y, input_bounds=torch.Tensor([0, 1]), output_bounds=None):
        """
        Perform min-max normalization to scale data to [0,1] range.

        Parameters:
        - X: Training input data (train_X).
        - Y: Training output data (train_Y).
        - input_bounds: Optional bounds for X in format [[min_x1, min_x2,...], [max_x1, max_x2,...]].
                       If None, bounds are determined from the data.
        - output_bounds: Optional bounds for Y in format [[min_y1, min_y2,...], [max_y1, max_y2,...]].
                        If None, bounds are determined from the data.

        Returns:
        - Normalized X and Y tensors scaled to [0,1].
        """
        # Determine input bounds if not provided
        if input_bounds is None:
            x_min, _ = torch.min(X, dim=0)
            x_max, _ = torch.max(X, dim=0)
        else:
            x_min = input_bounds[0]
            x_max = input_bounds[1]

        # Determine output bounds if not provided
        if output_bounds is None:
            y_min, _ = torch.min(Y, dim=0)
            y_max, _ = torch.max(Y, dim=0)
        else:
            y_min = output_bounds[0]
            y_max = output_bounds[1]

        # Add small epsilon to prevent division by zero
        x_range = (x_max - x_min) + 1e-8
        y_range = (y_max - y_min) + 1e-8

        # Min-max normalization to [0,1]
        X_normalized = (X - x_min) / x_range
        Y_normalized = (Y - y_min) / y_range

        # Store the current normalization parameters
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        return X_normalized, Y_normalized

    def denormalize_minmax_data(self, X_norm, Y_norm=None):
        """
        Convert normalized data back to original scale.

        Parameters:
        - X_norm: Normalized input data.
        - Y_norm: Normalized output data (optional).

        Returns:
        - Denormalized X and Y tensors (Y only if provided).
        """
        # Denormalize X
        X = X_norm * (self.x_max - self.x_min) + self.x_min

        # Denormalize Y if provided
        if Y_norm is not None:
            Y = Y_norm * (self.y_max - self.y_min) + self.y_min
            return X, Y

        return X

    def normalize_inference(self, X):
        """
        Normalize new input points during inference using stored scaling factors.

        Parameters:
        - X: New input points.

        Returns:
        - Normalized X tensor.
        """
        if self.x_mean is not None and self.x_std is not None:
            return (X - self.x_mean) / self.x_std
        return X

    def denormalize_input(self, X):
        """
        Denormalize input points to original space.

        Parameters:
        - X: Normalized input points.

        Returns:
        - Denormalized X tensor.
        """
        if self.x_mean is not None and self.x_std is not None:
            return X * self.x_std + self.x_mean
        return X

    def denormalize_output(self, Y):
        """
        Denormalize predictions to the original scale.

        Parameters:
        - Y: Normalized predictions.

        Returns:
        - Denormalized Y tensor.
        """
        if self.y_mean is not None and self.y_std is not None:
            return Y * self.y_std + self.y_mean
        return Y

    def normalize_output(self, Y):
        """
        Normalize predictions to the original scale.

        Parameters:
        - Y: Normalized predictions.

        Returns:
        - Denormalized Y tensor.
        """
        if self.y_mean is not None and self.y_std is not None:
            return (Y - self.y_mean) / self.y_std
        return Y

    def build_model(self, X_train, y_train):
        if self.model_type == 'SVGP':
            model = SVGPModel(self.inducing_points, input_dim=self.dim)
        elif self.model_type == 'ArdGP':
            model = ArdGPModel(X_train, y_train, self.likelihood)
        else:
            model = ExactGPModel(X_train, y_train, self.likelihood)
        return model

    def optimize(self, X_train, y_train, n_iter=50, beta=2.0):
        best_y = []

        for i in range(n_iter):
            X_train_norm, y_train_norm = self.normalize_data(X_train.clone(), y_train.clone())
            # print(X_train_norm)
            # print(y_train_norm)

            model = self.build_model(X_train_norm, y_train_norm)
            model.train()
            self.likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=0.01) if self.optimizer_type == 'adam' else torch.optim.LBFGS(
                model.parameters(), lr=0.1, max_iter=20)

            # mll = gpytorch.mlls.VariationalELBO(self.likelihood, model, num_data=y_train.size(0))
            if self.model_type == 'SVGP':
                mll = gpytorch.mlls.VariationalELBO(self.likelihood, model, num_data=y_train.size(0))
            else:
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)

            if self.optimizer_type == 'lbfgs':
                prev_loss = float('inf')
                for _ in range(20):
                    def closure():
                        optimizer.zero_grad()
                        output = model(X_train_norm)
                        loss = -mll(output, y_train_norm)
                        loss.backward()
                        return loss

                    curr_loss = optimizer.step(closure)
                    if abs(prev_loss - curr_loss.item()) < 1e-5:
                        break
                    prev_loss = curr_loss.item()

            else:
                prev_loss = float('inf')
                for _ in range(self.train_steps):
                    optimizer.zero_grad()
                    output = model(X_train_norm)
                    loss = -mll(output, y_train_norm)
                    loss.backward()
                    optimizer.step()
                    prev_loss = loss.item()

            next_x = optimize_acquisition(model, likelihood=self.likelihood, beta=beta,
                                          dim=self.dim, x_mean=self.x_mean, x_std=self.x_std)
            next_y = self.objective_func.evaluate(next_x)

            X_train = torch.cat([X_train, next_x.unsqueeze(0)])
            y_train = torch.cat([y_train, next_y.unsqueeze(0)])
            best_y.append(y_train.min().item())

            if i % 5 == 0:
                print(f'Iteration {i}/{n_iter}, Best y: {y_train.min().item():.3f}')

            self.model = model

        return X_train, y_train, best_y


class ContextualBayesianOptimization:
    def __init__(
            self,
            objective_func,
            inducing_points: Optional[torch.Tensor] = None,
            train_steps: int = 500,
            model_type: str = 'SVGP',
            optimizer_type: str = 'adam'
    ):
        self.objective_func = objective_func
        self.x_dim = objective_func.dim
        self.context_dim = objective_func.context_dim
        self.dim = self.x_dim + self.context_dim

        self.inducing_points = inducing_points
        self.train_steps = train_steps
        self.model_type = model_type
        self.optimizer_type = optimizer_type.lower()

        self.likelihood_new = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-3)
        )

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.model = None

        # Placeholders for normalization parameters
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        # Dictionary to track best values per context
        self.context_best_values = {}

    def normalize_minmax_data(self, X, Y, input_bounds=torch.Tensor([0, 1]), output_bounds=None):
        """
        Perform min-max normalization to scale data to [0,1] range.

        Parameters:
        - X: Training input data (train_X).
        - Y: Training output data (train_Y).
        - input_bounds: Optional bounds for X in format [[min_x1, min_x2,...], [max_x1, max_x2,...]].
                       If None, bounds are determined from the data.
        - output_bounds: Optional bounds for Y in format [[min_y1, min_y2,...], [max_y1, max_y2,...]].
                        If None, bounds are determined from the data.

        Returns:
        - Normalized X and Y tensors scaled to [0,1].
        """
        # Determine input bounds if not provided
        if input_bounds is None:
            x_min, _ = torch.min(X, dim=0)
            x_max, _ = torch.max(X, dim=0)
        else:
            x_min = input_bounds[0]
            x_max = input_bounds[1]

        # Determine output bounds if not provided
        if output_bounds is None:
            y_min, _ = torch.min(Y, dim=0)
            y_max, _ = torch.max(Y, dim=0)
        else:
            y_min = output_bounds[0]
            y_max = output_bounds[1]

        # Add small epsilon to prevent division by zero
        x_range = (x_max - x_min) + 1e-8
        y_range = (y_max - y_min) + 1e-8

        # Min-max normalization to [0,1]
        X_normalized = (X - x_min) / x_range
        Y_normalized = (Y - y_min) / y_range

        # Store the current normalization parameters
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        return X_normalized, Y_normalized

    def denormalize_minmax_data(self, X_norm, Y_norm=None):
        """
        Convert normalized data back to original scale.

        Parameters:
        - X_norm: Normalized input data.
        - Y_norm: Normalized output data (optional).

        Returns:
        - Denormalized X and Y tensors (Y only if provided).
        """
        # Denormalize X
        X = X_norm * (self.x_max - self.x_min) + self.x_min

        # Denormalize Y if provided
        if Y_norm is not None:
            Y = Y_norm * (self.y_max - self.y_min) + self.y_min
            return X, Y

        return X

    def normalize_data(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize input and output data."""
        # Compute mean and std for X and Y
        x_mean, x_std = X.mean(dim=0), X.std(dim=0)
        y_mean, y_std = Y.mean(dim=0), Y.std(dim=0)

        # Z-score normalization
        X_normalized = (X - x_mean) / x_std
        Y_normalized = (Y - y_mean) / y_std

        # Store normalization parameters
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

        return X_normalized, Y_normalized

    def normalize_minmax_output(self, Y):
        """
        Normalize predictions to the original scale.

        Parameters:
        - Y: Normalized predictions.

        Returns:
        - Denormalized Y tensor.
        """
        if self.y_min is not None and self.y_max is not None:
            return (Y - self.y_min) / (self.y_max - self.y_min)
        return Y

    def normalize_output(self, Y):
        """
        Normalize predictions to the original scale.

        Parameters:
        - Y: Normalized predictions.

        Returns:
        - Denormalized Y tensor.
        """
        if self.y_mean is not None and self.y_std is not None:
            return (Y - self.y_mean) / self.y_std
        return Y

    def build_model(self, X_train: torch.Tensor, y_train: torch.Tensor, if_noise=False):
        """Build GP model based on specified type."""
        if self.model_type == 'SVGP':
            model = SVGPModel(self.inducing_points, input_dim=self.dim)
        elif self.model_type == 'ArdGP':
            model = ArdGPModel(X_train, y_train, self.likelihood)
        elif self.model_type == 'CustomGP':
            if if_noise:
                model = CustomGPModel(X_train, y_train, self.likelihood_new, self.x_dim - self.context_dim, self.context_dim)
            else:
                model = CustomGPModel(X_train, y_train, self.likelihood, self.x_dim - self.context_dim, self.context_dim)
        else:
            if if_noise:
                model = ExactGPModel(X_train, y_train, self.likelihood_new)
            else:
                model = ExactGPModel(X_train, y_train, self.likelihood)

        self.model = model
        return model

    def update_context_best_values(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            contexts: torch.Tensor
    ):
        """Update best values for each context."""
        for context in contexts:
            context_key = tuple(context.numpy())

            # Find all points with this context
            context_mask = torch.all(X[:, self.x_dim:] == context, dim=1)
            if torch.any(context_mask):
                context_values = Y[context_mask]
                current_best = context_values.min().item()

                # Update if better than previous best or no previous best exists
                if context_key not in self.context_best_values:
                    self.context_best_values[context_key] = []
                self.context_best_values[context_key].append(current_best)

    def optimize(
            self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            contexts: torch.Tensor,
            n_iter: int = 50,
            beta: float = 2.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[tuple, List[float]]]:

        # Initialize best values tracking
        self.update_context_best_values(X_train, y_train, contexts)

        for iteration in range(n_iter):
            # Normalize data
            # TODO: The normalization is conducted uniformly for all contexts
            X_train_norm, y_train_norm = self.normalize_data(X_train.clone(), y_train.clone())

            # Build and train model
            model = self.build_model(X_train_norm, y_train_norm)
            model.train()
            self.likelihood.train()

            # Set up optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=0.01) if self.optimizer_type == 'adam' else torch.optim.LBFGS(
                model.parameters(), lr=0.1, max_iter=20)

            if self.model_type == 'SVGP':
                mll = gpytorch.mlls.VariationalELBO(self.likelihood, model, num_data=y_train.size(0))
            else:
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)

            # Train the model
            if self.optimizer_type == 'lbfgs':
                prev_loss = float('inf')
                for _ in range(20):
                    def closure():
                        optimizer.zero_grad()
                        output = model(X_train_norm)
                        loss = -mll(output, y_train_norm)
                        loss.backward()
                        return loss

                    curr_loss = optimizer.step(closure)
                    if abs(prev_loss - curr_loss.item()) < 1e-5:
                        break
                    prev_loss = curr_loss.item()

            else:
                prev_loss = float('inf')
                for _ in range(self.train_steps):
                    optimizer.zero_grad()
                    output = model(X_train_norm)
                    loss = -mll(output, y_train_norm)
                    loss.backward()
                    optimizer.step()
                    prev_loss = loss.item()

            # Optimize acquisition for each context
            next_points = []
            next_values = []

            for context in contexts:
                # Find best x for this context
                next_x = optimize_acquisition_for_context(
                    model=model,
                    likelihood=self.likelihood,
                    context=context,
                    x_dim=self.x_dim,
                    beta=beta,
                    x_mean=self.x_mean,
                    x_std=self.x_std
                )

                # Evaluate objective
                x_c = torch.cat([next_x, context])
                next_y = self.objective_func.evaluate(x_c)

                next_points.append(x_c)
                next_values.append(next_y)

            # Stack new points and values
            next_points = torch.stack(next_points)
            next_values = torch.stack(next_values)

            # Update training data
            X_train = torch.cat([X_train, next_points])
            y_train = torch.cat([y_train, next_values])

            # Update best values per context
            self.update_context_best_values(next_points, next_values, contexts)

            if iteration % 3 == 0:
                print(f'Iteration {iteration}/{n_iter}')
                for context in contexts:
                    context_key = tuple(context.numpy())
                    print(f'Context {context_key}: Best value = {self.context_best_values[context_key][-1]:.3f}')

            self.model = model

        return X_train, y_train, self.context_best_values


class ContextualMultiObjectiveBayesianOptimization:
    def __init__(
            self,
            objective_func,
            reference_point: torch.Tensor = None,
            inducing_points: Optional[torch.Tensor] = None,
            train_steps: int = 200,
            model_type: str = 'ExactGP',
            optimizer_type: str = 'adam',
            rho: float = 0.001
    ):
        self.objective_func = objective_func
        self.input_dim = objective_func.input_dim
        self.context_dim = objective_func.context_dim
        self.dim = self.input_dim + self.context_dim
        self.output_dim = objective_func.output_dim
        self.model_type = model_type
        self.contexts = None
        # TODO: What is the intention for base_beta/beta/rho?
        self.base_beta = None
        self.beta = None
        self.rho = rho

        # Initialize reference point if not provided
        if reference_point is None:
            self.reference_point = torch.zeros(self.output_dim)
        else:
            self.reference_point = reference_point

        # Context-specific reference and nadir points
        self.current_reference_points = {}
        self.global_reference_point = None
        self.current_nadir_points = {}
        self.global_nadir_point = None

        self.nadir_point = self.objective_func.nadir_point
        self.hv = Hypervolume(ref_point=self.nadir_point.numpy())
        self.current_hv = -1

        if SCALAR == "AT":
            self.scalarization = AugmentedTchebycheff(
                reference_point=self.reference_point,
                rho=self.rho
            )
        else:
            self.scalarization = HypervolumeScalarization(
                nadir_point=self.nadir_point,
                exponent=self.output_dim
            )

        # Setup the upper threshold for the training steps
        # new_train_steps = min(max(600, self.dim * train_steps), 1000)
        # new_train_steps = min(max(600, self.dim * train_steps), 750)
        new_train_steps = 1250
        self.new_train_steps = new_train_steps

        # Create individual BO models for each objective
        self.bo_models = []
        for _ in range(self.output_dim):
            # Create a wrapper single-objective function for each output dimension
            single_obj = PseudoObjectiveFunction(
                func=lambda x, dim=_: self.objective_func.evaluate(x)[:, dim],
                dim=self.dim,
                context_dim=self.context_dim
            )

            bo = ContextualBayesianOptimization(
                objective_func=single_obj,
                inducing_points=inducing_points,
                train_steps=new_train_steps,
                model_type=model_type,
                optimizer_type=optimizer_type
            )
            self.bo_models.append(bo)

        # Initialize hypervolume calculator
        # self.hv = Hypervolume(ref_point=self.reference_point.numpy())

        # Dictionary to track Pareto fronts and hypervolumes per context
        self.context_pareto_fronts = {}
        self.context_pareto_sets = {}
        self.context_hv = {}
        self.model_list = []
        self.predictions = []

    def _update_context_reference_and_nadir_points(self, context, Y_context):
        """
        Update reference and nadir points for a specific context
        """
        context_key = tuple(context.numpy())

        # Initialize if not exist
        self.current_reference_points[context_key] = torch.min(Y_context, dim=0)[0]
        self.current_reference_points[context_key] = self.current_reference_points[context_key] - 0.1

        self.current_nadir_points[context_key] = torch.max(Y_context, dim=0)[0]
        self.current_nadir_points[context_key] = self.current_nadir_points[context_key] + 0.1 * torch.abs(
            self.current_nadir_points[context_key])

        # Normalize the nadir points and reference points
        for ind in range(self.output_dim):
            self.current_reference_points[context_key][ind] = self.bo_models[ind].normalize_output(
                self.current_reference_points[context_key][ind])
            self.current_nadir_points[context_key][ind] = self.bo_models[ind].normalize_output(
                self.current_nadir_points[context_key][ind])

    def _update_global_reference_and_nadir_points(self, Y_train):
        self.global_reference_point = torch.min(Y_train, dim=0)[0] - 0.1
        self.global_nadir_point = torch.max(Y_train, dim=0)[0] + 0.1 * torch.abs(
            torch.max(Y_train, dim=0)[0])

        for ind in range(self.output_dim):
            self.global_reference_point[ind] = self.bo_models[ind].normalize_output(
                self.global_reference_point[ind])
            self.global_nadir_point[ind] = self.bo_models[ind].normalize_output(
                self.global_nadir_point[ind])

    def _update_beta(self, iteration):
        self.beta = math.sqrt(self.base_beta * log(1 + 2 * iteration))

    def _compute_acquisition_batch(self, predictions, log_sampled_points, beta, weights, context):
        """Combine multiple acquisition values using scalarization"""
        context_key = tuple(context.numpy())
        x_norm = log_sampled_points

        # Get acquisition values for each objective
        acq_values = []
        for model in predictions:
            acq_value = ucb_acquisition_group(model["model"], model["likelihood"], x_norm, beta)
            acq_values.append(torch.tensor(acq_value))

        # Stack and scalarize
        stacked_acq = torch.stack(acq_values, dim=-1)

        if SCALAR == "AT":
            self.scalarization = AugmentedTchebycheff(
                reference_point=self.current_reference_points[context_key],
                rho=self.rho
            )
        else:
            self.scalarization = HypervolumeScalarization(
                nadir_point=self.current_nadir_points[context_key],
                exponent=self.output_dim
            )

        scalarized = self.scalarization(stacked_acq, weights)

        return scalarized.numpy()  # Negative for minimization
    
    def _sample_from_gp_posterior(self, predictions, X_sample):
        """
        Sample from GP posterior (Thompson sampling).
        
        Args:
            predictions: List of GP models for each objective
            X_sample: Points to sample at [n_samples, dim]
        
        Returns:
            Sampled function values [n_samples, n_objectives]
        """
        sampled_values = []
        
        for model_dict in predictions:
            model = model_dict["model"]
            likelihood = model_dict["likelihood"]
            
            model.eval()
            likelihood.eval()
            
            with torch.no_grad():
                # Get posterior distribution
                output = model(X_sample)
                posterior = likelihood(output)
                
                # Sample from posterior
                sample = posterior.sample()  # [n_samples]
                sampled_values.append(sample)
        
        # Stack: [n_samples, n_objectives]
        return torch.stack(sampled_values, dim=-1)
    
    def _select_candidate_with_thompson_sampling(self, predictions, candidates, full_candidates, 
                                                  context, weights, use_global_reference=None):
        """
        Select best candidate using Thompson sampling (OCMOBO acquisition function).
        
        Args:
            predictions: List of GP models for each objective
            candidates: Action candidates [n_candidates, input_dim]
            full_candidates: Full candidates with context [n_candidates, dim]
            context: Context vector
            weights: Weight vector for scalarization
            use_global_reference: Whether to use global reference/nadir points (if None, uses SCALAR/IF_GLOBAL)
        
        Returns:
            best_candidate: Best action candidate [input_dim]
            best_idx: Index of best candidate
        """
        context_key = tuple(context.numpy())
        
        # Normalize inputs
        x_mean = self.bo_models[0].x_mean
        x_std = self.bo_models[0].x_std
        full_candidates_norm = (full_candidates - x_mean) / x_std
        
        # Sample from GP posterior at these points
        sampled_f = self._sample_from_gp_posterior(predictions, full_candidates_norm)
        # sampled_f: [n_candidates, n_objectives]
        
        # Determine scalarization settings
        if use_global_reference is None:
            # Try to use instance variable first, fall back to global constant
            use_global = getattr(self, 'IF_GLOBAL', IF_GLOBAL) if hasattr(self, 'IF_GLOBAL') else IF_GLOBAL
        else:
            use_global = use_global_reference
        
        scalar_type = getattr(self, 'SCALAR', SCALAR) if hasattr(self, 'SCALAR') else SCALAR
        
        # Scalarize sampled values
        if scalar_type == "AT":
            if use_global:
                scalarization = AugmentedTchebycheff(
                    reference_point=self.global_reference_point,
                    rho=self.rho
                )
            else:
                scalarization = AugmentedTchebycheff(
                    reference_point=self.current_reference_points[context_key],
                    rho=self.rho
                )
        else:
            if use_global:
                scalarization = HypervolumeScalarization(
                    nadir_point=self.global_nadir_point,
                    exponent=self.output_dim
                )
            else:
                scalarization = HypervolumeScalarization(
                    nadir_point=self.current_nadir_points[context_key],
                    exponent=self.output_dim
                )
        
        # Scalarize: [n_candidates]
        scalarized_sampled = scalarization(sampled_f, weights.unsqueeze(0).expand(sampled_f.shape[0], -1))
        
        # Get best candidate (minimization)
        best_idx = torch.argmin(scalarized_sampled)
        best_candidate = candidates[best_idx]
        
        return best_candidate, best_idx
    
    def _optimize_with_thompson_sampling_for_context(self, predictions, context, weights, 
                                                      n_candidates=1000, use_global_reference=None):
        """
        Optimize using Thompson sampling for traditional acquisition (fallback case).
        Follows the same strategy as OCMOBO's _select_action_for_context.
        
        Args:
            predictions: List of GP models for each objective
            context: Context vector
            weights: Weight vector for scalarization
            n_candidates: Number of random candidates to generate (default: 1000, OCMOBO uses 500)
            use_global_reference: Whether to use global reference/nadir points
        
        Returns:
            best_action: Best action for this context [input_dim]
        """
        context_key = tuple(context.numpy())
        
        # Generate random action candidates (same as OCMOBO)
        action_candidates = torch.rand(n_candidates, self.input_dim)
        
        # Create full input: [action, context] (same as OCMOBO)
        full_inputs = torch.cat([
            action_candidates,
            context.unsqueeze(0).expand(n_candidates, -1)
        ], dim=1)  # [n_candidates, dim]
        
        # Normalize inputs using the same approach as OCMOBO
        # OCMOBO uses self.x_mean/self.x_std computed from X_train
        # We use bo_models[0].x_mean/x_std which should be equivalent
        x_mean = self.bo_models[0].x_mean
        x_std = self.bo_models[0].x_std
        full_inputs_norm = (full_inputs - x_mean) / x_std
        
        # Sample from GP posterior at these points (same as OCMOBO)
        sampled_f = self._sample_from_gp_posterior(predictions, full_inputs_norm)
        # sampled_f: [n_candidates, n_objectives]
        
        # Determine scalarization settings (same logic as OCMOBO)
        if use_global_reference is None:
            # Try to use instance variable first, fall back to global constant
            use_global = getattr(self, 'IF_GLOBAL', IF_GLOBAL) if hasattr(self, 'IF_GLOBAL') else IF_GLOBAL
        else:
            use_global = use_global_reference
        
        scalar_type = getattr(self, 'SCALAR', SCALAR) if hasattr(self, 'SCALAR') else SCALAR
        
        # Scalarize sampled values (same as OCMOBO)
        if scalar_type == "AT":
            if use_global:
                scalarization = AugmentedTchebycheff(
                    reference_point=self.global_reference_point,
                    rho=self.rho
                )
            else:
                scalarization = AugmentedTchebycheff(
                    reference_point=self.current_reference_points[context_key],
                    rho=self.rho
                )
        else:
            if use_global:
                scalarization = HypervolumeScalarization(
                    nadir_point=self.global_nadir_point,
                    exponent=self.output_dim
                )
            else:
                scalarization = HypervolumeScalarization(
                    nadir_point=self.current_nadir_points[context_key],
                    exponent=self.output_dim
                )
        
        # Scalarize: [n_candidates] (same as OCMOBO)
        scalarized_sampled = scalarization(sampled_f, weights.unsqueeze(0).expand(n_candidates, -1))
        
        # Get best action for this context (minimization, same as OCMOBO)
        best_action_idx = torch.argmin(scalarized_sampled)
        selected_action = action_candidates[best_action_idx]
        
        return selected_action

    @staticmethod
    def _sample_points(n_points, n_decision_vars, n_context_vars):
        """Generate a sample of points for monitoring predictions"""
        total_dims = n_decision_vars + n_context_vars
        return torch.rand(n_points, total_dims)

    @staticmethod
    def _generate_weight_vector(dim: int) -> torch.Tensor:
        """Generate a random weight vector from a Dirichlet distribution."""
        alpha = torch.ones(dim)  # Symmetric Dirichlet distribution
        weights = torch.distributions.Dirichlet(alpha).sample()
        return weights

    def _update_pareto_front_for_context(self, X: torch.Tensor, Y: torch.Tensor, context: torch.Tensor):
        """Update Pareto front for a specific context."""
        context_key = tuple(context.numpy())

        # Convert to numpy for pymoo
        Y_np = Y.numpy()

        # Get non-dominated sorting
        front = NonDominatedSorting().do(Y_np)[0]

        if context_key not in self.context_hv:
            self.context_hv[context_key] = []
        if context_key not in self.context_pareto_fronts:
            self.context_pareto_fronts[context_key] = []
        if context_key not in self.context_pareto_sets:
            self.context_pareto_sets[context_key] = []

        # Update Pareto Front and Pareto Set
        pareto_front = Y[front]
        pareto_set = X[front]
        self.context_pareto_fronts[context_key].append(pareto_front)
        self.context_pareto_sets[context_key].append(pareto_set)

        # Calculate hypervolume
        hv = self.hv.do(pareto_front.numpy())
        self.context_hv[context_key].append(hv)

    def optimize(
            self,
            X_train: torch.Tensor,
            Y_train: torch.Tensor,
            contexts: torch.Tensor,
            n_iter: int = 50,
            beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self.contexts = contexts
        self.base_beta = beta

        if NOISE:
            Y_train_noise = Y_train + 0.01 * torch.randn_like(Y_train)


        # Initialize tracking for each context
        for context in contexts:
            context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
            if torch.any(context_mask):
                Y_context = Y_train[context_mask]
                X_context = X_train[context_mask][:, :self.input_dim]
                self._update_pareto_front_for_context(X_context, Y_context, context)

        for iteration in range(n_iter):
            self._update_beta(iteration)
            # Generate random weights
            weights = self._generate_weight_vector(self.output_dim)

            # Train models for each objective
            predictions = []
            if iteration % 1 == 0:
                for i, bo_model in enumerate(self.bo_models):
                    if NOISE:
                        X_norm, y_norm = bo_model.normalize_data(
                            X_train.clone(),
                            Y_train_noise[:, i].clone()
                        )
                    else:
                        X_norm, y_norm = bo_model.normalize_data(
                            X_train.clone(),
                            Y_train[:, i].clone()
                        )

                    if iteration > 60:
                        model = bo_model.build_model(X_norm, y_norm, True)
                    else:
                        model = bo_model.build_model(X_norm, y_norm, False)
                    model.train()
                    bo_model.likelihood.train()

                    # Training loop (same as before)
                    optimizer = torch.optim.Adam(model.parameters(),
                                                 lr=0.1) if bo_model.optimizer_type == 'adam' else FullBatchLBFGS(
                        model.parameters())

                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=[int(self.new_train_steps * 0.5), int(self.new_train_steps * 0.75)],
                        gamma=0.1
                    )
                    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    #     optimizer,
                    #     T_max=self.new_train_steps,  # First cycle length
                    #     eta_min=1e-4
                    # )

                    # Definition of likelihood
                    if bo_model.model_type == 'SVGP':
                        mll = gpytorch.mlls.VariationalELBO(
                            bo_model.likelihood,
                            model,
                            num_data=y_norm.size(0)
                        )
                    else:
                        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                            bo_model.likelihood,
                            model
                        )

                    # Training Loop
                    if bo_model.optimizer_type == 'lbfgs':
                        def closure():
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            return loss

                        prev_loss = float('inf')
                        loss = closure()
                        loss.backward()
                        for dummy_range in range(60):
                            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
                            loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)

                    else:
                        prev_loss = float('inf')
                        for _ in range(bo_model.train_steps):
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            prev_loss = loss.item()

                            if _ % 100 == 0:
                                print("current loss is {}".format(prev_loss))

                    bo_model.model = model
                    predictions.append({"model": model, "likelihood": bo_model.likelihood})

                self._update_global_reference_and_nadir_points(Y_train)

                for context_id, context in enumerate(contexts):
                    context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
                    if torch.any(context_mask):
                        if NOISE:
                            Y_context = Y_train_noise[context_mask]
                        else:
                            Y_context = Y_train[context_mask]
                        # X_context = X_train[context_mask][:, :self.input_dim]
                        self._update_context_reference_and_nadir_points(context, Y_context)

            if len(predictions) > 0:
                self.predictions = predictions
            else:
                predictions = self.predictions

            # Optimize for each context
            next_points = []
            next_values = []
            next_values_noise = []

            for context in contexts:
                context_key = tuple(context.numpy())

                if SCALAR == "AT":
                    if IF_GLOBAL:
                        self.scalarization = AugmentedTchebycheff(
                            reference_point=self.global_reference_point,
                            rho=self.rho
                        )
                    else:
                        self.scalarization = AugmentedTchebycheff(
                            reference_point=self.current_reference_points[context_key],
                            rho=self.rho
                        )
                else:
                    if IF_GLOBAL:
                        self.scalarization = HypervolumeScalarization(
                            nadir_point=self.global_nadir_point,
                            exponent=self.output_dim
                        )
                    else:
                        self.scalarization = HypervolumeScalarization(
                            nadir_point=self.current_nadir_points[context_key],
                            exponent=self.output_dim
                        )

                next_x = optimize_scalarized_acquisition_for_context(
                    models=predictions,
                    context=context,
                    x_dim=self.input_dim,
                    scalarization_func=self.scalarization,
                    weights=weights,
                    beta=beta,
                    x_mean=self.bo_models[0].x_mean,
                    x_std=self.bo_models[0].x_std
                )

                x_c = torch.cat([next_x, context])
                # print("x_c shape is:{}".format(x_c.shape))
                next_y = self.objective_func.evaluate(x_c.clone().unsqueeze(0))
                if NOISE:
                    next_y_noise = next_y + 0.01 * torch.randn_like(next_y)

                next_points.append(x_c)
                next_values.append(next_y)
                if NOISE:
                    next_values_noise.append(next_y_noise)

                # Update Pareto front for this context
                # context_mask = torch.all(x_c[self.input_dim:] == context, dim=0)
                # if context_mask:
                #     Y_context = torch.cat([
                #         Y_train[torch.all(X_train[:, self.input_dim:] == context, dim=1)],
                #         next_y.unsqueeze(0)
                #     ])
                #     self._update_pareto_front_for_context(context, Y_context)

            # Update training data
            next_points = torch.stack(next_points)
            next_values = torch.stack(next_values)
            if NOISE:
                next_values_noise = torch.stack(next_values_noise)
                Y_train_noise = torch.cat([Y_train_noise, next_values_noise.squeeze(1)])
            X_train = torch.cat([X_train, next_points])
            Y_train = torch.cat([Y_train, next_values.squeeze(1)])


            for context_id, context in enumerate(contexts):
                context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
                context_key = tuple(context.numpy())
                if torch.any(context_mask):
                    Y_context = Y_train[context_mask]
                    X_context = X_train[context_mask][:, :self.input_dim]
                    self._update_pareto_front_for_context(X_context, Y_context, context)

            if iteration % 5 == 0:
                print(f'Iteration {iteration}/{n_iter}')
                for context in contexts:
                    context_key = tuple(context.numpy())
                    print(f'Context {context_key}:')
                    print(f'  Hypervolume: {self.context_hv[context_key][-1]:.3f}')
                    print(f'  Pareto front size: {len(self.context_pareto_fronts[context_key])}')

        return X_train, Y_train


class VAEEnhancedCMOBO(ContextualMultiObjectiveBayesianOptimization):
    """
    VAE-enhanced Contextual Multi-Objective Bayesian Optimization.
    Extends the base CMOBO class with VAE capabilities for improved exploration.
    """

    def __init__(
            self,
            objective_func,
            reference_point: torch.Tensor = None,
            inducing_points: Optional[torch.Tensor] = None,
            # train_steps: int = 200,
            train_steps: int = 100,
            model_type: str = 'ExactGP',
            optimizer_type: str = 'adam',
            rho: float = 0.001,
            # VAE-specific parameters
            # vae_training_frequency: int = 5,
            vae_training_frequency: int = 2,
            vae_min_data_points: int = 8,
            vae_latent_dim: Optional[int] = None,
            vae_epochs: int = 50,
            vae_batch_size: int = 64,
            # vae_batch_size: int = 32,
            use_noise: bool = False,
            scalar_type: str = "HV",
            use_global_reference: bool = True,
            problem_name: str = None,
            true_conditional: bool = True,
            acquisition_type: str = "UCB",  # "UCB" or "TS" (Thompson Sampling)
            top_p: float = 0.1,  # Top percentage of solutions to retrieve for VAE training
            vae_num_candidates: int = 30000  # Number of candidates to generate via VAE
    ):
        # Initialize the parent class
        super().__init__(
            objective_func=objective_func,
            reference_point=reference_point,
            inducing_points=inducing_points,
            train_steps=train_steps,
            model_type=model_type,
            optimizer_type=optimizer_type,
            rho=rho
        )

        # VAE-specific parameters
        self.vae_training_frequency = vae_training_frequency
        self.vae_min_data_points = vae_min_data_points
        self.vae_latent_dim = vae_latent_dim or max(2, self.output_dim - 1)
        self.vae_epochs = vae_epochs
        self.vae_batch_size = vae_batch_size
        self.vae_model = None
        self.problem_name = problem_name
        self.top_p = top_p  # Top percentage for retrieving training data
        self.vae_num_candidates = vae_num_candidates  # Number of VAE-generated candidates

        # Settings - override global constants with instance variables for cleaner design
        self.USE_NOISE = use_noise
        self.SCALAR = scalar_type
        self.IF_GLOBAL = use_global_reference
        self.acquisition_type = acquisition_type.upper()  # "UCB" or "TS"

        # New structure for VAE training data (ranks 1 and 2)
        self.vae_training_sets = {}
        self.vae_training_fronts = {}
        self.vae_training_contexts = {}

        # Store all the available training data so far
        self.X_train = None
        self.Y_train = None

        # True-cVAE: Store the parameter
        self.true_conditional = true_conditional

    def _generate_uniform_weights(self, num_uniform_weights: int, output_dim: int) -> torch.Tensor:
        """
        Generate uniformly distributed weight vectors using Dirichlet distribution.
        Each weight vector sums to 1 and provides good coverage of the preference space.

        Args:
            num_uniform_weights: Number of weight vectors to generate
            output_dim: Dimension of each weight vector (number of objectives)

        Returns:
            torch.Tensor: Shape (num_uniform_weights, output_dim) with each row summing to 1
        """
        import scipy.stats as stats

        if output_dim == 1:
            return torch.ones(num_uniform_weights, 1)

        # Use symmetric Dirichlet distribution with alpha=1 for uniform distribution on simplex
        # Dirichlet(1, 1, ..., 1) gives uniform distribution over the probability simplex
        alpha = [1.0] * output_dim

        # Generate samples using scipy's Dirichlet
        dirichlet_samples = stats.dirichlet.rvs(alpha, size=num_uniform_weights)

        # Convert to torch tensor
        weights = torch.tensor(dirichlet_samples, dtype=torch.float32)

        return weights

    def _update_pareto_front_for_context(self, X: torch.Tensor, Y: torch.Tensor, context: torch.Tensor):
        """
        Override the parent method to additionally collect rank-1 and rank-2 solutions for VAE training.
        """
        context_key = tuple(context.numpy())

        # Create scalarization function based on this weight
        if self.SCALAR == "AT":
            scalarization = AugmentedTchebycheff(
                reference_point=self.global_reference_point,
                rho=self.rho
            )
        else:
            scalarization = HypervolumeScalarization(
                nadir_point=self.global_nadir_point,
                exponent=self.output_dim
            )

        # Convert to numpy for pymoo
        Y_np = Y.numpy()

        # Get non-dominated sorting with multiple fronts
        fronts = NonDominatedSorting().do(Y_np)

        # Initialize regular tracking structures (same as parent class)
        if context_key not in self.context_hv:
            self.context_hv[context_key] = []
        if context_key not in self.context_pareto_fronts:
            self.context_pareto_fronts[context_key] = []
        if context_key not in self.context_pareto_sets:
            self.context_pareto_sets[context_key] = []

        # Initialize VAE training data structures
        if context_key not in self.vae_training_sets:
            self.vae_training_sets[context_key] = []
        if context_key not in self.vae_training_fronts:
            self.vae_training_fronts[context_key] = []
        if context_key not in self.vae_training_contexts:
            self.vae_training_contexts[context_key] = []

        # Update regular Pareto front tracking (rank-1 only) - same as parent class
        pareto_front = Y[fronts[0]]
        pareto_set = X[fronts[0]]
        self.context_pareto_fronts[context_key].append(pareto_front)
        self.context_pareto_sets[context_key].append(pareto_set)

        # Calculate hypervolume using rank-1 solutions only
        hv = self.hv.do(pareto_front.numpy())
        self.context_hv[context_key].append(hv)

        # Collect solutions for VAE training (rank-1 and rank-2)
        vae_sets = []
        vae_fronts = []

        # Include rank-1 solutions
        vae_sets.append(pareto_set)
        vae_fronts.append(pareto_front)

        # Add rank-2 solutions if available
        if len(fronts) > 1 and len(fronts[1]) > 0:
            rank2_front = Y[fronts[1]]
            rank2_set = X[fronts[1]]
            vae_sets.append(rank2_set)
            vae_fronts.append(rank2_front)

        # Combine all solutions
        combined_set = torch.cat(vae_sets) if len(vae_sets) > 0 else torch.tensor([])
        combined_front = torch.cat(vae_fronts) if len(vae_fronts) > 0 else torch.tensor([])

        # Only proceed if we have data to work with
        if len(combined_set) > 0:
            # Compute weight vectors for each solution
            reference_point = self.global_reference_point
            # Use instance variable for top_p
            top_p = self.top_p
            context_mask = torch.all(self.X_train[:, self.input_dim:] == context, dim=1)
            combined_contexts = []
            augmented_vae_sets = []
            augmented_vae_fronts = []
            augmented_contexts = []

            all_X_context = self.X_train[context_mask][:, :self.input_dim]
            all_Y_context = self.Y_train[context_mask]
            num_uniform_weights = combined_front.shape[0]
            uniform_weights = self._generate_uniform_weights(num_uniform_weights, self.output_dim)

            for weight in uniform_weights:
                scalarized_values = scalarization(all_Y_context, weight)
                # Find the indices of the top p% solutions according to this weight vector
                num_to_select = max(1, int(len(scalarized_values) * top_p))
                _, top_indices = torch.topk(scalarized_values, num_to_select, largest=False)
                # Combine context and weight for VAE conditioning
                combined_context = torch.cat([context[1:], weight])

                combined_contexts.append(combined_context)

                # Efficiently select solutions and replicate contexts in one shot
                selected_X = all_X_context[top_indices]
                selected_Y = all_Y_context[top_indices]

                # Replicate the context for each selected solution
                replicated_context = combined_context.unsqueeze(0).expand(num_to_select, -1)

                # Add to our lists
                augmented_vae_sets.append(selected_X)
                augmented_vae_fronts.append(selected_Y)
                augmented_contexts.append(replicated_context)

            for y_value in combined_front:
                weight = self.compute_weight_from_solution(y_value, reference_point, context_key)

                scalarized_values = scalarization(all_Y_context, weight)
                # Find the indices of the top p% solutions according to this weight vector
                num_to_select = max(1, int(len(scalarized_values) * top_p))
                _, top_indices = torch.topk(scalarized_values, num_to_select, largest=False)
                # Combine context and weight for VAE conditioning
                combined_context = torch.cat([context[1:], weight])

                combined_contexts.append(combined_context)

                # Efficiently select solutions and replicate contexts in one shot
                selected_X = all_X_context[top_indices]
                selected_Y = all_Y_context[top_indices]

                # Replicate the context for each selected solution
                replicated_context = combined_context.unsqueeze(0).expand(num_to_select, -1)

                # Add to our lists
                augmented_vae_sets.append(selected_X)
                augmented_vae_fronts.append(selected_Y)
                augmented_contexts.append(replicated_context)

            if len(augmented_vae_sets) > 0:
                augmented_vae_sets = torch.cat(augmented_vae_sets, dim=0)
                augmented_vae_fronts = torch.cat(augmented_vae_fronts, dim=0)
                augmented_contexts = torch.cat(augmented_contexts, dim=0)

                # Store for VAE training
                self.vae_training_sets[context_key].append(augmented_vae_sets)
                self.vae_training_fronts[context_key].append(augmented_vae_fronts)
                self.vae_training_contexts[context_key].append(augmented_contexts)

    def compute_weight_from_solution(self, y, reference_point, context_key):
        """
        Compute weight vector from objective values using the scalarization method.

        For Tchebycheff scalarization: w_i = 1/|f_i - r_i| (normalized)
        For Hypervolume scalarization: w_i = (n_i - f_i) (normalized)

        Args:
            y: Solution's objective values
            reference_point: Reference point for this context
            context_key: Key for the context (used to get nadir point)

        Returns:
            Computed weight vector
        """
        # Check which scalarization method is being used
        if self.SCALAR == "AT":
            # Augmented Tchebycheff scalarization uses reference point
            # For Pareto-optimal solution y, the weights are inversely proportional to |y_i - r_i|
            diff = y - reference_point

            # Avoid division by zero (where y_i = r_i)
            diff = torch.clamp(diff, min=1e-6)

            # Compute weights: w_i = 1/|y_i - r_i|
            weights = 1.0 / diff

        else:
            # Hypervolume scalarization uses nadir point
            nadir_point = self.global_nadir_point

            # For Pareto-optimal solution y, the weights are proportional to (n_i - y_i)
            diff = nadir_point - y

            # Ensure weights are non-negative (nadir should be worse than y, but handle edge cases)
            diff = torch.clamp(diff, min=1e-6)

            # Compute weights: w_i = (n_i - y_i)
            weights = diff

        # Normalize weights to sum to 1
        weights_sum = torch.sum(weights)
        if weights_sum > 1e-10:
            weights = weights / weights_sum
        else:
            # Fallback to uniform weights
            weights = torch.ones_like(y) / len(y)

        return weights

    def initialize_or_update_vae(self, iteration, full_training=False):
        """
        Initialize a new VAE model or update the existing one.

        Args:
            iteration: Current iteration number (for naming)
            full_training: Whether to do full training or incremental update
        """
        # Collect training data from all contexts
        all_X = []
        all_contexts = []

        for context_key in self.vae_training_sets.keys():
            if len(self.vae_training_sets[context_key]) > 0:
                # Get the latest data
                latest_set = self.vae_training_sets[context_key][-1]
                latest_contexts = self.vae_training_contexts[context_key][-1]

                all_X.append(latest_set)
                all_contexts.append(latest_contexts)

        if len(all_X) == 0:
            print("No training data available for VAE")
            return

        # Convert lists to tensors
        X_train = torch.cat(all_X)
        contexts_train = torch.cat(all_contexts)

        if len(X_train) < self.vae_min_data_points:
            print(f"Not enough data points for VAE training ({len(X_train)} < {self.vae_min_data_points})")
            return

        # if self.vae_model is None:
            # Initialize new VAE model
        if self.vae_model is None:
            self.vae_model = ParetoVAETrainer(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                latent_dim=self.vae_latent_dim,
                context_dim=contexts_train.shape[1],  # Context + weights
                conditional=True,
                true_conditional=self.true_conditional,  # True-cVAE: pass the parameter
                epochs=self.vae_epochs,
                batch_size=min(self.vae_batch_size, len(X_train)),
                trainer_id=f"CMOBO_VAE_{iteration}"
            )
            full_training = True  # Always do full training for new model
        else:
            full_training = False

        if full_training:
            # Full training
            print(f"Iteration {iteration}: Training VAE model with {len(X_train)} points...")
            print(f"  Epochs: {self.vae_model.epochs}, Batch size: {self.vae_model.batch_size}")
            self.vae_model.train(X=X_train, contexts=contexts_train)
            print(f"Iteration {iteration}: VAE model training completed.")

        else:
            # Incremental training
            original_epochs = self.vae_model.epochs
            self.vae_model.epochs = max(30, int(original_epochs * 0.3))  # Use fewer epochs for incremental updates
            print(f"Iteration {iteration}: Incrementally updating VAE model with {len(X_train)} points...")
            print(f"  Epochs: {self.vae_model.epochs}, Batch size: {self.vae_model.batch_size}")
            self.vae_model.train(X=X_train, contexts=contexts_train)
            self.vae_model.epochs = original_epochs  # Restore original setting
            print(f"Iteration {iteration}: VAE model incremental update completed.")

    def generate_cvae_candidates(self, context, weight_vector, num_samples=500):
        """
        Generate candidate solutions by sampling from the latent space.
        Optimized for batch evaluation with acquisition functions.

        Args:
            context: Context vector
            weight_vector: Current weight vector
            num_samples: Number of samples to generate

        Returns:
            Tensor of candidate solutions and their full inputs with context
        """
        if self.vae_model is None:
            return None, None

        # Combine context and weight vector
        combined_context = torch.cat([context[1:], weight_vector])

        # Sample latent vectors directly
        # latent_dim = self.vae_model.latent_dim
        # z_samples = torch.randn(num_samples, latent_dim).to(self.vae_model.device)

        # True-cVAE: Use conditional prior if enabled, otherwise standard prior
        if self.true_conditional:
            # Create batch of contexts for conditional sampling
            context_batch = combined_context.unsqueeze(0).expand(num_samples, -1)
            z_samples = self.vae_model.model.sample_from_conditional_prior(context_batch, num_samples=1)
        else:
            # Original sampling from standard prior
            latent_dim = self.vae_model.latent_dim
            z_samples = torch.randn(num_samples, latent_dim).to(self.vae_model.device)

        # Create batch of identical contexts
        context_batch = combined_context.unsqueeze(0).expand(num_samples, -1)

        with torch.no_grad():
            # Generate solutions by decoding the latent vectors
            candidates = self.vae_model.model.inference(z_samples, context_batch)

        # Prepare full inputs including context for acquisition function evaluation (vectorized)
        context_rep = context.unsqueeze(0).expand(num_samples, -1)  # (num_samples, context_dim)
        full_candidates = torch.cat([candidates, context_rep], dim=1)  # (num_samples, input_dim + context_dim)

        return candidates.detach(), full_candidates.detach()

    def optimize(
            self,
            X_train: torch.Tensor,
            Y_train: torch.Tensor,
            contexts: torch.Tensor,
            n_iter: int = 50,
            beta: float = 1.0,
            run: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override the optimize method to incorporate VAE capabilities.
        """
        self.contexts = contexts
        self.base_beta = beta
        self.X_train = X_train.clone()
        self.Y_train = Y_train.clone()

        self.base_dir_name = f"VAE_CMOBO_{self.problem_name}_{self.input_dim}_{self.output_dim}_{self.model_type}_{self.vae_training_frequency}_{run}_0.1"

        # Open timing log file
        timing_log_file = f"{self.base_dir_name}_timing_log.txt"
        timing_log_fh = open(timing_log_file, 'w')
        timing_log_fh.write(f"Runtime Overhead Analysis - VAEEnhancedCMOBO\n")
        timing_log_fh.write(f"Problem: {self.problem_name}, Iterations: {n_iter}\n")
        timing_log_fh.write(f"{'='*100}\n")
        timing_log_fh.write(f"{'Iter':<6} {'Mode':<12} {'GP Train (s)':<15} {'Acq Search (s)':<18} {'Gen Train (s)':<16} {'Gen Sample (s)':<17}\n")
        timing_log_fh.write(f"{'-'*100}\n")

        if self.USE_NOISE:
            Y_train_noise = Y_train + 0.01 * torch.randn_like(Y_train)
        else:
            Y_train_noise = None  # Explicitly set to None if not using noise

        # the update_pareto_front function requires this function call
        # so that the global ref point and nadir point can be obtained
        self._update_global_reference_and_nadir_points(Y_train)
        # Initialize tracking for each context
        for context in contexts:
            context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
            if torch.any(context_mask):
                Y_context = Y_train[context_mask]
                X_context = X_train[context_mask][:, :self.input_dim]
                self._update_pareto_front_for_context(X_context, Y_context, context)

        for iteration in range(n_iter):
            # Initialize timing variables for this iteration
            t_gp_train = 0.0
            t_gen_train = 0.0
            t_acq_search = 0.0
            t_gen_sample = 0.0
            mode = "ACQUISITION"  # Default mode
            self._update_beta(iteration)
            # Generate random weights
            weights = self._generate_weight_vector(self.output_dim)

            # Train models for each objective
            predictions = []
            if iteration % 1 == 0:
                # Time GP training
                t_gp_train_start = time.time()
                for i, bo_model in enumerate(self.bo_models):
                    if self.USE_NOISE:
                        X_norm, y_norm = bo_model.normalize_data(
                            X_train.clone(),
                            Y_train_noise[:, i].clone()
                        )
                    else:
                        X_norm, y_norm = bo_model.normalize_data(
                            X_train.clone(),
                            Y_train[:, i].clone()
                        )

                    if iteration > 60:
                        model = bo_model.build_model(X_norm, y_norm, True)
                    else:
                        model = bo_model.build_model(X_norm, y_norm, False)
                    model.train()
                    bo_model.likelihood.train()

                    # Training loop (same as before)
                    optimizer = torch.optim.Adam(model.parameters(),
                                                 lr=0.1) if bo_model.optimizer_type == 'adam' else FullBatchLBFGS(
                        model.parameters())

                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=[int(self.new_train_steps * 0.5), int(self.new_train_steps * 0.75)],
                        gamma=0.1
                    )

                    # Definition of likelihood
                    if bo_model.model_type == 'SVGP':
                        mll = gpytorch.mlls.VariationalELBO(
                            bo_model.likelihood,
                            model,
                            num_data=y_norm.size(0)
                        )
                    else:
                        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                            bo_model.likelihood,
                            model
                        )

                    # Training Loop
                    if bo_model.optimizer_type == 'lbfgs':
                        def closure():
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            return loss

                        prev_loss = float('inf')
                        loss = closure()
                        loss.backward()
                        for dummy_range in range(60):
                            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
                            loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)

                    else:
                        prev_loss = float('inf')
                        for _ in range(bo_model.train_steps):
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            prev_loss = loss.item()

                            if _ % 100 == 0:
                                print(f"Current loss is {prev_loss}")

                    bo_model.model = model
                    predictions.append({"model": model, "likelihood": bo_model.likelihood})
                # End GP training timing
                t_gp_train = time.time() - t_gp_train_start

                self._update_global_reference_and_nadir_points(Y_train)

                for context_id, context in enumerate(contexts):
                    context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
                    if torch.any(context_mask):
                        if self.USE_NOISE:
                            Y_context = Y_train_noise[context_mask]
                        else:
                            Y_context = Y_train[context_mask]
                        self._update_context_reference_and_nadir_points(context, Y_context)

            if len(predictions) > 0:
                self.predictions = predictions
            else:
                predictions = self.predictions

            # Train or update VAE model if it's time
            if iteration % self.vae_training_frequency == 0:
                # Time generative model training
                t_gen_train_start = time.time()
                # sum(len(front[-1]) if len(front) > 0 else 0
                #     for front in self.vae_training_sets.values()) >= self.vae_min_data_points):`
                # Do full training periodically or incremental otherwise
                # full_training = (self.vae_model is None or
                #                  iteration % (self.vae_training_frequency * 2) == 0)
                full_training = True
                self.initialize_or_update_vae(iteration, full_training)
                # End generative model training timing
                t_gen_train = time.time() - t_gen_train_start

            # Optimize for each context
            print(f"Iteration {iteration}: Optimizing for {len(contexts)} contexts...")
            next_points = []
            next_values = []
            next_values_noise = []

            for ctx_idx, context in enumerate(contexts):
                context_key = tuple(context.numpy())

                # Set up scalarization function
                if self.SCALAR == "AT":
                    if self.IF_GLOBAL:
                        self.scalarization = AugmentedTchebycheff(
                            reference_point=self.global_reference_point,
                            rho=self.rho
                        )
                    else:
                        self.scalarization = AugmentedTchebycheff(
                            reference_point=self.current_reference_points[context_key],
                            rho=self.rho
                        )
                else:
                    if self.IF_GLOBAL:
                        self.scalarization = HypervolumeScalarization(
                            nadir_point=self.global_nadir_point,
                            exponent=self.output_dim
                        )
                    else:
                        self.scalarization = HypervolumeScalarization(
                            nadir_point=self.current_nadir_points[context_key],
                            exponent=self.output_dim
                        )

                # Decide whether to use VAE-generated candidates or traditional acquisition
                use_vae = (self.vae_model is not None and
                           iteration >= self.vae_training_frequency and
                           iteration % self.vae_training_frequency == 0)  # Use VAE every other iteration

                if use_vae:
                    mode = "GENERATIVE"
                    print(f"  Context {ctx_idx+1}/{len(contexts)}: Using VAE-generated candidates")
                    # Generate candidates using VAE with latent perturbation
                    print(f"  Context {ctx_idx+1}/{len(contexts)}: Generating {self.vae_num_candidates} VAE candidates...")
                    # Time generative sampling
                    t_gen_sample_start = time.time()
                    cvae_candidates, full_candidates = self.generate_cvae_candidates(
                        context=context,
                        weight_vector=weights,
                        num_samples=self.vae_num_candidates
                    )

                    if cvae_candidates is not None and len(cvae_candidates) > 0:
                        # Select best candidate using specified acquisition function
                        if self.acquisition_type == "TS":
                            # Use Thompson sampling (OCMOBO acquisition)
                            print(f"    Evaluating Thompson Sampling on {len(cvae_candidates)} candidates (this may take a while)...")
                            next_x, _ = self._select_candidate_with_thompson_sampling(
                                predictions=predictions,
                                candidates=cvae_candidates,
                                full_candidates=full_candidates,
                                context=context,
                                weights=weights,
                                use_global_reference=self.IF_GLOBAL
                            )
                            print(f"    Context {ctx_idx+1} completed.")
                        else:
                            # Use UCB acquisition (default)
                            print(f"    Evaluating UCB acquisition on {len(cvae_candidates)} candidates (this may take a while)...")
                            norm_candidates = (full_candidates - self.bo_models[0].x_mean) / self.bo_models[0].x_std
                            acq_values = self._compute_acquisition_batch(
                                predictions,
                                norm_candidates,
                                self.beta,
                                weights,
                                context
                            )
                            best_idx = torch.argmin(torch.tensor(acq_values))
                            next_x = cvae_candidates[best_idx]
                            print(f"    Context {ctx_idx+1} completed.")
                        # End generative sampling timing
                        t_gen_sample += time.time() - t_gen_sample_start

                    else:
                        # Fall back to traditional acquisition if VAE fails
                        # Close generative sampling timing (it was 0 since generation failed)
                        t_gen_sample += time.time() - t_gen_sample_start
                        # Time acquisition search (fallback)
                        t_acq_search_start = time.time()
                        print(f"  Context {ctx_idx+1}/{len(contexts)}: VAE generation failed, using traditional acquisition...")
                        if self.acquisition_type == "TS":
                            print(f"    Optimizing with Thompson Sampling (n_candidates=10000, this may take a while)...")
                            next_x = self._optimize_with_thompson_sampling_for_context(
                                predictions=predictions,
                                context=context,
                                weights=weights,
                                n_candidates=30000,
                                use_global_reference=self.IF_GLOBAL
                            )
                            print(f"    Context {ctx_idx+1} completed.")
                        else:
                            print(f"    Optimizing scalarized acquisition (this may take a while)...")
                            next_x = optimize_scalarized_acquisition_for_context(
                                models=predictions,
                                context=context,
                                x_dim=self.input_dim,
                                scalarization_func=self.scalarization,
                                weights=weights,
                                beta=beta,
                                x_mean=self.bo_models[0].x_mean,
                                x_std=self.bo_models[0].x_std
                            )
                            print(f"    Context {ctx_idx+1} completed.")
                        # End acquisition search timing (fallback)
                        t_acq_search += time.time() - t_acq_search_start
                else:
                    # Use traditional acquisition optimization
                    mode = "ACQUISITION"
                    # Time acquisition search
                    t_acq_search_start = time.time()
                    if self.acquisition_type == "TS":
                        next_x = self._optimize_with_thompson_sampling_for_context(
                            predictions=predictions,
                            context=context,
                            weights=weights,
                            n_candidates=5000,
                            use_global_reference=self.IF_GLOBAL
                        )
                    else:
                        next_x = optimize_scalarized_acquisition_for_context(
                            models=predictions,
                            context=context,
                            x_dim=self.input_dim,
                            scalarization_func=self.scalarization,
                            weights=weights,
                            beta=beta,
                            x_mean=self.bo_models[0].x_mean,
                            x_std=self.bo_models[0].x_std
                        )
                    # End acquisition search timing
                    t_acq_search += time.time() - t_acq_search_start

                # Evaluate selected point
                x_c = torch.cat([next_x, context])
                next_y = self.objective_func.evaluate(x_c.clone().unsqueeze(0))

                if self.USE_NOISE:
                    next_y_noise = next_y + 0.01 * torch.randn_like(next_y)
                    next_values_noise.append(next_y_noise)

                next_points.append(x_c)
                next_values.append(next_y)

            print(f"Iteration {iteration}: All contexts optimized, updating training data...")
            # Update training data
            next_points = torch.stack(next_points)
            next_values = torch.stack(next_values)

            if self.USE_NOISE:
                next_values_noise = torch.stack(next_values_noise)
                Y_train_noise = torch.cat([Y_train_noise, next_values_noise.squeeze(1)])

            X_train = torch.cat([X_train, next_points])
            Y_train = torch.cat([Y_train, next_values.detach().squeeze(1)])
            self.X_train = X_train.clone()
            self.Y_train = Y_train.clone()

            print(f"Iteration {iteration}: Training data updated, updating Pareto fronts...")
            # Update Pareto fronts for all contexts
            for context_id, context in enumerate(contexts):
                context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
                context_key = tuple(context.numpy())
                if torch.any(context_mask):
                    Y_context = Y_train[context_mask]
                    X_context = X_train[context_mask][:, :self.input_dim]
                    self._update_pareto_front_for_context(X_context, Y_context, context)

            if iteration % 5 == 0:
                print(f'Iteration {iteration}/{n_iter}')
                for context in contexts:
                    context_key = tuple(context.numpy())
                    print(f'Context {context_key}:')
                    print(f'  Hypervolume: {self.context_hv[context_key][-1]:.3f}')
                    print(f'  Pareto front size: {len(self.context_pareto_fronts[context_key][-1])}')
                    # print(
                    #     f'  VAE training data size: {len(self.vae_training_sets[context_key][-1]) if context_key in self.vae_training_sets and len(self.vae_training_sets[context_key]) > 0 else 0}')

                # Add VAE-specific logging
                # if self.vae_model is not None:
                #     print(f'VAE model status:')
                #     print(f'  Latent dimension: {self.vae_model.latent_dim}')
                #     print(
                #         f'  Last trained: iteration {(iteration // self.vae_training_frequency) * self.vae_training_frequency}')
            
            # Print timing log at the end of each iteration and write to file
            timing_log_line = f"[Iter {iteration}] Mode: {mode} | GP Train: {t_gp_train:.4f}s | Acq Search: {t_acq_search:.4f}s | Gen Train: {t_gen_train:.4f}s | Gen Sample: {t_gen_sample:.4f}s"
            print(timing_log_line)
            timing_log_fh.write(f"{iteration:<6} {mode:<12} {t_gp_train:<15.4f} {t_acq_search:<18.4f} {t_gen_train:<16.4f} {t_gen_sample:<17.4f}\n")
            timing_log_fh.flush()  # Ensure data is written immediately

        # Close timing log file
        timing_log_fh.close()

        return X_train, Y_train


class SimpleDiffusionContextualMOBO(ContextualMultiObjectiveBayesianOptimization):
    """
    Simplified Diffusion-enhanced Contextual Multi-Objective Bayesian Optimization.
    Uses simple DDPM instead of DDIM for better stability and simplicity.
    """

    def __init__(
            self,
            objective_func,
            reference_point: torch.Tensor = None,
            inducing_points: Optional[torch.Tensor] = None,
            train_steps: int = 200,
            model_type: str = 'ExactGP',
            optimizer_type: str = 'adam',
            rho: float = 0.001,
            # Simple DDPM-specific parameters
            diffusion_training_frequency: int = 1,
            diffusion_min_data_points: int = 8,
            diffusion_timesteps: int = 100,
            diffusion_epochs: int = 50,
            diffusion_batch_size: int = 64,
            # diffusion_hidden_dim: int = 128,
            diffusion_hidden_dim: int = 64,
            # diffusion_num_layers: int = 4,
            diffusion_num_layers: int = 3,
            use_noise: bool = False,
            scalar_type: str = "HV",
            use_global_reference: bool = True,
            problem_name: str = None,
            acquisition_type: str = "UCB",  # "UCB" or "TS" (Thompson Sampling)
            top_p: float = 0.1,  # Top percentage of solutions to retrieve for DDPM training
            use_batch_norm: bool = False,  # Whether to use batch normalization in DDPM model
            ddpm_num_candidates: int = 15000  # Number of candidates to generate via DDPM
    ):
        # Initialize the parent class
        super().__init__(
            objective_func=objective_func,
            reference_point=reference_point,
            inducing_points=inducing_points,
            train_steps=train_steps,
            model_type=model_type,
            optimizer_type=optimizer_type,
            rho=rho
        )

        # Simple DDPM-specific parameters
        self.diffusion_training_frequency = diffusion_training_frequency
        self.diffusion_min_data_points = diffusion_min_data_points
        self.diffusion_timesteps = diffusion_timesteps
        self.diffusion_epochs = diffusion_epochs
        self.diffusion_batch_size = diffusion_batch_size
        self.diffusion_hidden_dim = diffusion_hidden_dim
        self.diffusion_num_layers = diffusion_num_layers
        self.use_batch_norm = use_batch_norm
        self.diffusion_model = None
        self.problem_name = problem_name
        self.top_p = top_p  # Top percentage for retrieving training data
        self.ddpm_num_candidates = ddpm_num_candidates  # Number of DDPM-generated candidates

        # Settings
        self.USE_NOISE = use_noise
        self.SCALAR = scalar_type
        self.IF_GLOBAL = use_global_reference
        self.acquisition_type = acquisition_type.upper()  # "UCB" or "TS"

        # Structure for diffusion training data (same as your original approach)
        self.diffusion_training_sets = {}
        self.diffusion_training_fronts = {}
        self.diffusion_training_contexts = {}

        # Store all available training data
        self.X_train = None
        self.Y_train = None

    def _generate_uniform_weights(self, num_uniform_weights: int, output_dim: int) -> torch.Tensor:
        """Generate uniformly distributed weight vectors using Dirichlet distribution."""
        import scipy.stats as stats

        if output_dim == 1:
            return torch.ones(num_uniform_weights, 1)

        # Use symmetric Dirichlet distribution with alpha=1 for uniform distribution on simplex
        alpha = [1.0] * output_dim
        dirichlet_samples = stats.dirichlet.rvs(alpha, size=num_uniform_weights)
        weights = torch.tensor(dirichlet_samples, dtype=torch.float32)

        return weights

    def _update_pareto_front_for_context(self, X: torch.Tensor, Y: torch.Tensor, context: torch.Tensor):
        """
        Override to collect rank-1 and rank-2 solutions for simple DDPM training.
        """
        context_key = tuple(context.numpy())

        # Create scalarization function based on this weight
        if self.SCALAR == "AT":
            scalarization = AugmentedTchebycheff(
                reference_point=self.global_reference_point,
                rho=self.rho
            )
        else:
            scalarization = HypervolumeScalarization(
                nadir_point=self.global_nadir_point,
                exponent=self.output_dim
            )

        # Convert to numpy for pymoo
        Y_np = Y.numpy()

        # Get non-dominated sorting with multiple fronts
        fronts = NonDominatedSorting().do(Y_np)

        # Initialize regular tracking structures (same as parent class)
        if context_key not in self.context_hv:
            self.context_hv[context_key] = []
        if context_key not in self.context_pareto_fronts:
            self.context_pareto_fronts[context_key] = []
        if context_key not in self.context_pareto_sets:
            self.context_pareto_sets[context_key] = []

        # Initialize diffusion training data structures
        if context_key not in self.diffusion_training_sets:
            self.diffusion_training_sets[context_key] = []
        if context_key not in self.diffusion_training_fronts:
            self.diffusion_training_fronts[context_key] = []
        if context_key not in self.diffusion_training_contexts:
            self.diffusion_training_contexts[context_key] = []

        # Update regular Pareto front tracking (rank-1 only)
        pareto_front = Y[fronts[0]]
        pareto_set = X[fronts[0]]
        self.context_pareto_fronts[context_key].append(pareto_front)
        self.context_pareto_sets[context_key].append(pareto_set)

        # Calculate hypervolume using rank-1 solutions only
        hv = self.hv.do(pareto_front.numpy())
        self.context_hv[context_key].append(hv)

        # Collect solutions for diffusion training (rank-1 and rank-2)
        diffusion_sets = []
        diffusion_fronts = []

        # Include rank-1 solutions
        diffusion_sets.append(pareto_set)
        diffusion_fronts.append(pareto_front)

        # Add rank-2 solutions if available
        if len(fronts) > 1 and len(fronts[1]) > 0:
            rank2_front = Y[fronts[1]]
            rank2_set = X[fronts[1]]
            diffusion_sets.append(rank2_set)
            diffusion_fronts.append(rank2_front)

        # Combine all solutions
        combined_set = torch.cat(diffusion_sets) if len(diffusion_sets) > 0 else torch.tensor([])
        combined_front = torch.cat(diffusion_fronts) if len(diffusion_fronts) > 0 else torch.tensor([])

        # Only proceed if we have data to work with
        if len(combined_set) > 0:
            # Compute weight vectors for each solution
            reference_point = self.global_reference_point
            top_p = self.top_p
            context_mask = torch.all(self.X_train[:, self.input_dim:] == context, dim=1)
            combined_contexts = []
            augmented_diffusion_sets = []
            augmented_diffusion_fronts = []
            augmented_contexts = []

            all_X_context = self.X_train[context_mask][:, :self.input_dim]
            all_Y_context = self.Y_train[context_mask]
            num_uniform_weights = combined_front.shape[0]
            uniform_weights = self._generate_uniform_weights(num_uniform_weights ** (self.output_dim - 1),
                                                             self.output_dim)

            # Process uniform weights
            for weight in uniform_weights:
                scalarized_values = scalarization(all_Y_context, weight)
                num_to_select = max(1, int(len(scalarized_values) * top_p))
                _, top_indices = torch.topk(scalarized_values, num_to_select, largest=False)

                # Combine context and weight for conditioning (simplified concatenation)
                combined_context = torch.cat([context[1:], weight])
                combined_contexts.append(combined_context)

                # Select solutions and replicate contexts
                selected_X = all_X_context[top_indices]
                selected_Y = all_Y_context[top_indices]
                replicated_context = combined_context.unsqueeze(0).expand(num_to_select, -1)

                augmented_diffusion_sets.append(selected_X)
                augmented_diffusion_fronts.append(selected_Y)
                augmented_contexts.append(replicated_context)

            # Process combined front solutions
            for y_value in combined_front:
                weight = self.compute_weight_from_solution(y_value, reference_point, context_key)

                scalarized_values = scalarization(all_Y_context, weight)
                num_to_select = max(1, int(len(scalarized_values) * top_p))
                _, top_indices = torch.topk(scalarized_values, num_to_select, largest=False)

                # Combine context and weight for conditioning
                combined_context = torch.cat([context[1:], weight])
                combined_contexts.append(combined_context)

                # Select solutions and replicate contexts
                selected_X = all_X_context[top_indices]
                selected_Y = all_Y_context[top_indices]
                replicated_context = combined_context.unsqueeze(0).expand(num_to_select, -1)

                augmented_diffusion_sets.append(selected_X)
                augmented_diffusion_fronts.append(selected_Y)
                augmented_contexts.append(replicated_context)

            if len(augmented_diffusion_sets) > 0:
                augmented_diffusion_sets = torch.cat(augmented_diffusion_sets, dim=0)
                augmented_diffusion_fronts = torch.cat(augmented_diffusion_fronts, dim=0)
                augmented_contexts = torch.cat(augmented_contexts, dim=0)

                # Store for diffusion training
                self.diffusion_training_sets[context_key].append(augmented_diffusion_sets)
                self.diffusion_training_fronts[context_key].append(augmented_diffusion_fronts)
                self.diffusion_training_contexts[context_key].append(augmented_contexts)

    def compute_weight_from_solution(self, y, reference_point, context_key):
        """Compute weight vector from objective values using the scalarization method."""
        if self.SCALAR == "AT":
            # Augmented Tchebycheff scalarization
            diff = y - reference_point
            diff = torch.clamp(diff, min=1e-6)
            weights = 1.0 / diff
        else:
            # Hypervolume scalarization
            nadir_point = self.global_nadir_point
            diff = nadir_point - y
            diff = torch.clamp(diff, min=1e-6)
            weights = diff

        # Normalize weights to sum to 1
        weights_sum = torch.sum(weights)
        if weights_sum > 1e-10:
            weights = weights / weights_sum
        else:
            weights = torch.ones_like(y) / len(y)

        return weights

    def initialize_or_update_diffusion(self, iteration, full_training=False):
        """
        Initialize a new simple DDPM model or update the existing one.
        """
        # Collect training data from all contexts
        all_X = []
        all_contexts = []

        for context_key in self.diffusion_training_sets.keys():
            if len(self.diffusion_training_sets[context_key]) > 0:
                # Get the latest data
                latest_set = self.diffusion_training_sets[context_key][-1]
                latest_contexts = self.diffusion_training_contexts[context_key][-1]

                all_X.append(latest_set)
                all_contexts.append(latest_contexts)

        if len(all_X) == 0:
            print("No training data available for simple DDPM model")
            return

        # Convert lists to tensors
        X_train = torch.cat(all_X)
        contexts_train = torch.cat(all_contexts)

        if len(X_train) < self.diffusion_min_data_points:
            print(
                f"Not enough data points for simple DDPM training ({len(X_train)} < {self.diffusion_min_data_points})")
            return

        # Calculate conditioning dimension
        conditioning_dim = contexts_train.shape[1]  # Context + weights combined

        # Initialize new simple DDPM model if needed
        if self.diffusion_model is None:
            self.diffusion_model = SimpleParetoTrainer(
                input_dim=self.input_dim,
                conditioning_dim=conditioning_dim,
                timesteps=self.diffusion_timesteps,
                hidden_dim=self.diffusion_hidden_dim,
                num_layers=self.diffusion_num_layers,
                epochs=self.diffusion_epochs,
                batch_size=min(self.diffusion_batch_size, len(X_train)),
                trainer_id=f"CMOBO_SimpleDDPM_{iteration}",
                use_batch_norm=self.use_batch_norm
            )
            full_training = True  # Always do full training for new model
        else:
            full_training = False

        if full_training:
            # Full training
            print(f"Iteration {iteration}: Training Simple DDPM model with {len(X_train)} points...")
            print(f"  Epochs: {self.diffusion_model.epochs}, Batch size: {self.diffusion_model.batch_size}")
            self.diffusion_model.train(X=X_train, contexts=contexts_train)
            print(f"Iteration {iteration}: Simple DDPM model training completed.")
        else:
            # Incremental training
            original_epochs = self.diffusion_model.epochs
            self.diffusion_model.epochs = max(30,
                                              int(original_epochs * 0.3))  # Use fewer epochs for incremental updates
            print(f"Iteration {iteration}: Incrementally updating Simple DDPM model with {len(X_train)} points...")
            print(f"  Epochs: {self.diffusion_model.epochs}, Batch size: {self.diffusion_model.batch_size}")
            self.diffusion_model.train(X=X_train, contexts=contexts_train)
            self.diffusion_model.epochs = original_epochs  # Restore original setting
            print(f"Iteration {iteration}: Simple DDPM model incremental update completed.")

    def generate_diffusion_candidates(self, context, weight_vector, num_samples=500):
        if self.diffusion_model is None:
            return None, None

        device = context.device
        combined_context = torch.cat([context[1:], weight_vector], dim=0)  # (C',)

        context_batch = combined_context.unsqueeze(0).expand(num_samples, -1)  # (B, C')

        candidates = torch.from_numpy(
            self.diffusion_model.generate_solutions(
                contexts=context_batch.detach().cpu().numpy(),
                num_samples=num_samples
            )
        ).to(device=device, dtype=torch.float32)  # (B, D)

        # full_candidate = [candidate, context]
        context_rep = context.unsqueeze(0).expand(num_samples, -1)  # (B, C)
        full_candidates = torch.cat([candidates, context_rep], dim=1)  # (B, D+C)

        return candidates.detach(), full_candidates.detach()

    def optimize(
            self,
            X_train: torch.Tensor,
            Y_train: torch.Tensor,
            contexts: torch.Tensor,
            n_iter: int = 50,
            beta: float = 1.0,
            run: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override the optimize method to incorporate simple DDPM capabilities.
        """
        self.contexts = contexts
        self.base_beta = beta
        self.X_train = X_train.clone()
        self.Y_train = Y_train.clone()

        self.base_dir_name = f"SimpleDiffusion_CMOBO_{self.problem_name}_{self.input_dim}_{self.output_dim}_{self.model_type}_{self.diffusion_training_frequency}_{run}_0.1"

        # Open timing log file
        timing_log_file = f"{self.base_dir_name}_timing_log.txt"
        timing_log_fh = open(timing_log_file, 'w')
        timing_log_fh.write(f"Runtime Overhead Analysis - SimpleDiffusionContextualMOBO\n")
        timing_log_fh.write(f"Problem: {self.problem_name}, Iterations: {n_iter}\n")
        timing_log_fh.write(f"{'='*100}\n")
        timing_log_fh.write(f"{'Iter':<6} {'Mode':<12} {'GP Train (s)':<15} {'Acq Search (s)':<18} {'Gen Train (s)':<16} {'Gen Sample (s)':<17}\n")
        timing_log_fh.write(f"{'-'*100}\n")

        if self.USE_NOISE:
            Y_train_noise = Y_train + 0.01 * torch.randn_like(Y_train)
        else:
            Y_train_noise = None

        # Initialize global reference and nadir points
        self._update_global_reference_and_nadir_points(Y_train)

        # Initialize tracking for each context
        for context in contexts:
            context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
            if torch.any(context_mask):
                Y_context = Y_train[context_mask]
                X_context = X_train[context_mask][:, :self.input_dim]
                self._update_pareto_front_for_context(X_context, Y_context, context)

        for iteration in range(n_iter):
            # Initialize timing variables for this iteration
            t_gp_train = 0.0
            t_gen_train = 0.0
            t_acq_search = 0.0
            t_gen_sample = 0.0
            mode = "ACQUISITION"  # Default mode
            self._update_beta(iteration)
            # Generate random weights
            weights = self._generate_weight_vector(self.output_dim)

            # Train models for each objective (same as original implementation)
            predictions = []
            if iteration % 1 == 0:
                # Time GP training
                t_gp_train_start = time.time()
                for i, bo_model in enumerate(self.bo_models):
                    if self.USE_NOISE:
                        X_norm, y_norm = bo_model.normalize_data(
                            X_train.clone(),
                            Y_train_noise[:, i].clone()
                        )
                    else:
                        X_norm, y_norm = bo_model.normalize_data(
                            X_train.clone(),
                            Y_train[:, i].clone()
                        )

                    if iteration > 60:
                        model = bo_model.build_model(X_norm, y_norm, True)
                    else:
                        model = bo_model.build_model(X_norm, y_norm, False)
                    model.train()
                    bo_model.likelihood.train()

                    # Training loop (same as original)
                    optimizer = torch.optim.Adam(model.parameters(),
                                                 lr=0.1) if bo_model.optimizer_type == 'adam' else FullBatchLBFGS(
                        model.parameters())

                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=[int(self.new_train_steps * 0.5), int(self.new_train_steps * 0.75)],
                        gamma=0.1
                    )

                    # Define likelihood
                    if bo_model.model_type == 'SVGP':
                        mll = gpytorch.mlls.VariationalELBO(
                            bo_model.likelihood,
                            model,
                            num_data=y_norm.size(0)
                        )
                    else:
                        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                            bo_model.likelihood,
                            model
                        )

                    # Training Loop
                    if bo_model.optimizer_type == 'lbfgs':
                        def closure():
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            return loss

                        prev_loss = float('inf')
                        loss = closure()
                        loss.backward()
                        for dummy_range in range(60):
                            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
                            loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
                    else:
                        prev_loss = float('inf')
                        for _ in range(bo_model.train_steps):
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            prev_loss = loss.item()

                            if _ % 100 == 0:
                                print(f"Current loss is {prev_loss}")

                    bo_model.model = model
                    predictions.append({"model": model, "likelihood": bo_model.likelihood})

                # End GP training timing
                t_gp_train = time.time() - t_gp_train_start

                self._update_global_reference_and_nadir_points(Y_train)

                for context_id, context in enumerate(contexts):
                    context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
                    if torch.any(context_mask):
                        if self.USE_NOISE:
                            Y_context = Y_train_noise[context_mask]
                        else:
                            Y_context = Y_train[context_mask]
                        self._update_context_reference_and_nadir_points(context, Y_context)

            if len(predictions) > 0:
                self.predictions = predictions
            else:
                predictions = self.predictions

            # Train or update simple DDPM model if it's time
            if iteration % self.diffusion_training_frequency == 0:
                # Time generative model training
                t_gen_train_start = time.time()
                # Do full training periodically
                full_training = True
                self.initialize_or_update_diffusion(iteration, full_training)
                # End generative model training timing
                t_gen_train = time.time() - t_gen_train_start

            # Optimize for each context
            print(f"Iteration {iteration}: Optimizing for {len(contexts)} contexts...")
            next_points = []
            next_values = []
            next_values_noise = []

            for ctx_idx, context in enumerate(contexts):
                context_key = tuple(context.numpy())

                # Set up scalarization function
                if self.SCALAR == "AT":
                    if self.IF_GLOBAL:
                        self.scalarization = AugmentedTchebycheff(
                            reference_point=self.global_reference_point,
                            rho=self.rho
                        )
                    else:
                        self.scalarization = AugmentedTchebycheff(
                            reference_point=self.current_reference_points[context_key],
                            rho=self.rho
                        )
                else:
                    if self.IF_GLOBAL:
                        self.scalarization = HypervolumeScalarization(
                            nadir_point=self.global_nadir_point,
                            exponent=self.output_dim
                        )
                    else:
                        self.scalarization = HypervolumeScalarization(
                            nadir_point=self.current_nadir_points[context_key],
                            exponent=self.output_dim
                        )

                # Decide whether to use simple DDPM-generated candidates or traditional acquisition
                use_diffusion = (self.diffusion_model is not None and
                                 iteration >= self.diffusion_training_frequency and
                                 iteration % self.diffusion_training_frequency == 0)

                if use_diffusion:
                    mode = "GENERATIVE"
                    # Generate candidates using simple DDPM sampling
                    print(f"  Context {ctx_idx+1}/{len(contexts)}: Generating {self.ddpm_num_candidates} Simple DDPM candidates...")
                    # Time generative sampling
                    t_gen_sample_start = time.time()
                    diffusion_candidates, full_candidates = self.generate_diffusion_candidates(
                        context=context,
                        weight_vector=weights,
                        num_samples=self.ddpm_num_candidates
                    )

                    if diffusion_candidates is not None and len(diffusion_candidates) > 0:
                        # Select best candidate using specified acquisition function
                        if self.acquisition_type == "TS":
                            # Use Thompson sampling (OCMOBO acquisition)
                            print(f"    Evaluating Thompson Sampling on {len(diffusion_candidates)} candidates (this may take a while)...")
                            next_x, _ = self._select_candidate_with_thompson_sampling(
                                predictions=predictions,
                                candidates=diffusion_candidates,
                                full_candidates=full_candidates,
                                context=context,
                                weights=weights,
                                use_global_reference=self.IF_GLOBAL
                            )
                            print(f"    Context {ctx_idx+1} completed.")
                        else:
                            # Use UCB acquisition (default)
                            print(f"    Evaluating UCB acquisition on {len(diffusion_candidates)} candidates (this may take a while)...")
                            norm_candidates = (full_candidates - self.bo_models[0].x_mean) / self.bo_models[0].x_std
                            acq_values = self._compute_acquisition_batch(
                                predictions,
                                norm_candidates,
                                self.beta,
                                weights,
                                context
                            )
                            best_idx = torch.argmin(torch.tensor(acq_values))
                            next_x = diffusion_candidates[best_idx]
                            print(f"    Context {ctx_idx+1} completed.")
                        # End generative sampling timing
                        t_gen_sample += time.time() - t_gen_sample_start

                    else:
                        # Fall back to traditional acquisition if simple DDPM fails
                        # Close generative sampling timing (it was 0 since generation failed)
                        t_gen_sample += time.time() - t_gen_sample_start
                        # Time acquisition search (fallback)
                        t_acq_search_start = time.time()
                        print(f"  Context {ctx_idx+1}/{len(contexts)}: Simple DDPM generation failed, using traditional acquisition...")
                        if self.acquisition_type == "TS":
                            print(f"    Optimizing with Thompson Sampling (n_candidates=10000, this may take a while)...")
                            next_x = self._optimize_with_thompson_sampling_for_context(
                                predictions=predictions,
                                context=context,
                                weights=weights,
                                n_candidates=self.ddpm_num_candidates,
                                use_global_reference=self.IF_GLOBAL
                            )
                            print(f"    Context {ctx_idx+1} completed.")
                        else:
                            print(f"    Optimizing scalarized acquisition (this may take a while)...")
                            next_x = optimize_scalarized_acquisition_for_context(
                                models=predictions,
                                context=context,
                                x_dim=self.input_dim,
                                scalarization_func=self.scalarization,
                                weights=weights,
                                beta=beta,
                                x_mean=self.bo_models[0].x_mean,
                            x_std=self.bo_models[0].x_std
                        )
                            print(f"    Context {ctx_idx+1} completed.")
                        # End acquisition search timing (fallback)
                        t_acq_search += time.time() - t_acq_search_start
                else:
                    # Use traditional acquisition optimization
                    mode = "ACQUISITION"
                    # Time acquisition search
                    t_acq_search_start = time.time()
                    print(f"  Context {ctx_idx+1}/{len(contexts)}: Using traditional acquisition optimization...")
                    if self.acquisition_type == "TS":
                        print(f"    Optimizing with Thompson Sampling (n_candidates=10000, this may take a while)...")
                        next_x = self._optimize_with_thompson_sampling_for_context(
                            predictions=predictions,
                            context=context,
                            weights=weights,
                            n_candidates=5000,
                            use_global_reference=self.IF_GLOBAL
                        )
                        print(f"    Context {ctx_idx+1} completed.")
                    else:
                        print(f"    Optimizing scalarized acquisition (this may take a while)...")
                        next_x = optimize_scalarized_acquisition_for_context(
                            models=predictions,
                            context=context,
                            x_dim=self.input_dim,
                            scalarization_func=self.scalarization,
                            weights=weights,
                            beta=beta,
                            x_mean=self.bo_models[0].x_mean,
                            x_std=self.bo_models[0].x_std
                        )
                        print(f"    Context {ctx_idx+1} completed.")
                    # End acquisition search timing
                    t_acq_search += time.time() - t_acq_search_start

                # Evaluate selected point
                x_c = torch.cat([next_x, context])
                next_y = self.objective_func.evaluate(x_c.clone().unsqueeze(0))

                if self.USE_NOISE:
                    next_y_noise = next_y + 0.01 * torch.randn_like(next_y)
                    next_values_noise.append(next_y_noise)

                next_points.append(x_c)
                next_values.append(next_y)

            print(f"Iteration {iteration}: All contexts optimized, updating training data...")
            # Update training data
            next_points = torch.stack(next_points)
            next_values = torch.stack(next_values)

            if self.USE_NOISE:
                next_values_noise = torch.stack(next_values_noise)
                Y_train_noise = torch.cat([Y_train_noise, next_values_noise.squeeze(1)])

            X_train = torch.cat([X_train, next_points])
            Y_train = torch.cat([Y_train, next_values.detach().squeeze(1)])
            self.X_train = X_train.clone()
            self.Y_train = Y_train.clone()

            print(f"Iteration {iteration}: Training data updated, updating Pareto fronts...")
            # Update Pareto fronts for all contexts
            for context_id, context in enumerate(contexts):
                context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
                context_key = tuple(context.numpy())
                if torch.any(context_mask):
                    Y_context = Y_train[context_mask]
                    X_context = X_train[context_mask][:, :self.input_dim]
                    self._update_pareto_front_for_context(X_context, Y_context, context)

            if iteration % 5 == 0:
                print(f'Iteration {iteration}/{n_iter}')
                for context in contexts:
                    context_key = tuple(context.numpy())
                    print(f'Context {context_key}:')
                    print(f'  Hypervolume: {self.context_hv[context_key][-1]:.3f}')
                    print(f'  Pareto front size: {len(self.context_pareto_fronts[context_key][-1])}')
            
            # Print timing log at the end of each iteration and write to file
            timing_log_line = f"[Iter {iteration}] Mode: {mode} | GP Train: {t_gp_train:.4f}s | Acq Search: {t_acq_search:.4f}s | Gen Train: {t_gen_train:.4f}s | Gen Sample: {t_gen_sample:.4f}s"
            print(timing_log_line)
            timing_log_fh.write(f"{iteration:<6} {mode:<12} {t_gp_train:<15.4f} {t_acq_search:<18.4f} {t_gen_train:<16.4f} {t_gen_sample:<17.4f}\n")
            timing_log_fh.flush()  # Ensure data is written immediately

        # Close timing log file
        timing_log_fh.close()

        return X_train, Y_train


