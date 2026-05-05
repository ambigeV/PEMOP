import torch
import numpy as np


class ContextualMultiObjectiveFunction:
    def __init__(self, func_name='dtlz1', n_objectives=2, n_variables=None,
                 bounds=(0, 1), context_dim=None):
        self.func_name = func_name.lower()
        self.n_objectives = n_objectives

        if self.func_name not in ('dtlz1', 'dtlz2', 'dtlz3'):
            raise ValueError("Only 'dtlz1', 'dtlz2', 'dtlz3' are supported in PEMOP.")

        default_k = {'dtlz1': 5, 'dtlz2': 5, 'dtlz3': 5}[self.func_name]

        if n_variables is None:
            self.n_variables = n_objectives + default_k - 1
        else:
            self.n_variables = n_variables

        self.k = self.n_variables - n_objectives + 1
        self.context_dim = 2  # Fixed 2D context for DTLZ

        self.x_dim = self.n_variables
        self.bounds = bounds
        self.input_dim = self.n_variables
        self.output_dim = self.n_objectives

        self.nadir_point = {
            'dtlz1': (160 + 100 * (self.n_variables - 2)) * torch.ones(self.n_objectives),
            'dtlz2': torch.ones(self.n_objectives) * 2.0,
            'dtlz3': 90 * (self.n_variables + self.n_variables) * torch.ones(self.n_objectives),
        }[self.func_name]

    def scale_x(self, x):
        min_bound, max_bound = self.bounds
        return min_bound + (max_bound - min_bound) * x

    def get_context_shift(self, c):
        return 0 * c[:, 0]

    def get_context_power(self, c):
        return 0.8 + 0.2 * c[:, 1]

    def g_dtlz1(self, x_m, c):
        c_shift = self.get_context_shift(c)
        c_power = self.get_context_power(c)
        x_shifted = torch.pow(x_m - c_shift.unsqueeze(-1), c_power.unsqueeze(-1))
        return 100 * (x_m.shape[1] + torch.sum(
            (x_shifted - 0.5) ** 2 - torch.cos(20 * np.pi * (x_shifted - 0.5)), dim=1
        ))

    def g_dtlz2(self, x_m, c):
        c_shift = self.get_context_shift(c)
        c_power = self.get_context_power(c)
        x_shifted = torch.pow(x_m - c_shift.unsqueeze(-1), c_power.unsqueeze(-1))
        return torch.sum((x_shifted - 0.5) ** 2, dim=1)

    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs[:, :self.n_variables]
        c = inputs[:, self.n_variables:]
        x_scaled = self.scale_x(x)

        if self.func_name == 'dtlz1':
            return self._contextual_dtlz1(x_scaled, c)
        elif self.func_name == 'dtlz2':
            return self._contextual_dtlz2(x_scaled, c)
        elif self.func_name == 'dtlz3':
            return self._contextual_dtlz3(x_scaled, c)
        else:
            raise ValueError("Unsupported function.")

    def _contextual_dtlz1(self, x, c):
        x_p = x[:, :self.n_objectives - 1]
        x_m = x[:, self.n_objectives - 1:]
        g = self.g_dtlz1(x_m, c)
        power = self.get_context_power(c)
        f = torch.zeros((x.shape[0], self.n_objectives))
        for i in range(self.n_objectives):
            f[:, i] = 0.5 * (1 + g)
            for j in range(self.n_objectives - 1 - i):
                f[:, i] = f[:, i] * torch.pow(x_p[:, j], power)
            if i > 0:
                f[:, i] = f[:, i] * (1 - torch.pow(x_p[:, self.n_objectives - 1 - i], power))
        return f

    def _contextual_dtlz2(self, x, c):
        x_p = x[:, :self.n_objectives - 1]
        x_m = x[:, self.n_objectives - 1:]
        g = self.g_dtlz2(x_m, c)
        power = self.get_context_power(c)
        f = torch.zeros((x.shape[0], self.n_objectives))
        for i in range(self.n_objectives):
            f[:, i] = 1 + g
            for j in range(self.n_objectives - 1 - i):
                f[:, i] = f[:, i] * torch.cos(torch.pow(x_p[:, j], power) * np.pi / 2)
            if i > 0:
                f[:, i] = f[:, i] * torch.sin(
                    torch.pow(x_p[:, self.n_objectives - 1 - i], power) * np.pi / 2
                )
        return f

    def _contextual_dtlz3(self, x, c):
        x_p = x[:, :self.n_objectives - 1]
        x_m = x[:, self.n_objectives - 1:]
        g = self.g_dtlz1(x_m, c)
        power = self.get_context_power(c)
        f = torch.zeros((x.shape[0], self.n_objectives))
        for i in range(self.n_objectives):
            f[:, i] = 1 + g
            for j in range(self.n_objectives - 1 - i):
                f[:, i] = f[:, i] * torch.cos(torch.pow(x_p[:, j], power) * np.pi / 2)
            if i > 0:
                f[:, i] = f[:, i] * torch.sin(
                    torch.pow(x_p[:, self.n_objectives - 1 - i], power) * np.pi / 2
                )
        return f
