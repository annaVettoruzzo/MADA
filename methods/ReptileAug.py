import torch
import numpy as np, copy
from TaskGenerator import augment_support_set
from utils import func_call, DEVICE


class Reptile():
    def __init__(self, model, loss_fn, lr_inner, nb_augs, adapt_steps=10, eps=0.1):
        self.model = model  # An instance of SimpleNN()
        self.loss_fn = loss_fn  # An instance of SimpleNN()
        self.adapt_steps = max(2, adapt_steps)  # Number of GD adaptation steps (to get task specific parameters)
        self.eps = eps  # 0 < epsilon << 1, for interpolation
        self.weights = torch.nn.parameter.Parameter(torch.ones(nb_augs + 1).to(DEVICE).float()) #one weight value per each augmented signal

        self.inner_opt = torch.optim.SGD(model.parameters(), lr=lr_inner)
        self.optimizer = torch.optim.Adam(self.weights, lr=1e-3)

    # -------------------------------------------------------------------
    def fit(self, tgen, steps=10000):
        for step in range(steps):
            # Sample a training task (no need for separate support/query sets)
            X_sp, y_sp, X_qr, y_qr = tgen.batch()

            # Parameters before adaptation
            theta = copy.deepcopy(self.model.state_dict())

            # Adapt the model to this task (using several gradient steps)
            for _ in range(self.adapt_steps):
                y_pred_sp, y_pred_qr = self.model(X_sp, X_qr)
                loss = self.loss_fn(y_pred_sp, y_sp) + self.loss_fn(y_pred_qr, y_qr)
                self.inner_opt.zero_grad()
                loss.backward()
                self.inner_opt.step()

            # Parameters after adaptation (i.e. task specific parameters)
            params = self.model.state_dict()

            # Interpolate between the meta-parameters (theta) and the task specific parameters (params)
            dico = {name: theta[name] + self.eps * (params[name] - theta[name]) for name in theta.keys()}
            self.model.load_state_dict(dico)

            if (step + 1) % 50 == 0:
                print(f"Step: {step + 1}", end="\t\r")
