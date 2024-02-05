import torch
import numpy as np
from TaskGenerator import augment_support_set
from utils import func_call, DEVICE


# -------------------------------------------------------------------
class MAMLAug:
    def __init__(self, model, loss_fn, lr_inner, nb_augs, lr_outer=0.001, adapt_steps=1, with_weights=False):
        self.model = model
        self.loss_fn = loss_fn
        self.lr_inner = lr_inner
        self.adapt_steps = adapt_steps
        self.with_weights = with_weights

        self.theta = dict(self.model.named_parameters())

        if self.with_weights:
            # Initialize weights with 1/(NB_AUGS+1)
            # initial_value = 1.0 / (nb_augs+1)
            # self.weights = torch.nn.parameter.Parameter(initial_value*torch.ones(nb_augs + 1).to(DEVICE).float())

            # Initialize weights with 1
            self.weights = torch.nn.parameter.Parameter(torch.ones(nb_augs+1).to(DEVICE).float())  # one weight value per each augmented signal

            meta_params = list(self.theta.values()) + [self.weights]
        else:
            meta_params = list(self.theta.values())

        self.optimizer = torch.optim.Adam(meta_params, lr=lr_outer)

    # -------------------------------------------------------------------
    def adapt(self, params_dict, tsk):
        y_sp_pred, _ = func_call(self.model, params_dict, tsk)

        chunked_y_sp_pred, chunked_y_sp = y_sp_pred.chunk(tsk.nb_augs), tsk.y_sp.chunk(tsk.nb_augs)
        inner_losses = torch.stack([self.loss_fn(y_pred, y_true) for (y_pred, y_true) in zip(chunked_y_sp_pred, chunked_y_sp)])

        if self.with_weights:
            weights = torch.nn.functional.sigmoid(self.weights)
            inner_loss = torch.sum(inner_losses * weights)
        else:
            inner_loss = torch.sum(inner_losses)

        grads = torch.autograd.grad(inner_loss, params_dict.values(), create_graph=True)
        adapted_params_dict = {name: w - self.lr_inner * w_grad for (name, w), w_grad in zip(params_dict.items(), grads)}

        return adapted_params_dict

    # -------------------------------------------------------------------
    def get_adapted_parameters(self, tsk):
        phi = self.adapt(self.theta, tsk)
        for _ in range(self.adapt_steps - 1):
            phi = self.adapt(phi, tsk)
        return phi

    # -------------------------------------------------------------------
    def fit(self, tgen, steps=10000):
        for step in range(steps):
            tsk = tgen.batch()
            tsk = augment_support_set(tsk)

            phi = self.get_adapted_parameters(tsk)
            _, y_qr_pred = func_call(self.model, phi, tsk)
            loss = self.loss_fn(y_qr_pred, tsk.y_qr)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (step + 1) % 50 == 0:
                if self.with_weights:
                    w_arr = self.weights.detach().cpu().numpy()
                    print(f"Step: {step + 1}, loss: {loss.item():.5f}, w: {(min(w_arr), max(w_arr), np.mean(w_arr))}", end="\t\r")
                else:
                    print(f"Step: {step + 1}, loss: {loss.item():.5f}", end="\t\r")

        if self.with_weights:
            self.model.weights = self.weights.detach()

        return self
