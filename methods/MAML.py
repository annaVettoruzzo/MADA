from utils import func_call
import torch


# -------------------------------------------------------------------
class MAML:
    def __init__(self, model, loss_fn, lr_inner, lr_outer=0.001, adapt_steps=1):
        self.model = model
        self.loss_fn = loss_fn
        self.lr_inner = lr_inner
        self.adapt_steps = adapt_steps

        self.theta = dict(self.model.named_parameters())
        meta_params = list(self.theta.values())
        self.optimizer = torch.optim.Adam(meta_params, lr=lr_outer)

        # -------------------------------------------------------------------

    def adapt(self, params_dict, tsk):
        y_sp_pred, _ = func_call(self.model, params_dict, tsk)
        inner_loss = self.loss_fn(y_sp_pred, tsk.y_sp)

        grads = torch.autograd.grad(inner_loss, params_dict.values())
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

            phi = self.get_adapted_parameters(tsk)
            _, y_qr_pred = func_call(self.model, phi, tsk)
            loss = self.loss_fn(y_qr_pred, tsk.y_qr)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (step + 1) % 50 == 0:
                print(f"Step: {step + 1}, loss: {loss.item():.5f}", end="\t\r")
        return self
