from dl.configs import AWP_Config
import torch  # type: ignore


class AWP:
    def __init__(
        self,
        model: torch.nn.Module,
        config: AWP_Config
    ):
        self.model = model
        self.config = config

        self.weight_backup: dict[str, torch.Tensor] = {}
        self.weight_range: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _iter_params(self):
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None or not name.endswith("weight"):
                continue
            if not any(pat in name for pat in self.config.apply_to_names_with):
                continue
            # print(f"[AWP] will perturb param '{name}'")
            yield name, p

    def backup_and_perturb(self):
        """Call after first backward, before second forward."""
        self.weight_backup.clear()
        self.weight_range.clear()

        eps = self.config.eps

        # backup & setup eps bounds
        for name, p in self._iter_params():
            data = p.data
            self.weight_backup[name] = data.clone()
            # per-parameter epsilon proportional to |w|
            grad_eps = eps * data.abs()
            self.weight_range[name] = (data - grad_eps, data + grad_eps)

        e = 1e-6
        # apply adversarial step
        for name, p in self._iter_params():
            grad = p.grad
            if grad is None:
                continue
            grad_norm = torch.norm(grad)
            if grad_norm == 0:
                continue
            assert not torch.isnan(grad_norm)

            w = p.data
            w_norm = torch.norm(w)
            step = self.config.lr * grad / (grad_norm + e) * (w_norm + e)
            p.data = p.data + step

            low, high = self.weight_range[name]
            p.data.clamp_(min=low, max=high)

    def restore(self):
        """Restore weights after second backward."""
        for name, p in self._iter_params():
            if name in self.weight_backup:
                p.data.copy_(self.weight_backup[name])

        self.weight_backup.clear()
        self.weight_range.clear()
