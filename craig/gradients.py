"""Per-sample gradient engine (Phase 2).

g_i = grad_theta loss(f(x_i; theta), y_i), stacked into an [N, D] matrix.
Primary path: torch.func.vmap(torch.func.grad(...)) over functional_call.
Fallback: micro-batch loop with one backward per sample (true per-sample,
never a batch gradient reused per sample).

Note: gradients are computed with the model in eval() mode so BatchNorm/
Dropout behave deterministically per sample (BN uses running statistics).
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _unpack(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    return batch["input"], batch["target"]


def select_param_names(model: nn.Module, layers: Union[str, List[str]] = "all") -> List[str]:
    """Resolve which parameters participate in the gradient representation.

    layers: 'all' | 'last' (final nn.Linear, i.e. classifier head) | list of name prefixes.
    """
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if layers == "all":
        return names
    if layers in ("last", "last_layer", "classifier"):
        last = None
        for mname, m in model.named_modules():
            if isinstance(m, nn.Linear):
                last = mname
        if last is None:
            raise ValueError("No nn.Linear layer found for 'last' layer selection.")
        prefix = f"{last}." if last else ""
        return [n for n in names if n.startswith(prefix)]
    if isinstance(layers, (list, tuple)):
        sel = [n for n in names if any(n == l or n.startswith(l + ".") or n.startswith(l) for l in layers)]
        if not sel:
            raise ValueError(f"No parameters match {layers}")
        return sel
    raise ValueError(f"Unknown layers spec: {layers}")


def _vmap_grads(model, names, x, y, loss_fn):
    from torch.func import functional_call, grad, vmap
    params = {n: p.detach() for n, p in model.named_parameters()}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    sel = {n: params[n] for n in names}
    frozen = {n: p for n, p in params.items() if n not in sel}

    def loss_single(sel_params, xi, yi):
        out = functional_call(model, ({**frozen, **sel_params}, buffers), (xi.unsqueeze(0),))
        return loss_fn(out, yi.unsqueeze(0))

    g = vmap(grad(loss_single), in_dims=(None, 0, 0))(sel, x, y)
    return torch.cat([g[n].reshape(x.shape[0], -1) for n in names], dim=1)


def _loop_grads(model, names, x, y, loss_fn):
    pmap = dict(model.named_parameters())
    plist = [pmap[n] for n in names]
    rows = []
    for i in range(x.shape[0]):
        out = model(x[i:i + 1])
        loss = loss_fn(out, y[i:i + 1])
        gs = torch.autograd.grad(loss, plist)
        rows.append(torch.cat([g.reshape(-1) for g in gs]))
    return torch.stack(rows)


def compute_per_sample_gradients(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn,
    device: Union[str, torch.device] = "cpu",
    layers: Union[str, List[str]] = "all",
    chunk_size: int = 128,
    max_samples: Optional[int] = None,
    method: str = "auto",
    return_meta: bool = False,
):
    """Return per-sample gradients [N, D] (on CPU, float32).

    loss_fn must reduce to a scalar mean over a batch (e.g. nn.CrossEntropyLoss()).
    chunk_size bounds the number of samples vectorised at once (memory guard).
    """
    device = torch.device(device)
    model = model.to(device)
    was_training = model.training
    model.eval()
    names = select_param_names(model, layers)
    pmap = dict(model.named_parameters())
    shapes = [(n, tuple(pmap[n].shape)) for n in names]
    out_rows, labels, n_seen, used = [], [], 0, method
    for batch in dataloader:
        x, y = _unpack(batch)
        if max_samples is not None:
            rem = max_samples - n_seen
            if rem <= 0:
                break
            x, y = x[:rem], y[:rem]
        for s in range(0, x.shape[0], chunk_size):
            xc = x[s:s + chunk_size].to(device)
            yc = y[s:s + chunk_size].to(device)
            if used in ("auto", "vmap"):
                try:
                    g = _vmap_grads(model, names, xc, yc, loss_fn)
                    used = "vmap"
                except Exception:
                    if method == "vmap":
                        raise
                    used = "loop"
                    g = _loop_grads(model, names, xc, yc, loss_fn)
            else:
                g = _loop_grads(model, names, xc, yc, loss_fn)
            out_rows.append(g.detach().float().cpu())
            labels.append(yc.cpu())
        n_seen += x.shape[0]
    model.train(was_training)
    G = torch.cat(out_rows, dim=0)
    if return_meta:
        return G, {"param_names": names, "shapes": shapes, "method": used, "labels": torch.cat(labels)}
    return G
