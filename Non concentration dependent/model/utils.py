import torch

def _groups_batch_from_a2g_local(xg_local: torch.Tensor,
                                 a2g_local: torch.Tensor,
                                 batch_sub: torch.Tensor) -> torch.Tensor:

    device = xg_local.device
    Gm = xg_local.size(0)
    if Gm == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    group_batch_self = torch.full((Gm,), -1, dtype=torch.long, device=device)
    if a2g_local.numel() > 0:
        g = a2g_local[0]
        a = a2g_local[1]
        group_batch_self[g] = batch_sub[a]

    if (group_batch_self < 0).any():
        default_b = batch_sub.mode()[0] if batch_sub.numel() > 0 else torch.tensor(0, device=device)
        group_batch_self[group_batch_self < 0] = default_b

    present = torch.unique(batch_sub)
    id_map = -torch.ones(int(present.max().item()) + 1, dtype=torch.long, device=device)
    id_map[present] = torch.arange(present.numel(), device=device)
    compact = id_map[group_batch_self]
    return compact


