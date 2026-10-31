"""Named deterministic random streams for the paired experiment."""

from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch


def stable_seed(*parts) -> int:
    payload = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


@contextmanager
def isolated_global_rng(*parts):
    """Temporarily seed libraries used by legacy GP/model code, then restore."""
    seed = stable_seed(*parts)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield seed
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
