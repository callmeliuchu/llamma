import torch
import numpy as np
from optimizer import AdamW

seed = 0


def test_optimizer(opt_class) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        correct_bias=True,
    )
    for i in range(1000):
        opt.zero_grad()
        x = torch.FloatTensor(rng.uniform(size=[model.in_features]))
        y_hat = model(x)
        y = torch.Tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


ref = torch.tensor(np.load("optimizer_test.npy"))
actual = test_optimizer(AdamW)
<<<<<<< HEAD
<<<<<<< HEAD
print('true',ref)
print('calcuted',actual)
assert torch.allclose(ref, actual)
=======
assert torch.allclose(ref, actual, atol=1e-5, rtol=1e-3)
>>>>>>> 2951cddba218b7ddfe69295b77ad51a05107d384
=======
assert torch.allclose(ref, actual, atol=1e-5, rtol=1e-3)
print('true',ref)
print('calcuted',actual)
assert torch.allclose(ref, actual)
>>>>>>> 428e453cf76a940b246d3c9786edb7672de7ebcc
print("Optimizer test passed!")
