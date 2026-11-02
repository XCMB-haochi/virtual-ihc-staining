import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from ihc_single import MarkerHeadStainer

torch.set_num_threads(2)
torch.manual_seed(2026)
model=MarkerHeadStainer().eval()
x=torch.rand(1,1,32,32)
with torch.no_grad():
    before=[model(x,torch.tensor([i])) for i in (0,1)]
state=torch.get_rng_state().clone()
model.split_output_head()
assert torch.equal(state,torch.get_rng_state())
with torch.no_grad():
    for i in (0,1): assert torch.equal(before[i],model(x,torch.tensor([i])))
for i in (0,1):
    model.zero_grad(set_to_none=True)
    calls=[]
    hooks=[h.register_forward_hook(lambda m,a,y,j=j:calls.append(j)) for j,h in model.marker_heads.items()]
    model(x,torch.tensor([i])).mean().backward()
    assert calls==[str(i)]
    assert all(p.grad is None for p in model.marker_heads[str(1-i)].parameters())
    assert all(p.grad is not None for p in model.marker_heads[str(i)].parameters())
    for h in hooks: h.remove()
saved=model.state_dict()
restored=MarkerHeadStainer(); restored.split_output_head(); restored.load_state_dict(saved,strict=True)
assert all(torch.equal(saved[k],v) for k,v in restored.state_dict().items())
print('Initial output/RNG identity, one active head, gradient isolation and strict reload passed')
