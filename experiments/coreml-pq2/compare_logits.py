import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("--output", type=Path)
args = p.parse_args()
a = np.fromfile(str(args.baseline)+".logits.f32", dtype=np.float32).astype(np.float64).reshape(9, -1)
b = np.fromfile(str(args.candidate)+".logits.f32", dtype=np.float32).astype(np.float64).reshape(9, -1)
assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
assert Path(str(args.baseline)+".tokens.i32").read_bytes() == Path(str(args.candidate)+".tokens.i32").read_bytes()
def logsoftmax(x):
    x = x-x.max(axis=1, keepdims=True)
    return x-np.log(np.exp(x).sum(axis=1, keepdims=True))
la, lb = logsoftmax(a), logsoftmax(b)
kl = (np.exp(la)*(la-lb)).sum(axis=1)
result = {
    "prompt_tokens": 4096, "decode_steps": 8, "vocab": a.shape[1],
    "kl_baseline_to_candidate": kl.tolist(),
    "max_abs_logit_error": np.abs(a-b).max(axis=1).tolist(),
    "rms_logit_error": np.sqrt(np.mean((a-b)**2, axis=1)).tolist(),
    "top1_baseline": a.argmax(axis=1).tolist(), "top1_candidate": b.argmax(axis=1).tolist(),
    "top1_matches": int(np.sum(a.argmax(axis=1) == b.argmax(axis=1))),
    "all_finite": True,
}
text = json.dumps(result, indent=2)+"\n"
print(text, end="")
if args.output:
    args.output.write_text(text)
