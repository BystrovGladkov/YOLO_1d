import sys, time, glob, numpy as np, torch
sys.path.insert(0, "src")
from yolo_1d.utils.transforms.square_noises import new_smooth_noise, ICA_filtration, SquareNoises
np.random.seed(0)
fr, nseg, amp, durs = [], [], [], []
for _ in range(300):
    x, z, xs = new_smooth_noise(49152)
    fr.append(z.mean())
    d = np.diff(np.concatenate([[0], z, [0]]))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    nseg.append(len(s)); durs += list(e - s)
    if z.any(): amp.append(np.abs(xs[z > 0]).mean())
fr = np.array(fr)
print("noise fraction: mean %.3f median %.3f P(no noise) %.3f P(>50%%) %.3f" % (fr.mean(), np.median(fr), (fr == 0).mean(), (fr > .5).mean()))
print("segments/signal mean %.2f; seg duration median %d, p10 %d, p90 %d" % (np.mean(nseg), np.median(durs), np.percentile(durs, 10), np.percentile(durs, 90)))
print("mean |amp| inside noise %.2f" % np.mean(amp))
f = sorted(glob.glob("H:/dataset/data_1d/dataset/data/train/*.pt"))[0]
d = torch.load(f, weights_only=False).float()
print(f, tuple(d.shape), "raw ch std median %.2f, |x| p99 %.2f" % (d[:15].std(1).median(), d[:15].abs().quantile(.99)))
d = d[:, :49152]
t = time.time(); SquareNoises()(d, None, None); print("SquareNoises time %.1f s" % (time.time() - t))
t = time.time(); SquareNoises()(d, None, None); print("SquareNoises time (2nd) %.1f s" % (time.time() - t))
