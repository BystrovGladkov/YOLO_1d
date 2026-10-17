"""E1: смоук-прогоны реального Trainer на полном ECRv1 (каталог experiments/ec_ablation/e1/runs).

    python experiments/ec_ablation/e1_smoke.py A0 --epochs 3 --workers 15
Время каждой эпохи — по моменту появления строки в step.csv (поток-наблюдатель), пишется в epoch_times.json.
"""
import argparse, json, sys, threading, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
from run_ec_ablation import EXP_DIR, load_config, run_name, run_one

def watch(step_csv: Path, out: Path, stop: threading.Event):
    t0, seen, times = time.time(), 0, []
    while not stop.is_set():
        if step_csv.is_file():
            n = max(sum(1 for _ in step_csv.open()) - 1, 0)
            while seen < n:
                seen += 1; times.append(round(time.time() - t0, 1))
                out.write_text(json.dumps({"epoch_end_s": times}), encoding="utf-8")
        time.sleep(2)

def main():
    p = argparse.ArgumentParser(); p.add_argument("config"); p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--workers", type=int, default=15); p.add_argument("--seed", type=int, default=0); a = p.parse_args()
    config = load_config(a.config, epochs=a.epochs)
    config["training"]["num_workers"] = a.workers
    runs = EXP_DIR / "e1" / "runs"
    run_dir = runs / run_name(config, a.seed)
    stop = threading.Event()
    threading.Thread(target=watch, args=(run_dir / "train1" / "step.csv", run_dir / "epoch_times.json", stop), daemon=True).start()
    for s in (sys.stdout, sys.stderr): s.reconfigure(encoding="utf-8")
    run_one(config, a.seed, runs, rerun=True)
    stop.set()


if __name__ == "__main__":
    main()
