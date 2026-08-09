"""Check E2 retrain status."""
import argparse
import re
import sys
import time

from remote_config import connect_ssh

ROOT = "/home/szw/lhm2"
PIDFILE = ROOT + "/runtime/kisaki_e2pp_rag.pid"
LOGFILE = ROOT + "/runtime/logs/kisaki_e2pp_rag.log"
FINAL_ADAPTER = ROOT + "/runtime/loras/kisaki/e2pp_rag_r32/final/adapter_config.json"

COMPLETION_MARKERS = ("training_complete", "save_final_adapter", "Early stopping", "final_adapter_saved", "Final adapter saved")


def run(cli, cmd, timeout=30):
    _, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err


def get_status(cli):
    status = {"pid": None, "running": False, "adapter_ready": False, "log_tail": [], "progress": None, "completion_marker": None, "started_at": None}
    out, _ = run(cli, "cat " + PIDFILE + " 2>/dev/null")
    if out:
        try:
            status["pid"] = int(out.strip())
        except ValueError:
            status["pid"] = None
    if status["pid"]:
        out, _ = run(cli, "ps -p " + str(status["pid"]) + " -o pid,etime,cmd --no-headers 2>/dev/null")
        if out:
            status["running"] = True
            parts = out.split(None, 2)
            if len(parts) >= 2:
                status["started_at"] = parts[1]
    out, _ = run(cli, "test -f " + FINAL_ADAPTER + " && echo yes || echo no")
    status["adapter_ready"] = out.strip() == "yes"
    out, _ = run(cli, "tail -30 " + LOGFILE + " 2>&1")
    if out:
        status["log_tail"] = out.splitlines()
    for line in status["log_tail"]:
        for marker in COMPLETION_MARKERS:
            if marker in line:
                status["completion_marker"] = marker
                break
        if status["completion_marker"]:
            break
    for line in reversed(status["log_tail"]):
        m = re.search(r"(\d+)/(\d+)\s*\[", line)
        if m:
            status["progress"] = {"step": int(m.group(1)), "total": int(m.group(2)), "pct": round(int(m.group(1)) / int(m.group(2)) * 100, 1)}
            break
    return status


def print_status(status):
    print("=" * 60)
    print("E2 retrain status")
    print("=" * 60)
    print("PID:", status["pid"])
    print("Running:", "yes" if status["running"] else "no")
    if status["started_at"]:
        print("Elapsed:", status["started_at"])
    print("Adapter ready:", "yes" if status["adapter_ready"] else "no")
    if status["progress"]:
        p = status["progress"]
        print("Progress: " + str(p["step"]) + "/" + str(p["total"]) + " (" + str(p["pct"]) + "%)")
    if status["completion_marker"]:
        print("Completion marker:", status["completion_marker"])
    print("")
    print("Log tail:")
    for line in status["log_tail"][-10:]:
        print("  " + line)
    is_complete = (not status["running"]) and status["adapter_ready"]
    print("")
    print("=" * 60)
    if is_complete:
        print("[OK] Training complete, adapter ready for evaluation")
    elif not status["running"]:
        print("[WARN] Training process exited but adapter not ready, check log")
    else:
        print("[WAIT] Training in progress")
    print("=" * 60)
    return is_complete


def main():
    parser = argparse.ArgumentParser(description="Check E2 retrain status")
    parser.add_argument("--wait", action="store_true", help="Wait for training to complete")
    parser.add_argument("--poll-interval", type=int, default=60, help="Poll interval (seconds)")
    parser.add_argument("--max-wait", type=int, default=3600, help="Max wait time (seconds)")
    args = parser.parse_args()

    cli = connect_ssh()

    if not args.wait:
        status = get_status(cli)
        is_complete = print_status(status)
        cli.close()
        sys.exit(0 if is_complete else 1)

    print("Waiting for training to complete, poll interval " + str(args.poll_interval) + "s, max wait " + str(args.max_wait) + "s...")
    elapsed = 0
    while elapsed < args.max_wait:
        status = get_status(cli)
        is_complete = (not status["running"]) and status["adapter_ready"]
        if status["progress"]:
            p = status["progress"]
            print("[" + str(elapsed) + "s] step " + str(p["step"]) + "/" + str(p["total"]) + " (" + str(p["pct"]) + "%) running=" + str(status["running"]) + " adapter_ready=" + str(status["adapter_ready"]))
        else:
            print("[" + str(elapsed) + "s] running=" + str(status["running"]) + " adapter_ready=" + str(status["adapter_ready"]))
        if is_complete:
            print("")
            print("[OK] Training complete!")
            print_status(status)
            cli.close()
            sys.exit(0)
        if not status["running"]:
            print("")
            print("[WARN] Training process exited but adapter not ready, check log:")
            for line in status["log_tail"][-15:]:
                print("  " + line)
            cli.close()
            sys.exit(2)
        time.sleep(args.poll_interval)
        elapsed += args.poll_interval
    print("")
    print("[FAIL] Wait timeout (" + str(args.max_wait) + "s)")
    print_status(status)
    cli.close()
    sys.exit(3)


if __name__ == "__main__":
    main()
