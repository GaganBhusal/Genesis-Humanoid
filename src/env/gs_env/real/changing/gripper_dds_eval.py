"""
Test script for CycloneDDS gripper topics.

Uses Gripper to subscribe to state (printed once a second) and publish
toggling commands once a second.

Run (with gripper_dds_server running or standalone to test DDS):
    python -m gs_env.real.changing.gripper_dds_test
"""

from __future__ import annotations

import argparse
import signal
import time

from gs_env.real.changing.gripper import Gripper

COMMAND_FLIP_INTERVAL = 1.0  # print state and flip command once a second


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test CycloneDDS gripper topics: print state 1 Hz, flip command 1 Hz."
    )
    parser.add_argument(
        "--domain",
        type=int,
        default=0,
        help="DDS domain id (default: 0)",
    )
    args = parser.parse_args()

    gripper = Gripper(domain_id=args.domain)
    next_cmd = time.time() + COMMAND_FLIP_INTERVAL
    toggle = False
    stop = False

    def on_signal(*_args: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print(f"Printing state and flipping command every {COMMAND_FLIP_INTERVAL}s (Ctrl+C to stop)")
    while not stop:
        now = time.time()

        if now >= next_cmd:
            gripper.update_state()
            if gripper.has_state:
                print(f"[state] closure1={gripper.closure1:.3f}  closure2={gripper.closure2:.3f}")
            else:
                print("[state] (no data)")
            c1, c2 = (1.0, 0.0) if toggle else (0.0, 1.0)
            gripper.set_closure(c1, c2)
            print(f"[command] sent closure1={c1:.1f} closure2={c2:.1f}")
            toggle = not toggle
            next_cmd = now + COMMAND_FLIP_INTERVAL

        time.sleep(0.05)

    print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
