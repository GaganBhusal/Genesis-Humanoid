"""
CycloneDDS server for ChangingTek grippers.

Publishes the closure state of both grippers and subscribes to commands.
Uses CycloneDDS for discovery and transport.

Run:
    python -m gs_env.real.changing.gripper_dds_server

Requires: pip install cyclonedds
"""

from __future__ import annotations

import argparse
import os
import signal
import threading
import time
from dataclasses import dataclass

from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import float64
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

try:
    from .utils.changingtek_p_rtu_Servo import MotorController
except ImportError:
    from utils.changingtek_p_rtu_Servo import MotorController  # type: ignore

_dir = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CYCLONEDDS_URI", f"file://{_dir}/cyclonedds.xml")


# ---------------------------------------------------------------------------
# ChangingTek gripper state and client
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GripperState:
    position: int
    speed: int
    current: int


class GripperClient:
    """Client for controlling one ChangingTek gripper on RS-485."""

    _bus_lock = threading.Lock()

    def __init__(
        self,
        port: str,
        slave_id: int = 1,
        baudrate: int = 115200,
        timeout: float = 1.0,
        init_speed: int = 50,
        init_force: int = 25,
        init_acceleration: int = 60,
        init_deceleration: int = 60,
    ) -> None:
        self.slave_id = slave_id
        self._controller = MotorController(
            port=port, slave_id=slave_id, baudrate=baudrate, timeout=timeout
        )
        self._initialize_motion_params(
            speed=init_speed,
            force=init_force,
            acceleration=init_acceleration,
            deceleration=init_deceleration,
        )

    def move(
        self,
        position: float,
        trigger: bool = True,
    ) -> None:
        position = (1 - max(0, min(position, 1.0))) * 9000
        with self._bus_lock:
            self._controller.set_target_position(int(position))
            if trigger:
                self._controller.trigger_motion()

    def open(self) -> None:
        self.move(position=1.0)

    def close(self) -> None:
        self.move(position=0.0)

    def trigger(self) -> None:
        with self._bus_lock:
            self._controller.trigger_motion()

    def read_state(self) -> GripperState:
        with self._bus_lock:
            return GripperState(
                position=self._controller.read_real_position(),
                speed=self._controller.read_real_speed(),
                current=self._controller.read_real_current(),
            )

    def _initialize_motion_params(
        self,
        speed: int,
        force: int,
        acceleration: int,
        deceleration: int,
    ) -> None:
        with self._bus_lock:
            self._controller.set_target_speed(speed)
            self._controller.set_target_force(force)
            self._controller.set_target_acceleration(acceleration)
            self._controller.set_target_deceleration(deceleration)
            self._controller.set_target_position(0)
            self._controller.trigger_motion()


# ---------------------------------------------------------------------------
# DDS message types (IdlStruct for CycloneDDS)
# ---------------------------------------------------------------------------

# Closure: 0.0 = open, 1.0 = closed (same as GripperClient position 1.0 = open, 0.0 = closed)


@dataclass
class GripperClosureState(IdlStruct):
    """Published state: closure of both grippers (0=open, 1=closed)."""

    closure1: float64
    closure2: float64


@dataclass
class GripperCommand(IdlStruct):
    """Subscribed command: target closure for both grippers (0=open, 1=closed)."""

    closure1: float64
    closure2: float64


# ---------------------------------------------------------------------------
# Topic names
# ---------------------------------------------------------------------------

TOPIC_STATE = "changing/grippers/state"
TOPIC_COMMAND = "changing/grippers/command"

# Raw position range on hardware
POSITION_RANGE = 9000


def _state_to_closure(state: GripperState) -> float:
    """Convert gripper position (0..9000) to closure 0..1 (0=open, 1=closed)."""
    return max(0.0, min(1.0, state.position / POSITION_RANGE))


def _closure_to_move_position(closure: float) -> float:
    """Convert closure (0=open, 1=closed) to GripperClient move position (1=open, 0=closed)."""
    return 1.0 - max(0.0, min(1.0, closure))


class GripperDDSServer:
    """Maintains a CycloneDDS server: publish gripper closure state, subscribe to commands."""

    def __init__(
        self,
        client1: GripperClient,
        client2: GripperClient,
        domain_id: int = 0,
        state_hz: float = 25.0,
    ) -> None:
        self.client1 = client1
        self.client2 = client2
        self.domain_id = domain_id
        self.state_hz = state_hz
        self._stop = threading.Event()
        self._state_thread: threading.Thread | None = None
        self._command_thread: threading.Thread | None = None

        self._participant = DomainParticipant(domain_id=domain_id)
        self._topic_state = Topic(self._participant, TOPIC_STATE, GripperClosureState)
        self._topic_command = Topic(self._participant, TOPIC_COMMAND, GripperCommand)
        self._writer = DataWriter(self._participant, self._topic_state)
        self._reader = DataReader(self._participant, self._topic_command)

    def start(self) -> None:
        """Start state publisher and command subscriber threads."""
        self._stop.clear()
        self._state_thread = threading.Thread(
            target=self._publish_loop,
            daemon=True,
            name="gripper-dds-state",
        )
        self._command_thread = threading.Thread(
            target=self._subscribe_loop,
            daemon=True,
            name="gripper-dds-command",
        )
        self._state_thread.start()
        self._command_thread.start()
        print(
            f"[GripperDDSServer] Started: publishing '{TOPIC_STATE}' at {self.state_hz} Hz, "
            f"subscribing to '{TOPIC_COMMAND}'"
        )

    def stop(self) -> None:
        """Stop publisher and subscriber threads."""
        self._stop.set()
        if self._state_thread is not None:
            self._state_thread.join(timeout=2.0)
        if self._command_thread is not None:
            self._command_thread.join(timeout=2.0)
        print("[GripperDDSServer] Stopped.")

    def _publish_loop(self) -> None:
        interval = 1.0 / self.state_hz
        next_tick = time.time() + interval
        while not self._stop.is_set():
            try:
                s1 = self.client1.read_state()
                s2 = self.client2.read_state()
                msg = GripperClosureState(
                    closure1=_state_to_closure(s1),
                    closure2=_state_to_closure(s2),
                )
                self._writer.write(msg)
            except Exception as e:
                print(f"[GripperDDSServer] Publish error: {e}")
            now = time.time()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
                next_tick += interval
            else:
                next_tick = now + interval

    def _subscribe_loop(self) -> None:
        while not self._stop.is_set():
            try:
                for msg in self._reader.take_iter(timeout=duration(milliseconds=100)):
                    c1 = max(0.0, min(1.0, msg.closure1))
                    c2 = max(0.0, min(1.0, msg.closure2))
                    pos1 = _closure_to_move_position(c1)
                    pos2 = _closure_to_move_position(c2)
                    # Set both targets first, then trigger both so they start simultaneously
                    self.client1.move(position=pos1, trigger=False)
                    self.client2.move(position=pos2, trigger=False)
                    self.client1.trigger()
                    self.client2.trigger()
            except Exception as e:
                print(f"[GripperDDSServer] Command error: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CycloneDDS server for ChangingTek grippers (publish state, subscribe commands)."
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port for grippers (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--domain",
        type=int,
        default=0,
        help="DDS domain id (default: 0)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=25.0,
        help="State publish rate in Hz (default: 25)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Serial baudrate (default: 115200)",
    )
    args = parser.parse_args()

    max_attempts = 100
    retry_delay = 1.0
    server: GripperDDSServer | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            client1 = GripperClient(port=args.port, slave_id=1, baudrate=args.baudrate)
            client2 = GripperClient(port=args.port, slave_id=2, baudrate=args.baudrate)

            # Set initial closure to 0.9 for both grippers
            pos1 = _closure_to_move_position(0.9)
            pos2 = _closure_to_move_position(0.9)
            client1.move(position=pos1, trigger=False)
            client2.move(position=pos2, trigger=False)
            client1.trigger()
            client2.trigger()
            print("[GripperDDSServer] Set initial closure to 0.9 for both grippers")

            server = GripperDDSServer(
                client1=client1,
                client2=client2,
                domain_id=args.domain,
                state_hz=args.hz,
            )
            server.start()
            break
        except Exception as e:
            print(f"[GripperDDSServer] Connection attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(retry_delay)
            else:
                print(f"[GripperDDSServer] Giving up after {max_attempts} attempts.")
                return 1

    assert server is not None

    shutdown = threading.Event()

    def on_signal(*_args: object) -> None:
        server.stop()
        shutdown.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        while not shutdown.is_set():
            shutdown.wait(timeout=1.0)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
