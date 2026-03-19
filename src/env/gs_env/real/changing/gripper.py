"""
DDS client for ChangingTek grippers: subscribe to state, publish commands.

Runs a CycloneDDS participant; exposes state as properties and APIs to control
closure (0=open, 1=closed) for both grippers. Use with gripper_dds_server running.
"""

from __future__ import annotations

import os

from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

from .gripper_dds_server import (
    TOPIC_COMMAND,
    TOPIC_STATE,
    GripperClosureState,
    GripperCommand,
)

_dir = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CYCLONEDDS_URI", f"file://{_dir}/cyclonedds.xml")


class Gripper:
    """
    DDS participant for gripper state (read) and commands (write).

    State is read from the server; call update_state() to refresh before
    reading closure1/closure2. Commands are published immediately.
    """

    def __init__(self, domain_id: int = 0) -> None:
        self._participant = DomainParticipant(domain_id=domain_id)
        topic_state = Topic(self._participant, TOPIC_STATE, GripperClosureState)
        topic_command = Topic(self._participant, TOPIC_COMMAND, GripperCommand)
        self._reader = DataReader(self._participant, topic_state)
        self._writer = DataWriter(self._participant, topic_command)
        self._closure1 = 0.0
        self._closure2 = 0.0
        self._has_state = False

    # --- State (subscribe): refresh with update_state(), then read properties ---

    @property
    def closure1(self) -> float:
        """Closure of gripper 1 (0=open, 1=closed). Updated by update_state()."""
        return self._closure1

    @property
    def closure2(self) -> float:
        """Closure of gripper 2 (0=open, 1=closed). Updated by update_state()."""
        return self._closure2

    @property
    def has_state(self) -> bool:
        """True if at least one state sample has been received."""
        return self._has_state

    def update_state(self, timeout_ms: int = 10) -> None:
        """Read latest state from DDS and update closure1/closure2."""
        samples = list(self._reader.take_iter(timeout=duration(milliseconds=timeout_ms)))
        if samples:
            latest = samples[-1]
            self._closure1 = float(latest.closure1)
            self._closure2 = float(latest.closure2)
            self._has_state = True

    # --- Control (publish) ---

    def set_closure(self, closure1: float, closure2: float) -> None:
        """Send target closure for both grippers (0=open, 1=closed)."""
        c1 = max(0.0, min(1.0, closure1))
        c2 = max(0.0, min(1.0, closure2))
        self._writer.write(GripperCommand(closure1=c1, closure2=c2))

    def open_both(self) -> None:
        """Command both grippers to open (closure 0)."""
        self.set_closure(0.0, 0.0)

    def close_both(self) -> None:
        """Command both grippers to close (closure 1)."""
        self.set_closure(1.0, 1.0)

    def open_gripper1(self) -> None:
        """Command gripper 1 to open; gripper 2 unchanged (uses current closure2)."""
        self.set_closure(0.0, self._closure2)

    def close_gripper1(self) -> None:
        """Command gripper 1 to close; gripper 2 unchanged."""
        self.set_closure(1.0, self._closure2)

    def open_gripper2(self) -> None:
        """Command gripper 2 to open; gripper 1 unchanged."""
        self.set_closure(self._closure1, 0.0)

    def close_gripper2(self) -> None:
        """Command gripper 2 to close; gripper 1 unchanged."""
        self.set_closure(self._closure1, 1.0)
