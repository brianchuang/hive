"""
Durable wait / signal / timer runtime substrate.

Provides first-class primitives for:
- Wait: durable suspension without holding compute
- Signals: external events delivered to a run with deterministic matching
- Timers: time-based wake-ups (synthetic timeout signals)

Lifecycle: a node calls wait() and gets ExecutionPaused; the runner persists that
and stops. Later, signal() (external event) or tick() (timeout) matches one pending
wait; the runner resumes that run with the wait result (signal payload or timeout).

Guarantees:
- Run isolation: waits and signals are scoped per run_id.
- Exactly-once resume: each wait is resumed at most once (FIFO matching; "atomic
  claim" means the store marks the wait resumed when it is chosen).
- Deterministic matching: exact-match key/value selectors only; no arbitrary predicates.
- Signal idempotency: duplicate delivery of the same signal_id does not double-resume
  (store deduplicates by signal_id).

Durable data invariants (for replay and schema evolution):
- Immutability: durable fields are deep-immutable (primitives, tuples, frozen mappings).
- Versioning: every persisted object has schema_version for upgrade-on-read when
  loading old payloads.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from framework.runtime.event_bus import EventBus

logger = logging.getLogger(__name__)

# Synthetic signal type emitted when a wait times out. Timeout is surfaced as a signal
# so node logic can branch on matched_signal_type only (e.g. "email.reply" vs this),
WAIT_TIMEOUT_SIGNAL_TYPE = "wait.timeout"

# --- Durable value types (deep immutability for replay and schema evolution) ---

Primitive = str | int | float | bool
"""Only primitive values allowed in durable selectors/payloads; no nested mutables."""

ImmutableSelectors = tuple[tuple[str, Primitive], ...]
"""
Exact-match key/value selectors for deterministic, indexable matching.
Canonical representation: tuple of (key, value) pairs. Use selectors_from_dict()
to build from a dict; use selectors_to_dict() to read back.
"""


def selectors_from_dict(d: dict[str, Primitive] | None) -> ImmutableSelectors | None:
    """
    Build immutable selectors from a dict. None -> None; {} -> ().
    Sorted by key for deterministic order and equality.
    """
    if d is None:
        return None
    if not d:
        return ()
    return tuple(sorted(d.items()))


def selectors_to_dict(s: ImmutableSelectors | None) -> dict[str, Primitive]:
    """Convert immutable selectors back to a dict for matching/reading."""
    if s is None or len(s) == 0:
        return {}
    return dict(s)


# --- Versioned, immutable durable contracts ---


@dataclass(frozen=True)
class WaitRequest:
    """
    Durable suspension point created by a specific node attempt.

    Unique within a run. When persisted, execution suspends until a matching
    signal is delivered or timeout_at is reached.

    Durable invariants: schema_version for upgrade-on-read; match is
    immutable and deterministic (exact-match selectors only).
    """

    wait_id: str
    run_id: str
    node_id: str
    attempt: int
    signal_type: str  # e.g. "email.reply", "approval"
    match: ImmutableSelectors | None  # optional exact-match selectors
    timeout_at: datetime | None
    schema_version: int = 1
    type: str = "wait_request"


@dataclass(frozen=True)
class SignalEnvelope:
    """
    Externally delivered event.

    Delivered at least once; runtime guarantees exactly-once resume per wait,
    not exactly-once delivery. signal_id is required for idempotency: duplicate
    delivery of the same signal_id does not cause duplicate resume.
    """

    signal_type: str
    payload: ImmutableSelectors  # immutable key-value pairs for replay safety
    signal_id: str  # stable id for deduplication; store enforces uniqueness
    correlation_id: str | None
    causation_id: str | None
    received_at: datetime
    schema_version: int = 1
    type: str = "signal_envelope"

    def payload_as_dict(self) -> dict[str, Primitive]:
        """Return payload as dict for node consumption (e.g. _resume_signal_payload)."""
        return selectors_to_dict(self.payload)


@dataclass
class ExecutionPaused:
    """
    Result of runtime.wait(): execution suspended on a durable wait.

    Caller should persist this and resume when signal or tick delivers
    a matching resume for this wait_id/run_id.
    """

    wait_id: str
    run_id: str
    node_id: str
    attempt: int
    session_state: dict[str, Any]
    wait_request: WaitRequest = field(repr=False)


@dataclass
class WaitResumed:
    """
    Result of signal() or tick(): one wait was resumed (matched or timed out).

    Used by the runner to know which run/wait to resume and whether
    it was due to signal (timed_out=False) or timeout (timed_out=True).
    """

    run_id: str
    wait_id: str
    timed_out: bool
    matched_signal_type: str | None = None  # set when resumed by signal


def _match_filter(
    wait_match: ImmutableSelectors | None,
    payload: ImmutableSelectors,
) -> bool:
    """True if payload satisfies wait_match (all keys in wait_match equal in payload)."""
    if wait_match is None or len(wait_match) == 0:
        return True
    payload_dict = selectors_to_dict(payload)
    for k, v in wait_match:
        if payload_dict.get(k) != v:
            return False
    return True


class WaitStoreIfce(ABC):
    """
    Interface for persisting and querying pending waits.

    Implementations must guarantee run isolation and exactly-once
    resume per (run_id, wait_id). Waits have explicit lifecycle state
    (active -> resumed); match_signal and mark_resumed perform atomic claim.
    Duplicate delivery of the same signal_id must not cause duplicate resume.
    """

    @abstractmethod
    async def add(self, wait: WaitRequest) -> None:
        """Persist a wait in active state. Idempotent for same wait_id within run."""
        ...

    @abstractmethod
    async def get_pending(self, run_id: str) -> list[WaitRequest]:
        """Return pending (active) waits for the run, in creation order (FIFO)."""
        ...

    @abstractmethod
    async def match_signal(self, run_id: str, envelope: SignalEnvelope) -> str | None:
        """
        Find one active wait matching the signal (type + optional match filter).

        Deterministic: FIFO by creation order. Atomically claims the wait
        (exactly-once resume). If envelope.signal_id was already applied for
        this run, returns None (idempotency). Returns wait_id or None.
        """
        ...

    @abstractmethod
    async def mark_resumed(self, run_id: str, wait_id: str) -> None:
        """Mark an active wait as resumed (e.g. after timeout). Atomic claim."""
        ...

    @abstractmethod
    async def get_expired(self, now: datetime) -> list[tuple[str, str]]:
        """
        Return (run_id, wait_id) pairs for active waits with timeout_at <= now.

        Implementations may either only enumerate expired waits (caller calls
        mark_resumed) or enumerate and mark them resumed; see implementation.
        """
        ...


WaitStatus = Literal["active", "resumed"]


@dataclass
class _StoredWait:
    """Internal: wait plus explicit lifecycle state for exactly-once resume."""

    wait: WaitRequest
    status: WaitStatus


class InMemoryWaitStore(WaitStoreIfce):
    """
    In-memory wait store: pending waits per run, FIFO order.

    Run-isolated; exactly-once resume via explicit status (active -> resumed).
    Deduplicates by signal_id so duplicate delivery does not double-resume.
    """

    def __init__(self) -> None:
        # run_id -> list of _StoredWait (creation order; status active or resumed)
        self._pending: dict[str, list[_StoredWait]] = {}
        # run_id -> set of signal_id already applied (idempotency)
        self._processed_signal_ids: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    async def add(self, wait: WaitRequest) -> None:
        async with self._lock:
            if wait.run_id not in self._pending:
                self._pending[wait.run_id] = []
            existing_ids = {
                s.wait.wait_id for s in self._pending[wait.run_id] if s.status == "active"
            }
            if wait.wait_id not in existing_ids:
                self._pending[wait.run_id].append(_StoredWait(wait=wait, status="active"))

    async def get_pending(self, run_id: str) -> list[WaitRequest]:
        async with self._lock:
            return [s.wait for s in self._pending.get(run_id, []) if s.status == "active"]

    async def match_signal(self, run_id: str, envelope: SignalEnvelope) -> str | None:
        async with self._lock:
            # Idempotency: same signal_id applied twice -> no second resume
            processed = self._processed_signal_ids.setdefault(run_id, set())
            if envelope.signal_id in processed:
                return None
            pending = self._pending.get(run_id, [])
            for i, s in enumerate(pending):
                if s.status != "active":
                    continue
                w = s.wait
                if w.signal_type != envelope.signal_type:
                    continue
                if not _match_filter(w.match, envelope.payload):
                    continue
                # Atomic claim: mark resumed and record signal_id
                pending[i] = _StoredWait(wait=w, status="resumed")
                processed.add(envelope.signal_id)
                return w.wait_id
        return None

    async def mark_resumed(self, run_id: str, wait_id: str) -> None:
        async with self._lock:
            pending = self._pending.get(run_id, [])
            for i, s in enumerate(pending):
                if s.status == "active" and s.wait.wait_id == wait_id:
                    pending[i] = _StoredWait(wait=s.wait, status="resumed")
                    break

    async def get_expired(self, now: datetime) -> list[tuple[str, str]]:
        async with self._lock:
            result: list[tuple[str, str]] = []
            for rid, pending in list(self._pending.items()):
                for i, s in enumerate(pending):
                    if s.status != "active":
                        continue
                    w = s.wait
                    if w.timeout_at is not None and w.timeout_at <= now:
                        # Mark resumed here so we only hand out each wait once
                        pending[i] = _StoredWait(wait=w, status="resumed")
                        result.append((rid, w.wait_id))
            return result


class DurableWaitRuntime:
    """
    Runtime surface for durable wait / signal / tick.

    Additive API: wait() suspends and returns ExecutionPaused;
    signal() and tick() drive deterministic resume. Emits lifecycle
    events (wait.created, wait.matched, wait.timed_out) for auditing.
    """

    def __init__(
        self,
        wait_store: WaitStoreIfce,
        event_bus: EventBus | None = None,
        stream_id: str = "default",
    ) -> None:
        self._store = wait_store
        self._event_bus = event_bus
        self._stream_id = stream_id

    async def wait(
        self,
        wait_request: WaitRequest,
        session_state: dict[str, Any] | None = None,
    ) -> ExecutionPaused:
        """
        Persist the wait and suspend execution.

        Returns ExecutionPaused so the caller can persist and later
        resume when signal or tick fires for this wait.
        """
        await self._store.add(wait_request)
        state = session_state or {}

        if self._event_bus:
            from framework.runtime.event_bus import AgentEvent, EventType

            await self._event_bus.publish(
                AgentEvent(
                    type=EventType.WAIT_CREATED,
                    stream_id=self._stream_id,
                    execution_id=wait_request.run_id,
                    data={
                        "wait_id": wait_request.wait_id,
                        "run_id": wait_request.run_id,
                        "node_id": wait_request.node_id,
                        "attempt": wait_request.attempt,
                        "signal_type": wait_request.signal_type,
                        "timeout_at": (
                            wait_request.timeout_at.isoformat() if wait_request.timeout_at else None
                        ),
                    },
                )
            )

        return ExecutionPaused(
            wait_id=wait_request.wait_id,
            run_id=wait_request.run_id,
            node_id=wait_request.node_id,
            attempt=wait_request.attempt,
            session_state=state,
            wait_request=wait_request,
        )

    async def signal(
        self,
        run_id: str,
        envelope: SignalEnvelope,
    ) -> WaitResumed | None:
        """
        Match the signal against pending waits for the run; resume at most one.

        Deterministic (FIFO). Returns WaitResumed for the matched wait, or None.
        """
        from framework.runtime.event_bus import AgentEvent, EventType

        wait_id = await self._store.match_signal(run_id, envelope)
        if wait_id is None:
            return None

        if self._event_bus:
            await self._event_bus.publish(
                AgentEvent(
                    type=EventType.WAIT_MATCHED,
                    stream_id=self._stream_id,
                    execution_id=run_id,
                    data={
                        "wait_id": wait_id,
                        "run_id": run_id,
                        "signal_type": envelope.signal_type,
                    },
                )
            )
            await self._event_bus.publish(
                AgentEvent(
                    type=EventType.WAIT_RESUMED,
                    stream_id=self._stream_id,
                    execution_id=run_id,
                    data={
                        "wait_id": wait_id,
                        "run_id": run_id,
                        "timed_out": False,
                        "matched_signal_type": envelope.signal_type,
                    },
                )
            )

        return WaitResumed(
            run_id=run_id,
            wait_id=wait_id,
            timed_out=False,
            matched_signal_type=envelope.signal_type,
        )

    async def tick(self, now: datetime | None = None) -> list[WaitResumed]:
        """
        Emit timeout events for expired waits and return resumed list.

        Each expired wait is marked resumed and yields a WaitResumed
        with timed_out=True (synthetic wait.timeout signal semantics).
        """
        from framework.runtime.event_bus import AgentEvent, EventType

        if now is None:
            now = datetime.now(UTC)
        expired = await self._store.get_expired(now)
        resumed: list[WaitResumed] = []
        for run_id, wait_id in expired:
            await self._store.mark_resumed(run_id, wait_id)
            if self._event_bus:
                await self._event_bus.publish(
                    AgentEvent(
                        type=EventType.WAIT_TIMED_OUT,
                        stream_id=self._stream_id,
                        execution_id=run_id,
                        data={
                            "wait_id": wait_id,
                            "run_id": run_id,
                            "signal_type": WAIT_TIMEOUT_SIGNAL_TYPE,
                        },
                    )
                )
                await self._event_bus.publish(
                    AgentEvent(
                        type=EventType.WAIT_RESUMED,
                        stream_id=self._stream_id,
                        execution_id=run_id,
                        data={
                            "wait_id": wait_id,
                            "run_id": run_id,
                            "timed_out": True,
                            "matched_signal_type": WAIT_TIMEOUT_SIGNAL_TYPE,
                        },
                    )
                )
            resumed.append(
                WaitResumed(
                    run_id=run_id,
                    wait_id=wait_id,
                    timed_out=True,
                    matched_signal_type=WAIT_TIMEOUT_SIGNAL_TYPE,
                )
            )
        return resumed
