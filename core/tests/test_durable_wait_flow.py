"""
Integration test: Send email → wait for reply OR timeout → branch deterministically.

Uses durable wait primitives (WaitRequest, signal, tick) and the executor's
paused/session_state path to verify the full flow.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from framework.graph.edge import EdgeCondition, EdgeSpec, GraphSpec
from framework.graph.executor import GraphExecutor
from framework.graph.goal import Goal
from framework.graph.node import (
    FunctionNode,
    NodeContext,
    NodeProtocol,
    NodeResult,
    NodeSpec,
)
from framework.runtime.durable_wait import (
    WAIT_TIMEOUT_SIGNAL_TYPE,
    DurableWaitRuntime,
    InMemoryWaitStore,
    SignalEnvelope,
    WaitRequest,
    selectors_from_dict,
)


# ---- Runtime that exposes current_run.id for durable wait ----
class RuntimeWithRunId:
    def __init__(self) -> None:
        self._current_run: object | None = None

    @property
    def current_run(self) -> object | None:
        return self._current_run

    def start_run(self, **kwargs: object) -> str:
        run_id = f"run_{uuid4().hex[:8]}"
        self._current_run = type("Run", (), {"id": run_id})()
        return run_id

    def end_run(self, **kwargs: object) -> None:
        self._current_run = None

    def report_problem(self, **kwargs: object) -> None:
        pass

    def set_node(self, node_id: str) -> None:
        pass

    def decide(self, **kwargs: object) -> str:
        return "dec-1"

    def record_outcome(self, **kwargs: object) -> None:
        pass


# ---- Send email then wait node (yields WaitRequest, returns paused) ----
class SendEmailThenWaitNode(NodeProtocol):
    """Simulates send email then durable wait for reply or timeout."""

    def validate_input(self, ctx: NodeContext) -> list[str]:
        return []

    async def execute(self, ctx: NodeContext) -> NodeResult:
        if ctx.durable_wait_runtime is None or ctx.run_id is None:
            return NodeResult(
                success=False,
                error="durable_wait_runtime and run_id required for wait",
            )
        # "Send email" (mock)
        ctx.memory.write("email_sent", True)
        thread_id = "thread-1"
        timeout_at = datetime.now(UTC) + timedelta(seconds=60)
        wait_req = WaitRequest(
            wait_id=f"{ctx.node_id}-{ctx.attempt}-{uuid4().hex[:8]}",
            run_id=ctx.run_id,
            node_id=ctx.node_id,
            attempt=ctx.attempt,
            signal_type="email.reply",
            match=selectors_from_dict({"thread_id": thread_id}),
            timeout_at=timeout_at,
        )
        session_state = {
            "paused_at": ctx.node_id,
            "resume_from": "branch",
        }
        paused = await ctx.durable_wait_runtime.wait(wait_req, session_state=session_state)
        return NodeResult(
            success=True,
            output={"email_sent": True, "thread_id": thread_id},
            paused=paused,
        )


def branch_on_resume(
    _resume_timed_out: bool = False,
    _resume_signal_payload: dict | None = None,
) -> str:
    """Branch node: outcome 'replied' if signal, 'timeout' if timeout."""
    if _resume_timed_out:
        return "timeout"
    return "replied"


@pytest.mark.asyncio
async def test_send_email_wait_reply_or_timeout_branch_deterministically() -> None:
    # Graph: send_email -> branch
    graph = GraphSpec(
        id="email-flow",
        goal_id="g1",
        nodes=[
            NodeSpec(
                id="send_email",
                name="Send email",
                description="Send email and wait for reply or timeout",
                node_type="function",
                input_keys=[],
                output_keys=["email_sent", "thread_id"],
                max_retries=0,
            ),
            NodeSpec(
                id="branch",
                name="Branch on reply or timeout",
                description="Branch deterministically on resume",
                node_type="function",
                input_keys=["_resume_timed_out", "_resume_signal_payload"],
                output_keys=["outcome"],
                max_retries=0,
            ),
        ],
        edges=[
            EdgeSpec(
                id="send-to-branch",
                source="send_email",
                target="branch",
                condition=EdgeCondition.ON_SUCCESS,
                input_mapping={
                    "_resume_timed_out": "email_sent",
                    "_resume_signal_payload": "thread_id",
                },
            ),
        ],
        entry_node="send_email",
        terminal_nodes=["branch"],
    )

    store = InMemoryWaitStore()
    durable_runtime = DurableWaitRuntime(wait_store=store)
    runtime = RuntimeWithRunId()
    executor = GraphExecutor(
        runtime=runtime,
        node_registry={
            "send_email": SendEmailThenWaitNode(),
            "branch": FunctionNode(branch_on_resume),
        },
    )
    goal = Goal(id="g1", name="email-flow", description="Wait for reply or timeout")

    # --- Run until paused ---
    result = await executor.execute(
        graph=graph,
        goal=goal,
        durable_wait_runtime=durable_runtime,
    )

    assert result.success
    assert result.paused is not None
    assert result.paused_at == "send_email"
    run_id = result.paused.run_id
    session_state = dict(result.session_state)

    # --- Path A: resume via signal (reply) ---
    envelope = SignalEnvelope(
        signal_type="email.reply",
        payload=selectors_from_dict({"thread_id": "thread-1", "body": "I agree"}) or (),
        signal_id="sig_email_reply_1",
        correlation_id="c1",
        causation_id="e1",
        received_at=datetime.now(UTC),
    )
    resumed = await durable_runtime.signal(run_id, envelope)
    assert resumed is not None
    assert resumed.timed_out is False
    assert resumed.matched_signal_type == "email.reply"

    payload_dict = envelope.payload_as_dict()
    result_replied = await executor.execute(
        graph=graph,
        goal=goal,
        session_state={
            **session_state,
            "_resume_timed_out": False,
            "_resume_signal_payload": payload_dict,
        },
        input_data={
            "_resume_timed_out": False,
            "_resume_signal_payload": payload_dict,
        },
    )
    assert result_replied.success
    assert result_replied.paused is None
    assert result_replied.output.get("outcome") == "replied"

    # --- Path B: new run, pause again, then resume via timeout ---
    result2 = await executor.execute(
        graph=graph,
        goal=goal,
        durable_wait_runtime=durable_runtime,
    )
    assert result2.success and result2.paused is not None
    run_id2 = result2.paused.run_id
    session_state2 = dict(result2.session_state)

    # Tick with now past the wait's timeout (wait was created with timeout_at = now+60s)
    future_now = datetime.now(UTC) + timedelta(seconds=61)
    resumed_timeout = await durable_runtime.tick(now=future_now)
    assert len(resumed_timeout) >= 1
    r = next((x for x in resumed_timeout if x.run_id == run_id2), None)
    assert r is not None
    assert r.timed_out is True
    assert r.matched_signal_type == WAIT_TIMEOUT_SIGNAL_TYPE

    result_timeout = await executor.execute(
        graph=graph,
        goal=goal,
        session_state={
            **session_state2,
            "_resume_timed_out": True,
            "_resume_signal_payload": None,
        },
        input_data={
            "_resume_timed_out": True,
            "_resume_signal_payload": None,
        },
    )
    assert result_timeout.success
    assert result_timeout.paused is None
    # FunctionNode writes return value to first output key
    assert result_timeout.output.get("outcome") == "timeout"
