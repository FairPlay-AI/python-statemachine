"""Cancelling delayed events with ``cancel_event()``.

A cancelled trigger stays where it is in its queue, so the triggers around it keep their due
order, and is dropped when it reaches the head. A caller awaiting it is answered with ``None``.

Theme: signal fires, some of them called off before they are lit.
"""

import asyncio

import pytest
from statemachine.event import BoundEvent

from statemachine import State
from statemachine import StateChart

_CLOCKS = ("statemachine.event_data", "statemachine.engines.sync", "statemachine.engines.async_")


class Signals(StateChart):
    idle = State(initial=True)

    signal = idle.to.itself(internal=True, on="answer")

    def __init__(self, *args, **kwargs):
        self.answered = []
        super().__init__(*args, **kwargs)

    def answer(self, due):
        self.answered.append(due)


class AsyncSignals(Signals):
    async def answer(self, due):
        self.answered.append(due)


@pytest.fixture()
def clock(monkeypatch):
    """A monotonic clock that moves only when the test moves it."""
    now = [1000.0]
    for module in _CLOCKS:
        monkeypatch.setattr(f"{module}.monotonic", lambda: now[0])
    return now


def signal(sm, due, send_id):
    """Queue a signal due in ``due`` ms, without running the processing loop."""
    BoundEvent(id="signal", name="Signal", delay=due, _sm=sm).put(send_id=send_id, due=due)


@pytest.mark.timeout(10)
@pytest.mark.parametrize("cancelled", ["s0", "s2", "s9"])
async def test_cancelling_a_delayed_event_keeps_the_rest_in_due_order(sm_runner, clock, cancelled):
    sm = await sm_runner.start(Signals)
    delays = [10, 50, 20, 60, 70, 30, 40, 80, 90, 15]
    for number, delay in enumerate(delays):
        signal(sm, delay, f"s{number}")

    sm.cancel_event(cancelled)
    clock[0] += 1.0
    await sm_runner.processing_loop(sm)

    assert sm.answered == sorted(set(delays) - {delays[int(cancelled[1:])]})


@pytest.mark.timeout(10)
async def test_cancelling_a_send_id_cancels_every_event_sent_with_it(sm_runner, clock):
    sm = await sm_runner.start(Signals)
    signal(sm, 10, "dawn")
    signal(sm, 20, "dawn")
    signal(sm, 30, "dusk")
    signal(sm, 40, "dawn")

    sm.cancel_event("dawn")
    clock[0] += 1.0
    await sm_runner.processing_loop(sm)

    assert sm.answered == [30]


@pytest.mark.timeout(10)
async def test_cancelling_an_unknown_send_id_cancels_nothing(sm_runner, clock):
    sm = await sm_runner.start(Signals)
    signal(sm, 10, "dawn")
    signal(sm, 20, "dusk")

    sm.cancel_event("noon")
    clock[0] += 1.0
    await sm_runner.processing_loop(sm)

    assert sm.answered == [10, 20]


@pytest.mark.timeout(10)
async def test_the_loop_never_waits_for_a_cancelled_event(sm_runner, clock, monkeypatch):
    sm = await sm_runner.start(Signals)
    signal(sm, 10, "dawn")
    sm.cancel_event("dawn")

    def checked():
        raise AssertionError("the loop checked whether a cancelled event was due")

    for module in _CLOCKS[1:]:
        monkeypatch.setattr(f"{module}.monotonic", checked)
    await sm_runner.processing_loop(sm)

    assert sm.answered == []


async def awaiting(sm):
    """Send two delayed signals, each from its own task, and return both tasks.

    The first task runs the processing loop, which waits on the clock for its signal. The
    second signal, sent with ``send_id="dawn"`` and due first, is awaited by the second task.
    """
    holding = asyncio.create_task(sm.send("signal", delay=20, due=20))
    await asyncio.sleep(0)
    waiting = asyncio.create_task(sm.send("signal", delay=10, send_id="dawn", due=10))
    await asyncio.sleep(0)
    return holding, waiting


@pytest.mark.timeout(10)
@pytest.mark.parametrize("cancels", [1, 2])
async def test_a_caller_awaiting_a_cancelled_event_gets_none(clock, cancels):
    sm = AsyncSignals()
    await sm.activate_initial_state()
    holding, waiting = await awaiting(sm)

    for _ in range(cancels):
        sm.cancel_event("dawn")
    await asyncio.sleep(0)

    assert waiting.done()
    assert waiting.result() is None
    clock[0] += 1.0
    await holding
    assert sm.answered == [20]
