"""Cancelling delayed events with ``cancel_event()``.

A cancelled trigger stays where it is in its queue, so the triggers around it keep their due
order, and is dropped when it reaches the head. A caller awaiting it is answered with ``None``.

Theme: signal fires, some of them called off before they are lit.
"""

import asyncio

import pytest
from statemachine.engines import async_
from statemachine.engines import sync
from statemachine.event import BoundEvent

from statemachine import State
from statemachine import StateChart
from statemachine import event_data

_CLOCKS = ("statemachine.event_data", "statemachine.engines.sync", "statemachine.engines.async_")


class Signals(StateChart):
    idle = State(initial=True)

    signal = idle.to.itself(internal=True, on="answer")
    stand_down = idle.to.itself(internal=True, on="call_off")

    def __init__(self, *args, **kwargs):
        self.answered = []
        super().__init__(*args, **kwargs)

    def answer(self, due):
        self.answered.append(due)

    def call_off(self, which):
        self.cancel_event(which)


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


def signal(sm, due, send_id=None):
    """Queue a signal due in ``due`` ms, without running the processing loop."""
    return BoundEvent(id="signal", name="Signal", delay=due, _sm=sm).put(send_id=send_id, due=due)


def test_the_clock_fixture_moves_every_clock_the_engines_read(clock):
    modules = (event_data, sync, async_)
    assert [module.monotonic() for module in modules] == [1000.0, 1000.0, 1000.0]

    clock[0] += 0.5

    assert [module.monotonic() for module in modules] == [1000.5, 1000.5, 1000.5]


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "cancelled",
    [
        pytest.param("s0", id="first-due-and-first-queued"),
        pytest.param("s6", id="due-in-the-middle"),
        pytest.param("s8", id="last-due"),
        pytest.param("s9", id="last-queued"),
    ],
)
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
async def test_cancelling_two_send_ids_in_turn_cancels_each_and_nothing_else(sm_runner, clock):
    sm = await sm_runner.start(Signals)
    signal(sm, 10, "dawn")
    signal(sm, 20, "dusk")
    signal(sm, 30, "noon")

    sm.cancel_event("dawn")
    sm.cancel_event("dusk")
    clock[0] += 1.0
    await sm_runner.processing_loop(sm)

    assert sm.answered == [30]


@pytest.mark.timeout(10)
async def test_a_cancel_with_nothing_queued_does_not_cancel_what_is_sent_later(sm_runner, clock):
    sm = await sm_runner.start(Signals)

    sm.cancel_event("dawn")
    signal(sm, 10, "dawn")
    clock[0] += 1.0
    await sm_runner.processing_loop(sm)

    assert sm.answered == [10]


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    ("sent", "cancelled"),
    [
        pytest.param("dawn", "", id="empty"),
        pytest.param("dawn", " ", id="space"),
        pytest.param("dawn", " dawn", id="leading-space"),
        pytest.param("dawn", "dawn ", id="trailing-space"),
        pytest.param("dawn", "dawn\n", id="newline"),
        pytest.param("dawn\r\n", "dawn\n", id="crlf-and-lf"),
        pytest.param("dawn", "Dawn", id="case"),
        pytest.param("dawn", "dawn​", id="zero-width-space"),
        pytest.param("dawn", "﻿dawn", id="byte-order-mark"),
        pytest.param("da\0wn", "da", id="up-to-a-nul"),
        pytest.param("café", "café", id="composed-and-decomposed"),
    ],
)
async def test_a_send_id_cancels_only_events_sent_with_exactly_that_id(
    sm_runner, clock, sent, cancelled
):
    sm = await sm_runner.start(Signals)
    signal(sm, 10, sent)

    sm.cancel_event(cancelled)
    clock[0] += 1.0
    await sm_runner.processing_loop(sm)

    assert sm.answered == [10]


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "send_id",
    [
        pytest.param("", id="empty"),
        pytest.param(" ", id="space"),
        pytest.param("dawn\n", id="newline"),
        pytest.param("da\0wn", id="nul"),
        pytest.param("café", id="decomposed"),
        pytest.param("\U0001f525", id="astral"),
        pytest.param("\u6681", id="cjk"),
        pytest.param("\u202edawn", id="right-to-left-override"),
        pytest.param("x" * 255, id="255-characters"),
        pytest.param("x" * 256, id="256-characters"),
        pytest.param("x" * 257, id="257-characters"),
        pytest.param("x" * 1024, id="1024-characters"),
        pytest.param("x" * 2048, id="2048-characters"),
    ],
)
async def test_any_string_is_a_send_id_that_cancels_only_its_own_events(sm_runner, clock, send_id):
    sm = await sm_runner.start(Signals)
    signal(sm, 10, send_id)
    signal(sm, 20)

    sm.cancel_event(send_id)
    clock[0] += 1.0
    await sm_runner.processing_loop(sm)

    assert sm.answered == [20]


@pytest.mark.timeout(10)
async def test_cancelling_marks_each_trigger_sent_with_the_id(sm_runner):
    sm = await sm_runner.start(Signals)
    dawn = signal(sm, 10, "dawn")
    dusk = signal(sm, 20, "dusk")

    sm.cancel_event("dawn")

    assert (dawn.cancelled, dusk.cancelled) == (True, False)


@pytest.mark.timeout(10)
async def test_an_event_can_be_cancelled_from_a_callback(sm_runner, clock):
    sm = await sm_runner.start(Signals)
    BoundEvent(id="stand_down", name="Stand down", _sm=sm).put(which="dawn")
    signal(sm, 10, "dawn")
    signal(sm, 20, "dusk")

    clock[0] += 1.0
    await sm_runner.processing_loop(sm)

    assert sm.answered == [20]


@pytest.mark.timeout(10)
async def test_the_loop_never_waits_for_a_cancelled_event(sm_runner, clock, monkeypatch):
    sm = await sm_runner.start(Signals)
    dawn = signal(sm, 10, "dawn")
    sm.cancel_event("dawn")

    def checked():
        raise AssertionError("the loop checked whether a cancelled event was due")

    for module in _CLOCKS[1:]:
        monkeypatch.setattr(f"{module}.monotonic", checked)
    await sm_runner.processing_loop(sm)

    assert (dawn.cancelled, sm.answered) == (True, [])


async def awaiting(sm, started=True):
    """Queue two delayed signals, each awaited by its own task, and return the tasks.

    The first task runs the processing loop, which waits on the clock for its signal. The
    second signal, sent with ``send_id="dawn"`` and due first, is awaited through its trigger's
    future by the second task, as ``send()`` awaits it, once that task has started. The
    trigger is returned too.
    """
    holding = asyncio.create_task(sm.send("signal", delay=20, due=20))
    await asyncio.sleep(0)
    dawn = signal(sm, 10, "dawn")
    waiting = asyncio.create_task(sm._processing_loop(dawn.future))
    if started:
        await asyncio.sleep(0)
    return holding, waiting, dawn


def settled(future):
    """How a settled future was settled: cancelled, or answered with a value."""
    if future.cancelled():
        return "cancelled"
    return f"answered with {future.result()!r}"


@pytest.mark.timeout(10)
@pytest.mark.parametrize("cancels", [1, 2])
async def test_a_caller_awaiting_a_cancelled_event_gets_none(clock, cancels):
    sm = AsyncSignals()
    await sm.activate_initial_state()
    holding, waiting, dawn = await awaiting(sm)

    for _ in range(cancels):
        sm.cancel_event("dawn")
    await asyncio.sleep(0)

    assert (settled(dawn.future), settled(waiting)) == ("answered with None", "answered with None")
    clock[0] += 1.0
    await holding
    assert sm.answered == [20]


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    ("started", "settled_as"),
    [
        pytest.param(False, "answered with None", id="before-it-runs"),
        pytest.param(True, "cancelled", id="while-it-awaits"),
    ],
)
async def test_a_caller_that_stops_waiting_leaves_its_event_cancellable(
    clock, started, settled_as
):
    sm = AsyncSignals()
    await sm.activate_initial_state()
    holding, waiting, dawn = await awaiting(sm, started)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    sm.cancel_event("dawn")
    clock[0] += 1.0
    await holding

    assert settled(dawn.future) == settled_as
    assert sm.answered == [20]
