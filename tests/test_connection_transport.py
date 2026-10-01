import unittest
from datetime import datetime, timezone

from rampflow.buses import Bus, BusDispatcher
from rampflow.connection_transport import (
    ConnectionTransportCoordinator,
    REBOOK,
    StaleTransportPlan,
    WAIT_SHUTTLE,
)
from rampflow.passenger_connections import (
    Connection,
    ConnectionManager,
    ConnectionState,
    RebookingOption,
)
from rampflow.timeutil import Window

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 10, 1, h, m, tzinfo=UTC)


def make_manager(*, outbound=D(10, 52), assistance=10):
    connection = Connection(
        passenger_id="P1",
        inbound_flight="IN1",
        outbound_flight="OUT1",
        inbound_onblock=D(10, 0),
        outbound_offblock=outbound,
        minimum_connect_min=15,
        assistance_min=assistance,
        same_terminal=False,
    )
    manager = ConnectionManager(terminal_change_min=20)
    manager.register(connection)
    return manager


def make_dispatcher(*, accessible_spaces=2, busy_until=None):
    bus = Bus("B1", seats=10, standing=0, accessible_spaces=accessible_spaces, battery_pct=100)
    dispatcher = BusDispatcher([bus])
    if busy_until is not None:
        dispatcher._use["B1"].append(Window(D(9, 50), busy_until))
    return dispatcher


OPTIONS = (RebookingOption("OUT2", D(12, 0), seats=5, arrival=D(13, 30)),)


class AssistanceTimeTests(unittest.TestCase):
    def test_assistance_time_is_additive_not_maxed_with_transfer(self):
        manager = make_manager()
        connection = manager.connections["P1:OUT1"]
        # 15 base + 20 terminal change + 10 assistance = 45, not 35.
        self.assertEqual(manager.effective_minimum(connection), 45)
        # 52 available minutes minus 45 required = 7.
        self.assertEqual(manager.margin_minutes(connection), 7)

    def test_without_assistance_fix_system_marks_tight_connection_makeable(self):
        # 35-minute connection needs 45 with assistance: it is not makeable.
        manager = make_manager(outbound=D(10, 35))
        connection = manager.connections["P1:OUT1"]
        self.assertLess(manager.margin_minutes(connection), 0)

    def test_expired_protection_hold_becomes_misconnected(self):
        manager = make_manager(outbound=D(10, 35))
        manager.protect("P1", "OUT1", until=D(10, 35))
        connection = manager.connections["P1:OUT1"]
        self.assertIs(
            manager.evaluate(connection, now=D(10, 36)).state,
            ConnectionState.MISCONNECTED,
        )


class ConnectionTransportTests(unittest.TestCase):
    def test_wait_for_bus_when_vehicle_freed_within_deadline(self):
        manager = make_manager()
        dispatcher = make_dispatcher(busy_until=D(10, 34))
        coordinator = ConnectionTransportCoordinator(manager, dispatcher)
        verdict = coordinator.assess(
            "P1", "OUT1", at=D(10, 2),
            stand="S12", terminal="T2", travel_min=5, dwell_min=2,
        )
        self.assertTrue(verdict.makeable)
        self.assertIs(verdict.recommended_action, WAIT_SHUTTLE)
        self.assertEqual(verdict.wait_minutes, 4)
        self.assertEqual(verdict.bus_departs_at, D(10, 34))
        self.assertEqual(verdict.bus_arrives_at, D(10, 41))
        plan = coordinator.commit_transport(verdict)
        self.assertIsNotNone(plan)

    def test_bus_freed_too_late_gives_one_rebook_answer(self):
        manager = make_manager()
        dispatcher = make_dispatcher(busy_until=D(10, 40))
        coordinator = ConnectionTransportCoordinator(manager, dispatcher)
        verdict = coordinator.assess(
            "P1", "OUT1", at=D(10, 2),
            stand="S12", terminal="T2", travel_min=5, dwell_min=2,
            rebooking_candidates=OPTIONS,
        )
        self.assertFalse(verdict.makeable)
        self.assertIs(verdict.recommended_action, REBOOK)
        self.assertIs(verdict.connection_state, ConnectionState.MISCONNECTED)
        self.assertEqual([o.flight_id for o in verdict.rebooking_options], ["OUT2"])
        # No bus window was committed for the missed option.
        self.assertEqual(
            [w for w in dispatcher._use["B1"] if w.start >= D(10, 40)],
            [],
        )

    def test_rebook_releases_previously_held_bus(self):
        manager = make_manager()
        dispatcher = make_dispatcher(busy_until=D(10, 34))
        coordinator = ConnectionTransportCoordinator(manager, dispatcher)
        verdict = coordinator.assess(
            "P1", "OUT1", at=D(10, 2),
            stand="S12", terminal="T2", travel_min=5, dwell_min=2,
        )
        coordinator.commit_transport(verdict)
        held = list(dispatcher._use["B1"])
        self.assertEqual(len(held), 2)

        # Later same clock: the bus hold is no longer usable (fleet problem),
        # the verdict flips to rebook and applying it must free the hold.
        dispatcher._use["B1"].append(Window(D(10, 34), D(11, 30)))
        later = coordinator.assess(
            "P1", "OUT1", at=D(10, 5),
            stand="S12", terminal="T2", travel_min=5, dwell_min=2,
            rebooking_candidates=OPTIONS,
        )
        self.assertIs(later.recommended_action, REBOOK)
        new_connection = coordinator.rebook(later, later.rebooking_options[0])
        self.assertEqual(new_connection.outbound_flight, "OUT2")
        # The window reserved for the abandoned OUT1 plan is gone; the
        # unrelated new block remains.
        self.assertFalse(any(w.end == D(10, 48) for w in dispatcher._use["B1"]))

    def test_inaccessible_fleet_forces_rebook_not_silent_loading(self):
        manager = make_manager()
        dispatcher = make_dispatcher(accessible_spaces=0)
        coordinator = ConnectionTransportCoordinator(manager, dispatcher)
        verdict = coordinator.assess(
            "P1", "OUT1", at=D(10, 2),
            stand="S12", terminal="T2", travel_min=5, dwell_min=2,
            rebooking_candidates=OPTIONS,
        )
        self.assertFalse(verdict.makeable)
        self.assertIs(verdict.recommended_action, REBOOK)
        self.assertEqual(verdict.reason, "fleet-capacity")

    def test_stale_plan_cannot_be_committed(self):
        manager = make_manager()
        dispatcher = make_dispatcher()
        coordinator = ConnectionTransportCoordinator(manager, dispatcher)
        # Evaluated after the shuttle window should have started: the planner
        # still finds a mathematically feasible ride, but committing it now
        # would reserve a bus for a trip that should already have left.
        verdict = coordinator.assess(
            "P1", "OUT1", at=D(10, 40),
            stand="S12", terminal="T2", travel_min=5, dwell_min=2,
        )
        self.assertTrue(verdict.shuttle_plan.feasible)
        with self.assertRaises(StaleTransportPlan):
            coordinator.commit_transport(verdict)
        # Nothing was held.
        self.assertEqual(dispatcher.plans, {})

    def test_expired_plans_are_swept_and_do_not_block_rebooking(self):
        manager = make_manager()
        dispatcher = make_dispatcher(busy_until=D(10, 34))
        coordinator = ConnectionTransportCoordinator(manager, dispatcher)
        verdict = coordinator.assess(
            "P1", "OUT1", at=D(10, 2),
            stand="S12", terminal="T2", travel_min=5, dwell_min=2,
        )
        coordinator.commit_transport(verdict)
        self.assertEqual(set(dispatcher.plans), {"conn:P1:OUT1"})
        retired = dispatcher.expire(D(12, 0))
        self.assertEqual(retired, ("conn:P1:OUT1",))
        self.assertEqual(dispatcher.plans, {})

    def test_duplicate_commit_is_idempotent_and_conflicting_plan_rejected(self):
        manager = make_manager()
        dispatcher = make_dispatcher()
        coordinator = ConnectionTransportCoordinator(manager, dispatcher)
        verdict = coordinator.assess(
            "P1", "OUT1", at=D(10, 2),
            stand="S12", terminal="T2", travel_min=5, dwell_min=2,
        )
        first = coordinator.buses.commit(verdict.shuttle_plan, now=D(10, 2))
        second = coordinator.buses.commit(verdict.shuttle_plan, now=D(10, 2))
        self.assertTrue(first.accepted and second.accepted)
        # A different plan object for the same request must not overwrite.
        other = coordinator.assess(
            "P1", "OUT1", at=D(10, 3),
            stand="S12", terminal="T2", travel_min=5, dwell_min=2,
        ).shuttle_plan
        conflicting = coordinator.buses.commit(other, now=D(10, 3))
        self.assertFalse(conflicting.accepted)


if __name__ == "__main__":
    unittest.main()
