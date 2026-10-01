import unittest
from datetime import datetime, timezone

from rampflow.passenger_connections import (
    Connection,
    ConnectionManager,
    ConnectionState,
    RebookingOption,
)
from rampflow.resource_dispatch import (
    DispatchRequest,
    MobileResource,
    ResourceDispatcher,
    TravelGraph,
)
from rampflow.buses import Bus, BusDispatcher, ShuttleRequest
from rampflow.models import Decision
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected
from rampflow.connections import (
    Connection as RuleConnection,
    ConnectionManager as RuleConnectionManager,
    ConnectionRule,
    ReaccommodationOption,
)

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 10, 1, h, m, tzinfo=UTC)


class SpecialAssistanceConnectionTests(unittest.TestCase):
    def setUp(self):
        self.mgr = ConnectionManager(terminal_change_min=15)

    def test_assistance_adds_to_transfer_in_evaluation(self):
        # 40 min available; need 20 minimum + 15 terminal transfer + 10 assistance = 45
        conn = Connection(
            passenger_id="P1", inbound_flight="IN", outbound_flight="OUT",
            inbound_onblock=D(9), outbound_offblock=D(9, 40),
            minimum_connect_min=20, assistance_min=10, same_terminal=False,
        )
        self.mgr.register(conn)
        self.assertEqual(self.mgr.effective_minimum(conn), 45)
        self.assertEqual(self.mgr.margin_minutes(conn), -5)
        self.assertEqual(self.mgr.evaluate(conn).state, ConnectionState.AT_RISK)

    def test_assistance_added_even_without_transfer(self):
        # 25 min available same terminal; need 20 minimum + 10 assistance = 30
        conn = Connection(
            passenger_id="P1", inbound_flight="IN", outbound_flight="OUT",
            inbound_onblock=D(9), outbound_offblock=D(9, 25),
            minimum_connect_min=20, assistance_min=10, same_terminal=True,
        )
        self.mgr.register(conn)
        self.assertEqual(self.mgr.effective_minimum(conn), 30)
        self.assertEqual(self.mgr.evaluate(conn).state, ConnectionState.AT_RISK)

    def test_evaluation_and_rebooking_use_one_clock(self):
        conn = Connection(
            passenger_id="P1", inbound_flight="IN", outbound_flight="OUT",
            inbound_onblock=D(9), outbound_offblock=D(9, 40),
            minimum_connect_min=20, assistance_min=10, same_terminal=False,
        )
        self.mgr.register(conn)
        # Evaluate says the connection cannot be made.
        self.assertEqual(self.mgr.evaluate(conn).state, ConnectionState.AT_RISK)
        # 09:29 is one minute short of the 09:30 earliest (same terminal, 20 + 10).
        too_tight = RebookingOption("OUT3", D(9, 29), 1, D(11), terminal_change_min=0)
        self.assertEqual(self.mgr.rebooking_options("P1", "OUT", [too_tight]), ())
        # 09:30 exactly is catchable by the same accounting.
        ok = RebookingOption("OUT2", D(9, 30), 1, D(11), terminal_change_min=0)
        chosen = self.mgr.rebooking_options("P1", "OUT", [too_tight, ok])
        self.assertEqual([o.flight_id for o in chosen], ["OUT2"])


class ResourceAvailabilityDispatchTests(unittest.TestCase):
    def test_plan_waits_for_resource_available_at(self):
        g = TravelGraph()
        g.add("stand", "gate", 0)
        bus = MobileResource("B1", "shuttle", frozenset({"wheelchair"}),
                             "stand", available_at=D(9, 5))
        disp = ResourceDispatcher([bus], g)
        req = DispatchRequest("Q1", "shuttle", "wheelchair", "gate",
                              earliest=D(9), service_min=5, latest_finish=D(9, 12))
        plan = disp.plan(req)
        # Bus only frees up at 09:05; that clock must carry through to commit.
        self.assertEqual(plan.depart_at, D(9, 5))
        self.assertEqual(plan.arrive_at, D(9, 5))
        disp.commit(plan)  # must not raise "resource became busy"
        self.assertEqual(disp.state("B1").available_at, D(9, 10))

    def test_unavailable_in_time_is_not_offered(self):
        g = TravelGraph()
        g.add("stand", "gate", 0)
        bus = MobileResource("B1", "shuttle", frozenset(), "stand",
                             available_at=D(9, 10))
        disp = ResourceDispatcher([bus], g)
        req = DispatchRequest("Q1", "shuttle", None, "gate",
                              earliest=D(9), service_min=5, latest_finish=D(9, 12))
        with self.assertRaises(Exception):
            disp.plan(req)


class ShuttlePlanExpiryTests(unittest.TestCase):
    def _dispatcher_with_committed_plan(self):
        bus = Bus("SB1", seats=10, standing=0, accessible_spaces=2, battery_pct=100.0)
        bd = BusDispatcher([bus])
        req = ShuttleRequest(
            "R1", "IN", 1, 1, "stand", "T",
            ready_at=D(9), latest_arrival=D(9, 15), travel_min=5,
        )
        plan = bd.plan(req)
        self.assertTrue(plan.feasible)
        self.assertTrue(bd.commit(plan).accepted)
        self.assertTrue(bd._use["SB1"])
        return bd, req

    def test_plan_holds_bus_before_deadline(self):
        bd, _ = self._dispatcher_with_committed_plan()
        self.assertEqual(bd.expire_before(D(9, 14)), ())
        self.assertTrue(bd._use["SB1"])

    def test_expired_plan_releases_bus_and_energy(self):
        bd, _ = self._dispatcher_with_committed_plan()
        released = bd.expire_before(D(9, 15))
        self.assertEqual(released, ("R1",))
        self.assertEqual(bd._use["SB1"], [])
        self.assertNotIn("R1", bd.plans)
        self.assertAlmostEqual(bd.buses["SB1"].battery_pct, 100.0)

        follow_up = ShuttleRequest(
            "R2", "IN2", 1, 1, "stand", "T",
            ready_at=D(9, 20), latest_arrival=D(9, 35), travel_min=5,
        )
        plan = bd.plan(follow_up)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.final_arrival, D(9, 28))

    def test_explicit_cancel_releases_bus(self):
        bd, _ = self._dispatcher_with_committed_plan()
        self.assertTrue(bd.cancel("R1"))
        self.assertEqual(bd._use["SB1"], [])
        self.assertFalse(bd.cancel("R1"))  # idempotent / missing


class ConnectionTransportWorkflowTests(unittest.TestCase):
    def test_event_follows_successful_domain_change_only(self):
        control = RampWorkflowControl()
        try:
            control.update_connection_transport(
                "op-1", "P1", D(10),
                lambda: (_ for _ in ()).throw(RuntimeError("bus late")),
            )
        except RuntimeError:
            pass
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})

    def test_rejected_requirement_blocks_event(self):
        control = RampWorkflowControl()
        denied = Decision(False, "misconnect", "cannot reach outbound")
        with self.assertRaises(WorkflowRejected):
            control.update_connection_transport(
                "op-2", "P1", D(10), lambda: "x", requirements=(denied,),
            )
        self.assertEqual(control.stream.head, 0)

    def test_success_publishes_once_and_is_idempotent(self):
        control = RampWorkflowControl()
        first = control.update_connection_transport("op-3", "P1", D(10), lambda: "ok")
        second = control.update_connection_transport("op-3", "P1", D(10), lambda: "dup")
        self.assertIs(first, second)
        self.assertEqual(first.event.topic, "connection.transport.updated")
        self.assertEqual(control.stream.head, 1)


class RuleBasedConnectionReaccommodationTests(unittest.TestCase):
    def _manager(self):
        rule = ConnectionRule(
            inbound_type="D", outbound_type="D", terminal_pair=("*", "*"),
            minimum_min=20, mobility_extra_min=10,
        )
        mgr = RuleConnectionManager([rule], {("T1", "T1"): 0})
        conn = RuleConnection(
            connection_id="C1", inbound_flight="IN", outbound_flight="OUT",
            arrival=D(9), departure=D(9, 25),
            inbound_terminal="T1", outbound_terminal="T1",
            inbound_type="D", outbound_type="D",
            passengers=1, checked_bags=0, mobility_passengers=1,
        )
        mgr.protect(conn)
        return mgr, conn

    def test_status_marks_mobility_connection_missed(self):
        mgr, conn = self._manager()
        # 25 min available vs 20 minimum + 10 mobility = 30 required.
        decision = mgr.status(conn)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "misconnect")

    def test_reaccommodation_reserves_mobility_time(self):
        mgr, conn = self._manager()
        # 09:29 is one minute short of the 09:30 readiness (20 + 10).
        too_tight = ReaccommodationOption("X1", D(9, 29), 1, "T1")
        ok = ReaccommodationOption("X2", D(9, 30), 1, "T1")
        chosen = mgr.reaccommodate("C1", [too_tight, ok], D(9))
        self.assertEqual(chosen.flight_id, "X2")


if __name__ == "__main__":
    unittest.main()
