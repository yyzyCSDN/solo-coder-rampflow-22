"""Single-clock decision for assisted connecting passengers and shuttle buses.

The passenger connection domain and the bus dispatch domain each plan against
their own data.  Evaluating them separately lets two answers disagree: the
connection can be marked makeable while the shuttle plan is late, or a bus can
be held for a connection that is no longer protected.  This coordinator plans
both sides at the same point in time (special-assistance handling included)
and returns one consistent verdict: wait for the bus, or rebook.  A rebook
verdict releases the transport plan so an expired/abandoned option stops
holding the bus.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Iterable

from .buses import BusDispatcher, ShuttlePlan, ShuttleRequest
from .passenger_connections import (
    Connection,
    ConnectionManager,
    ConnectionState,
    RebookingOption,
)
from .timeutil import utc

WAIT_SHUTTLE = "wait_shuttle"
REBOOK = "rebook"


@dataclass(frozen=True)
class ConnectionTransportVerdict:
    passenger_id: str
    outbound_flight: str
    at: datetime
    passenger_ready_at: datetime
    bus_departs_at: datetime
    bus_arrives_at: datetime
    gate_deadline: datetime
    wait_minutes: int
    makeable: bool
    recommended_action: str
    reason: str
    connection_state: ConnectionState
    shuttle_plan: ShuttlePlan | None
    rebooking_options: tuple[RebookingOption, ...] = ()


class ConnectionTransportCoordinator:
    def __init__(
        self,
        connections: ConnectionManager,
        buses: BusDispatcher,
        gate_close_min: int = 10,
    ):
        if gate_close_min < 0:
            raise ValueError("gate_close_min cannot be negative")
        self.connections = connections
        self.buses = buses
        self.gate_close_min = gate_close_min

    def assess(
        self,
        passenger_id: str,
        outbound_flight: str,
        *,
        at: datetime,
        stand: str,
        terminal: str,
        travel_min: int,
        dwell_min: int = 3,
        accessible: bool | None = None,
        rebooking_candidates: Iterable[RebookingOption] = (),
        destination_arrival_limit: datetime | None = None,
    ) -> ConnectionTransportVerdict:
        """Evaluate passenger readiness and bus availability against ``at``.

        Special-assistance handling time is added to the passenger's readiness
        exactly as it is added to the connection minimum (see
        ``ConnectionManager.effective_minimum``); the shuttle is only allowed
        to depart once that handling is complete.  The deadline is the
        outbound departure minus the gate-close buffer.
        """
        at = utc(at)
        key = self.connections._key(passenger_id, outbound_flight)
        connection = self.connections.connections[key]
        inbound = self.connections.flight_arrivals.get(
            connection.inbound_flight,
            connection.inbound_onblock,
        )
        outbound = self.connections.flight_departures.get(
            connection.outbound_flight,
            connection.outbound_offblock,
        )

        transfer_min = 0 if connection.same_terminal else self.connections.terminal_change_min
        ready_at = inbound + timedelta(
            minutes=connection.assistance_min + transfer_min
        )
        gate_deadline = outbound - timedelta(minutes=self.gate_close_min)
        needs_accessible = connection.assistance_min > 0 if accessible is None else accessible

        plan: ShuttlePlan | None = None
        bus_departs = ready_at
        bus_arrives = ready_at
        wait_minutes = 0
        makeable = False
        action = REBOOK
        reason = ""

        if ready_at > gate_deadline or at > gate_deadline:
            reason = "assistance-handling-exceeds-gate-deadline"
        else:
            req = ShuttleRequest(
                request_id=self._request_id(passenger_id, outbound_flight),
                flight_id=connection.outbound_flight,
                passengers=1,
                accessible_passengers=1 if needs_accessible else 0,
                stand=stand,
                terminal=terminal,
                ready_at=ready_at,
                latest_arrival=gate_deadline,
                travel_min=travel_min,
                dwell_min=dwell_min,
            )
            plan = self.buses.plan(req)
            first = plan.trips[0] if plan.trips else None
            bus_departs = first.outbound.start if first else ready_at
            bus_arrives = plan.final_arrival
            wait_minutes = max(
                0,
                int((bus_departs - ready_at).total_seconds() // 60),
            )
            if plan.feasible and bus_arrives <= gate_deadline:
                makeable = True
                action = WAIT_SHUTTLE
                reason = "shuttle-meets-gate-deadline"
            else:
                makeable = False
                action = REBOOK
                reason = plan.reason if not plan.feasible else "shuttle-late-at-gate"

        evaluated = self.connections.evaluate(connection, now=at)
        if makeable:
            # The transport side proves the connection is physically makeable;
            # keep the two domains on one answer.
            state = evaluated.state
        else:
            # No bus can deliver the passenger to the gate on time, so the
            # connection cannot remain merely at-risk: it is misconnected and
            # the only consistent recommendation is to rebook.
            state = ConnectionState.MISCONNECTED
        stored = evaluated if evaluated.state == state else replace(evaluated, state=state)
        self.connections.connections[key] = stored

        options: tuple[RebookingOption, ...] = ()
        if action == REBOOK:
            options = self.connections.rebooking_options(
                passenger_id,
                outbound_flight,
                tuple(rebooking_candidates),
                destination_arrival_limit,
            )

        return ConnectionTransportVerdict(
            passenger_id=passenger_id,
            outbound_flight=outbound_flight,
            at=at,
            passenger_ready_at=ready_at,
            bus_departs_at=bus_departs,
            bus_arrives_at=bus_arrives,
            gate_deadline=gate_deadline,
            wait_minutes=wait_minutes,
            makeable=makeable,
            recommended_action=action,
            reason=reason,
            connection_state=state,
            shuttle_plan=plan,
            rebooking_options=options,
        )

    def commit_transport(
        self,
        verdict: ConnectionTransportVerdict,
    ) -> ShuttlePlan | None:
        """Hold the bus for a WAIT verdict; reject anything stale.

        Committing against the verdict's evaluation time means a plan whose
        window has already slipped past is refused instead of silently
        reserving a bus for a past trip.
        """
        if verdict.recommended_action != WAIT_SHUTTLE or verdict.shuttle_plan is None:
            return None
        if not verdict.shuttle_plan.feasible:
            raise StaleTransportPlan(
                verdict.shuttle_plan.reason,
                "no feasible shuttle plan to commit",
            )
        decision = self.buses.commit(verdict.shuttle_plan, now=verdict.at)
        if not decision.accepted:
            raise StaleTransportPlan(decision.code, decision.reason, decision.details)
        return verdict.shuttle_plan

    def rebook(
        self,
        verdict: ConnectionTransportVerdict,
        option: RebookingOption,
    ) -> Connection:
        """Move the passenger to the chosen flight and free the old bus hold.

        A rebook and the release of the abandoned shuttle plan are one
        decision: the expired/missed option cannot keep occupying the vehicle.
        """
        if verdict.recommended_action != REBOOK:
            raise ValueError("verdict is makeable; rebooking not warranted")
        if option not in verdict.rebooking_options:
            raise ValueError("option was not offered by the verdict")
        old_request = self._request_id(
            verdict.passenger_id,
            verdict.outbound_flight,
        )
        self.buses.release(old_request, verdict.at)
        return self.connections.apply_rebooking(
            verdict.passenger_id,
            verdict.outbound_flight,
            option,
        )

    @staticmethod
    def _request_id(passenger_id: str, outbound_flight: str) -> str:
        return f"conn:{passenger_id}:{outbound_flight}"


class StaleTransportPlan(RuntimeError):
    def __init__(self, code: str, reason: str, details: object = None):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {reason}")
