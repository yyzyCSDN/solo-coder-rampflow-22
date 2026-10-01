from __future__ import annotations
from dataclasses import dataclass,replace
from datetime import datetime,timedelta
from typing import Iterable
from .timeutil import Window,utc
from .models import Decision

@dataclass(frozen=True)
class Bus:
    bus_id:str
    seats:int
    standing:int
    accessible_spaces:int
    battery_pct:float
    min_dispatch_pct:float=20.0
    charge_pct_per_min:float=1.0
    def capacity(self,accessible_needed:int=0)->int:
        if accessible_needed>self.accessible_spaces:return 0
        return self.seats+self.standing-accessible_needed

@dataclass(frozen=True)
class ShuttleRequest:
    request_id:str
    flight_id:str
    passengers:int
    accessible_passengers:int
    stand:str
    terminal:str
    ready_at:datetime
    latest_arrival:datetime
    travel_min:int
    dwell_min:int=3
    def __post_init__(self):
        object.__setattr__(self,"ready_at",utc(self.ready_at));object.__setattr__(self,"latest_arrival",utc(self.latest_arrival))
        if self.passengers<=0 or self.accessible_passengers<0 or self.accessible_passengers>self.passengers:raise ValueError("invalid passenger counts")

@dataclass(frozen=True)
class ShuttleTrip:
    bus_id:str
    trip_no:int
    passengers:int
    accessible:int
    outbound:Window
    return_window:Window
    battery_after:float

@dataclass(frozen=True)
class ShuttlePlan:
    request_id:str
    trips:tuple[ShuttleTrip,...]
    final_arrival:datetime
    feasible:bool
    reason:str=""

class BusDispatcher:
    def __init__(self,buses:Iterable[Bus]):
        self.buses={b.bus_id:b for b in buses}
        self._use:dict[str,list[Window]]={b:[] for b in self.buses}
        self.plans:dict[str,ShuttlePlan]={}
        self._requests:dict[str,ShuttleRequest]={}
        self._planned_requests:dict[str,ShuttleRequest]={}

    def _energy_cost(self,travel_min:int,dwell_min:int)->float:
        return travel_min*0.55+dwell_min*0.08

    def _trip_window(self,trip:ShuttleTrip)->Window:
        return Window(trip.outbound.start,trip.return_window.end)

    def expire_before(self,now:datetime)->tuple[str,...]:
        """Release committed plans whose service deadline has passed.

        A plan is stale once ``now`` reaches its request's ``latest_arrival``:
        it can no longer get the passenger to the connection, so continuing to
        block its buses only strands later requests.  Returns the released
        request ids.
        """
        now=utc(now)
        stale=tuple(
            rid for rid,req in self._requests.items()
            if now>=req.latest_arrival
        )
        for rid in stale:
            self.cancel(rid)
        return stale

    def cancel(self,request_id:str)->bool:
        """Drop a committed plan and give its buses/energy back.

        Used both for explicit rebooking decisions and for expiry: a stale or
        superseded shuttle plan must keep holding neither the bus nor charge.
        """
        plan=self.plans.pop(request_id,None)
        req=self._requests.pop(request_id,None)
        self._planned_requests.pop(request_id,None)
        if plan is None:return False
        for trip in plan.trips:
            full=self._trip_window(trip)
            self._use[trip.bus_id]=[w for w in self._use[trip.bus_id] if w!=full]
        if req is not None:
            refund=self._energy_cost(req.travel_min*2,req.dwell_min*2)
            per_bus:dict[str,int]={}
            for trip in plan.trips:
                per_bus[trip.bus_id]=per_bus.get(trip.bus_id,0)+1
            for bid,count in per_bus.items():
                b=self.buses[bid]
                self.buses[bid]=replace(b,battery_pct=min(100.0,b.battery_pct+refund*count))
        return True

    def _next_free(self,windows:list[Window],at:datetime)->datetime:
        cur=at
        changed=True
        while changed:
            changed=False
            for w in sorted(windows,key=lambda x:x.start):
                if w.start<=cur<w.end:
                    cur=w.end;changed=True
        return cur

    def plan(self,req:ShuttleRequest)->ShuttlePlan:
        remaining=req.passengers
        accessible=req.accessible_passengers
        virtual={bid:b for bid,b in self.buses.items()}
        virtual_use={bid:list(v) for bid,v in self._use.items()}
        trips=[]
        trip_no=0
        while remaining>0:
            options=[]
            for bid,b in virtual.items():
                start=self._next_free(virtual_use[bid],req.ready_at)
                acc=min(accessible,b.accessible_spaces)
                cap=b.capacity(acc)
                if cap<=0:continue
                load=min(remaining,cap)
                out=Window(start,start+timedelta(minutes=req.dwell_min+req.travel_min))
                ret=Window(out.end,out.end+timedelta(minutes=req.travel_min+req.dwell_min))
                energy=self._energy_cost(req.travel_min*2,req.dwell_min*2)
                after=b.battery_pct-energy
                if after<b.min_dispatch_pct:continue
                options.append((out.end,-load,bid,load,acc,out,ret,after))
            if not options:
                return ShuttlePlan(req.request_id,tuple(trips),trips[-1].outbound.end if trips else req.ready_at,False,"fleet-capacity")
            options.sort()
            _,_,bid,load,acc,out,ret,after=options[0]
            trip_no+=1
            trips.append(ShuttleTrip(bid,trip_no,load,acc,out,ret,after))
            virtual[bid]=replace(virtual[bid],battery_pct=after)
            virtual_use[bid].append(ret.expand(before=int((out.end-out.start).total_seconds()//60),after=0))
            remaining-=load;accessible-=acc
        final=max(t.outbound.end for t in trips)
        if final>req.latest_arrival:
            return ShuttlePlan(req.request_id,tuple(trips),final,False,"late")
        plan=ShuttlePlan(req.request_id,tuple(trips),final,True)
        self._planned_requests[req.request_id]=req
        return plan

    def commit(self,plan:ShuttlePlan,request:ShuttleRequest|None=None)->Decision:
        if not plan.feasible:return Decision(False,"infeasible",plan.reason)
        if plan.request_id in self.plans and self.plans[plan.request_id]==plan:
            return Decision(True,"ok","shuttle plan committed",{"arrival":plan.final_arrival})
        req=request or self._planned_requests.get(plan.request_id)
        staged=[]
        battery={bid:b.battery_pct for bid,b in self.buses.items()}
        for trip in plan.trips:
            full=self._trip_window(trip)
            if any(w.overlaps(full) for w in self._use[trip.bus_id]):
                return Decision(False,"bus-stale","bus became unavailable",{"bus":trip.bus_id})
            cost=battery[trip.bus_id]-trip.battery_after
            if trip.battery_after<self.buses[trip.bus_id].min_dispatch_pct:
                return Decision(False,"battery","bus reserve violated")
            battery[trip.bus_id]=trip.battery_after
            staged.append((trip.bus_id,full))
        for bid,full in staged:self._use[bid].append(full)
        for bid,value in battery.items():
            if value!=self.buses[bid].battery_pct:self.buses[bid]=replace(self.buses[bid],battery_pct=value)
        self.plans[plan.request_id]=plan
        if req is not None:self._requests[plan.request_id]=req
        return Decision(True,"ok","shuttle plan committed",{"arrival":plan.final_arrival})

    def charge(self,bus_id:str,window:Window)->Bus:
        if any(w.overlaps(window) for w in self._use[bus_id]):raise RuntimeError("bus busy during charge")
        b=self.buses[bus_id]
        pct=min(100.0,b.battery_pct+window.minutes*b.charge_pct_per_min)
        self._use[bus_id].append(window)
        self.buses[bus_id]=replace(b,battery_pct=pct)
        return self.buses[bus_id]
