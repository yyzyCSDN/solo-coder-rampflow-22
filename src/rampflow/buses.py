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

    def _energy_cost(self,travel_min:int,dwell_min:int)->float:
        return travel_min*0.55+dwell_min*0.08

    def _next_free(self,bus_id:str,at:datetime)->datetime:
        cur=at
        changed=True
        while changed:
            changed=False
            for w in sorted(self._use[bus_id],key=lambda x:x.start):
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
                start=req.ready_at
                for w in sorted(virtual_use[bid],key=lambda x:x.start):
                    if w.start<=start<w.end:start=w.end
                # A bus with no free wheelchair space cannot pick up a trip
                # that still has accessible passengers waiting, otherwise the
                # accessible rider is silently loaded as a standing passenger.
                if accessible>0 and b.accessible_spaces<=0:continue
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
        return ShuttlePlan(req.request_id,tuple(trips),final,True)

    def commit(self,plan:ShuttlePlan,now:datetime|None=None)->Decision:
        if not plan.feasible:return Decision(False,"infeasible",plan.reason)
        existing=self.plans.get(plan.request_id)
        if existing is not None:
            if existing==plan:return Decision(True,"ok","shuttle plan already committed",{"arrival":plan.final_arrival})
            return Decision(False,"request-replanned","request already holds another plan")
        moment=utc(now) if now is not None else None
        first_start=min((t.outbound.start for t in plan.trips),default=None)
        if moment is not None and first_start is not None and first_start<=moment:
            return Decision(False,"plan-stale","plan window already started; replan against current time")
        staged=[]
        battery={bid:b.battery_pct for bid,b in self.buses.items()}
        for trip in plan.trips:
            full=Window(trip.outbound.start,trip.return_window.end)
            if moment is not None and full.end<=moment:
                return Decision(False,"plan-expired","shuttle window is in the past",{"bus":trip.bus_id})
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
        return Decision(True,"ok","shuttle plan committed",{"arrival":plan.final_arrival})

    def release(self,request_id:str,now:datetime)->Decision:
        """Drop a committed plan and give back any bus time not yet consumed.

        Expired plans (whose last trip has already ended) are simply removed:
        historical occupancy stays on the ledger, but future holds are taken
        off the bus so another request can use it.  Plans whose vehicle is
        currently out cannot be released automatically.
        """
        now=utc(now)
        plan=self.plans.pop(request_id,None)
        if plan is None:return Decision(False,"unknown-plan","no committed plan for request",{"request":request_id})
        freed=[]
        for trip in plan.trips:
            full=Window(trip.outbound.start,trip.return_window.end)
            if full.start<=now<full.end:
                self.plans[request_id]=plan
                return Decision(False,"bus-en-route","bus already serving the request",{"bus":trip.bus_id})
            if full.end<=now:continue
            if full in self._use[trip.bus_id]:
                self._use[trip.bus_id].remove(full)
                freed.append(trip.bus_id)
        return Decision(True,"ok","shuttle plan released",{"buses":tuple(sorted(set(freed)))})

    def expire(self,now:datetime)->tuple[str,...]:
        """Sweep committed plans whose transport window has fully elapsed.

        Returns the request ids that were retired.  Expired plans must not
        keep holding a bus; past windows cannot conflict with future trips,
        but retiring them keeps commit/release state consistent with one
        clock and prevents replays from being treated as live reservations.
        """
        now=utc(now)
        retired=[]
        for request_id,plan in tuple(self.plans.items()):
            last_end=max((t.return_window.end for t in plan.trips),default=None)
            if last_end is not None and last_end<=now:
                del self.plans[request_id]
                retired.append(request_id)
        return tuple(sorted(retired))

    def charge(self,bus_id:str,window:Window)->Bus:
        if any(w.overlaps(window) for w in self._use[bus_id]):raise RuntimeError("bus busy during charge")
        b=self.buses[bus_id]
        pct=min(100.0,b.battery_pct+window.minutes*b.charge_pct_per_min)
        self._use[bus_id].append(window)
        self.buses[bus_id]=replace(b,battery_pct=pct)
        return self.buses[bus_id]
