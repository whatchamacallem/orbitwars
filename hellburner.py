import math
import time
import sys
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Fleet, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS,
    distance, point_to_segment_distance
)

sys.path.insert(0, '/home/t/orbitwars')
from visualizer import Visualizer
viz = Visualizer()
def viz_save(seed=None):
    viz.save('/mnt/c/Users/ajohn/Downloads/orbitwars_viz.html', seed=seed)

@dataclass(slots=True)
class Pos:
    x: float
    y: float

@dataclass(slots=True, eq=False)
class HPlanet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: float
    production: float
    # nearest owned planet on shortest path to front
    reinforcement_target: 'HPlanet | None' = field(default=None)

@dataclass(slots=True)
class OrbitalEntry:
    r: float
    initial_angle: float

@dataclass(slots=True)
class GraphEdge:
    planet: HPlanet
    travel: float

@dataclass(slots=True)
class Intercept:
    angle: float
    x: float
    y: float
    travel: float

@dataclass(slots=True)
class Arrival:
    owner: int
    ships: float
    travel_time: float
    src_x: float
    src_y: float
    arrival_x: float
    arrival_y: float

@dataclass(slots=True)
class FleetOrder:
    planet_id: int
    angle: float
    ships: int

@dataclass(slots=True)
class Assignment:
    ships: float
    launch_turn: int
    arrival_turn: int

@dataclass(slots=True)
class OwnerShips:
    owner: int
    ships: float

@dataclass(slots=True)
class WarchestFleet:
    owner: int
    destination_id: int
    fleet_size: float
    garrison_on_arrival: float   # estimated net ships after battle resolution
    arrival_turn: int
    is_capture: bool             # True if destination is not currently owned by owner
    production_id: int = -1      # planet whose production to add after capture (-1 = none)

@dataclass(slots=True)
class WarchestState:
    turn: int
    garrison: dict[int, float]    # planet_id -> current ships on planet
    production: dict[int, float]  # planet_id -> ships/turn (non-comet planets only)
    ownership: dict[int, int]     # planet_id -> owner id (-1, 0, 1, ...)
    friendly_fleets: list[WarchestFleet]
    enemy_fleets: list[WarchestFleet]
    committed_ids: set[int]       # planet IDs already used as sources this planning pass

# HPlanet -> OrbitalEntry if the planet orbits the sun, else None
OrbitalInfo = dict[HPlanet, OrbitalEntry | None]
# HPlanets rotated by ROTATION_LOOK_AHEAD
FuturePos = dict[HPlanet, Pos]
# dst -> [GraphEdge(src, travel_steps)]: directed graph; src departs now, dst is its intercept position
ProximityGraph = dict[HPlanet, list[GraphEdge]]
# HPlanet -> [Arrival(...)]
DestinationList = dict[HPlanet, list[Arrival]]
FleetOrders = list[FleetOrder]


SHIP_SPEED_MAX = 6.0

def fleet_speed(ships: int | float, ship_speed_max: float = SHIP_SPEED_MAX) -> float:
    """Mirror the engine's speed formula exactly."""
    return min(ship_speed_max, 1.0 + (ship_speed_max - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5)


def warchest_state_copy(state: 'WarchestState') -> 'WarchestState':
    return WarchestState(
        turn=state.turn,
        garrison=dict(state.garrison),
        production=state.production,                  # read-only during search; share reference
        ownership=dict(state.ownership),
        friendly_fleets=list(state.friendly_fleets),  # WarchestFleet is immutable
        enemy_fleets=state.enemy_fleets,              # never mutated during search; share reference
        committed_ids=set(state.committed_ids),
    )


class Hellburner:
    def __init__(self):

        self.COMET_RADIUS        = 20.0   # inflated comet radius for path-blocking checks

        self.MAX_DISTANCE        = 38
        self.ROTATION_LOOK_AHEAD = 10
        self.REINFORCEMENT_SIZE  = 17
        self.GARRISON_SIZE       = 11

        # Warchest
        self.TIME_BUDGET_S       = 0.800  # hard wall; return best-found-so-far if exceeded
        self.DFS_EARLY_EXIT_S    = 0.100  # soft cutoff inside DFS loop (leaves room for emit/reinforce)
        self.WARCHEST_LOOK_AHEAD = 50     # horizon = min(scene_step + WARCHEST_LOOK_AHEAD, 500)
        self.MAX_CANDIDATES      = 20     # hard cap on DFS branching factor (reduce to 4 if timing spikes)
        self.ENEMY_PROD_WEIGHT   = 0.8    # penalty multiplier for enemy production (tune toward 1.0)
        self.ENEMY_ATTACK_WEIGHT = 0.2    # penalty multiplier for losses to attacking enemy in endgame

        self.player: int = 0
        self.scene_step: int = 0
        self.angular_velocity: float = 0.0
        self.planets: list[HPlanet] = []
        self.all_bodies: list[HPlanet] = []
        self.owned_planets: list[HPlanet] = []
        self.enemy_planets: list[HPlanet] = []
        self.owned_comets: list[HPlanet] = []
        self.comet_ids: set[int] = set()
        self.fleets: list[Fleet] = []
        self.orbital_info: OrbitalInfo = {}
        self.inbound_edges: ProximityGraph = {}
        self.outbound_edges: ProximityGraph = {}
        self.future_pos: FuturePos = {}
        self.destination_list: DestinationList = {}
        self._warchest_committed_ids: set[int] = set()
        # True when exactly one enemy player remains
        self.solo_endgame: bool = False

    def build_orbital_info(self, initial_planets: list[Any]) -> None:
        """Return dict mapping Planet -> OrbitalEntry if orbiting, else None."""
        cx = cy = CENTER
        ip_by_id = {ip[0]: ip for ip in initial_planets}
        self.orbital_info = {}
        for p in self.all_bodies:           # ← was self.planets
            if p.id in self.comet_ids:
                self.orbital_info[p] = None  # elliptical path; use current position as static
                continue
            r = distance((p.x, p.y), (cx, cy))
            if r + p.radius < ROTATION_RADIUS_LIMIT and p.id in ip_by_id:
                ip = ip_by_id[p.id]
                self.orbital_info[p] = OrbitalEntry(r=r, initial_angle=math.atan2(ip[3] - cy, ip[2] - cx))
            else:
                self.orbital_info[p] = None

    def build_proximity_graph(self) -> None:
        """Build directed adjacency list: dst -> [GraphEdge(src, travel_steps)].

        Directed because:
        - src departs from its current position immediately
        - dst is rotated into the future to its intercept position
        So travel from A->B and B->A may differ and one direction may exceed MAX_DISTANCE.

        future_pos stores each planet's current position (source frame).
        intercept_pos stores each planet's arrival position given a shot from the center
        """
        cx = cy = CENTER
        self.future_pos = {}
        for p in self.planets:
            orb = self.orbital_info[p]
            if orb is not None:
                a = orb.initial_angle + self.angular_velocity * (self.scene_step + 1 + self.ROTATION_LOOK_AHEAD)
                self.future_pos[p] = Pos(cx + orb.r * math.cos(a), cy + orb.r * math.sin(a))
            else:
                self.future_pos[p] = Pos(p.x, p.y)

        self.inbound_edges = {p: [] for p in self.planets}
        for src in self.planets:
            for dst in self.planets:
                if dst is src:
                    continue
                fp = self.future_pos[dst]
                travel = distance((src.x, src.y), (fp.x, fp.y))
                if travel <= self.MAX_DISTANCE:
                    self.inbound_edges[dst].append(GraphEdge(src, travel))

        # self.outbound_edges[p] = [GraphEdge(dst, travel)] — keyed by source, complement of the inbound-keyed inbound_edges.
        self.outbound_edges = {p: [] for p in self.planets}
        for dst, inbound in self.inbound_edges.items():
            for edge in inbound:
                self.outbound_edges[edge.planet].append(GraphEdge(dst, edge.travel))

    def intercept_planet(
        self,
        sx: float, sy: float, target: HPlanet, ships: int | float,
        tol: float = 1e-6, max_iters: int = 30,
    ) -> Intercept:
        """Aim angle from (sx, sy) toward where target will be when a fleet arrives.
        Returns Intercept(angle, x, y, travel).
        """
        speed = fleet_speed(ships)
        orb = self.orbital_info[target]
        if orb is None:
            tx, ty = target.x, target.y
            travel = distance((sx, sy), (tx, ty)) / speed
        else:
            cx = cy = CENTER
            # Seed: straight-line travel time to the planet's current position.
            travel = distance((sx, sy), (target.x, target.y)) / speed
            for _ in range(max_iters):
                a = orb.initial_angle + self.angular_velocity * (self.scene_step + travel - 0.5)
                new_tx, new_ty = cx + orb.r * math.cos(a), cy + orb.r * math.sin(a)
                new_travel = distance((sx, sy), (new_tx, new_ty)) / speed
                # Damp update: average old and new travel to suppress oscillation.
                new_travel = 0.5 * (travel + new_travel - 0.5)
                if abs(new_travel - travel) < tol:
                    travel = new_travel
                    break
                travel = new_travel
            else:
                # Diverged: fleet too slow to catch this planet's orbital speed.
                return Intercept(0.0, target.x, target.y, math.inf)
            # Recompute final position from converged travel so tx/ty/angle are consistent.
            a = orb.initial_angle + self.angular_velocity * (self.scene_step + travel - 0.5)
            tx, ty = cx + orb.r * math.cos(a), cy + orb.r * math.sin(a)
        angle = math.atan2(ty - sy, tx - sx)
        return Intercept(angle, tx, ty, travel)

    def first_planet_hit(self, sx: float, sy: float, angle: float, ships: int | float, source: HPlanet) -> HPlanet | None:
        """Return the first planet a fleet launched from (sx, sy) at `angle` would hit, or None.
        Returns None if the path crosses the sun before any planet is hit."""
        best = None
        best_t = float('inf')
        for planet in self.all_bodies:
            if planet is source:
                continue
            check_radius = self.COMET_RADIUS if planet.id in self.comet_ids else planet.radius
            ic = self.intercept_planet(sx, sy, planet, ships)
            dist = distance((sx, sy), (ic.x, ic.y))
            if dist < check_radius:
                half_cone = math.pi
            else:
                half_cone = math.asin(min(1.0, check_radius / dist))
            delta = abs(math.atan2(math.sin(angle - ic.angle), math.cos(angle - ic.angle)))
            if math.isfinite(ic.travel) and delta <= half_cone and ic.travel < best_t:
                best_t = ic.travel
                best = planet
        if best is None:
            return None
        # Check if the sun blocks the path to the first planet hit.
        ex, ey = sx + best_t * fleet_speed(ships) * math.cos(angle), sy + best_t * fleet_speed(ships) * math.sin(angle)
        if point_to_segment_distance((CENTER, CENTER), (sx, sy), (ex, ey)) <= SUN_RADIUS:
            return None
        return best

    def build_destination_list(self) -> None:
        """For each fleet, find the first planet it is on an interception course for.
        Populates self.destination_list: Planet -> list of Arrival.
        travel_time is continuous time in turns.
        """
        self.destination_list = defaultdict(list)
        for fleet in self.fleets:
            best = None
            best_t = float('inf')
            for planet in self.all_bodies:
                check_radius = self.COMET_RADIUS if planet.id in self.comet_ids else planet.radius
                ic = self.intercept_planet(fleet.x, fleet.y, planet, fleet.ships)
                dist = distance((fleet.x, fleet.y), (ic.x, ic.y))
                if dist < check_radius:
                    half_cone = math.pi
                else:
                    half_cone = math.asin(min(1.0, check_radius / dist))
                delta = abs(math.atan2(math.sin(fleet.angle - ic.angle),
                                       math.cos(fleet.angle - ic.angle)))
                if math.isfinite(ic.travel) and delta <= half_cone and ic.travel < best_t:
                    best_t = ic.travel
                    best = (planet, ic.travel, ic.x, ic.y)
            if best is not None:
                planet, travel, px, py = best
                if planet.id in self.comet_ids:
                    continue  # comet positions are unpredictable; don't track as destinations
                self.destination_list[planet].append(
                    Arrival(fleet.owner, fleet.ships, travel, fleet.x, fleet.y, px, py)
                )

    def build_reinforcement_targets(self) -> None:
        front_line = {
            p for p in self.owned_planets
            if any(e.planet.owner != self.player for e in self.inbound_edges[p])
            or any(e.planet.owner != self.player for e in self.outbound_edges[p])
        }

        # BFS hop-distance from every owned node to nearest frontline planet,
        # traversing only owned-planet edges (frontline nodes are sinks, not sources).
        hops_to_front: dict[HPlanet, int] = {p: 0 for p in front_line}
        queue: list[HPlanet] = list(front_line)
        head = 0
        while head < len(queue):
            node = queue[head]; head += 1
            for edge in self.inbound_edges[node]:
                if edge.planet.owner != self.player or edge.planet in hops_to_front:
                    continue
                hops_to_front[edge.planet] = hops_to_front[node] + 1
                queue.append(edge.planet)

        for p in self.owned_planets:
            p.reinforcement_target = None
            if p in front_line:
                continue

            direct_front = [
                e.planet for e in self.outbound_edges[p]
                if e.planet in front_line
            ]
            if direct_front:
                p.reinforcement_target = min(direct_front, key=lambda d: d.ships)
                continue

            # No direct edge to a frontline planet: pick the direct neighbor with
            # fewest hops to the front, breaking ties by fewest ships at destination.
            reachable = [
                e.planet for e in self.outbound_edges[p]
                if e.planet.owner == self.player and e.planet not in front_line and e.planet in hops_to_front
            ]
            if reachable:
                p.reinforcement_target = min(reachable, key=lambda d: (hops_to_front[d], d.ships))

    def send_reinforcements(self) -> FleetOrders:
        """ Allows sending by an intermediate planet if in the way. """
        orders: FleetOrders = []
        for p in self.owned_planets:
            if p.reinforcement_target is None:
                continue
            if p.ships < (self.REINFORCEMENT_SIZE + self.GARRISON_SIZE):
                continue
            has_enemy_incoming = any(
                e.planet.owner != self.player
                for e in self.inbound_edges.get(p, []) )
            if has_enemy_incoming:
                continue
            target = p.reinforcement_target
            ships = int(p.ships - self.GARRISON_SIZE)
            ic = self.intercept_planet(p.x, p.y, target, ships)
            if not math.isfinite(ic.travel):
                continue
            orders.append(FleetOrder(p.id, ic.angle, ships))
        return orders

    # ------------------------------------------------------------------
    # Warchest methods

    def warchest_advance(self, state: WarchestState, to_turn: int) -> None:
        for t in range(state.turn + 1, to_turn + 1):
            # 1. Production first (rules: production happens before fleet movement/combat)
            for pid, owner in state.ownership.items():
                if owner != -1:
                    state.garrison[pid] = state.garrison.get(pid, 0) + state.production.get(pid, 0)

            # 2. Collect arrivals at this turn
            arrivals_by_dest: dict[int, list[OwnerShips]] = defaultdict(list)
            remaining_friendly = []
            for f in state.friendly_fleets:
                if f.arrival_turn == t:
                    arrivals_by_dest[f.destination_id].append(OwnerShips(f.owner, f.fleet_size))
                else:
                    remaining_friendly.append(f)
            state.friendly_fleets = remaining_friendly

            remaining_enemy = []
            for f in state.enemy_fleets:
                if f.arrival_turn == t:
                    arrivals_by_dest[f.destination_id].append(OwnerShips(f.owner, f.fleet_size))
                else:
                    remaining_enemy.append(f)
            state.enemy_fleets = remaining_enemy

            # 3. Resolve battles
            for dest_id, arrivals in arrivals_by_dest.items():
                self.warchest_resolve_battle(state, dest_id, arrivals)

        state.turn = to_turn

    def warchest_resolve_battle(
        self,
        state: WarchestState,
        planet_id: int,
        arrivals: list[OwnerShips],
    ) -> None:
        # Stage 1: arriving fleets fight each other (garrison is separate)
        owner_ships: dict[int, float] = defaultdict(float)
        for a in arrivals:
            owner_ships[a.owner] += a.ships

        if not owner_ships:
            return

        sorted_owners = sorted(owner_ships.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_owners) == 1:
            survivor_owner, survivor_ships = sorted_owners[0]
        else:
            top_owner, top_ships = sorted_owners[0]
            second_ships          = sorted_owners[1][1]
            survivor_ships        = top_ships - second_ships
            survivor_owner        = top_owner if survivor_ships > 0 else -1

        if survivor_ships <= 0:
            return  # mutual annihilation; garrison unchanged

        # Stage 2: survivor fights the garrison
        cur_owner  = state.ownership.get(planet_id, -1)
        cur_ships  = state.garrison.get(planet_id, 0.0)

        if survivor_owner == cur_owner:
            # Same owner: reinforce
            state.garrison[planet_id] = cur_ships + survivor_ships
        else:
            net = cur_ships - survivor_ships
            if net > 0:
                state.garrison[planet_id] = net          # defender holds
            elif net < 0:
                state.ownership[planet_id] = survivor_owner
                state.garrison[planet_id]  = -net         # attacker captures
            else:
                state.garrison[planet_id] = 0.0           # exact tie: defender keeps with 0

    def warchest_initial_state(self) -> WarchestState:
        garrison   = {p.id: float(p.ships)      for p in self.planets}
        production = {p.id: float(p.production) for p in self.planets}
        ownership  = {p.id: p.owner             for p in self.planets}

        friendly_fleets = []
        enemy_fleets    = []
        # Track which planets already have a friendly capture fleet assigned, so only the
        # first (earliest-arriving) friendly fleet to a given enemy/neutral planet gets
        # production_id credit. Subsequent friendly fleets to the same planet arrive after
        # the capture and are reinforcements — they must not claim production_id or
        # warchest_score will double-count the planet's production income.
        friendly_capture_claimed: set[int] = set()

        for dest_planet, arrivals in self.destination_list.items():
            if dest_planet.id in self.comet_ids:
                continue  # ignore fleets targeting comets
            # Sort arrivals by travel time so the earliest-arriving friendly fleet is
            # processed first and gets the production_id claim.
            for arr in sorted(arrivals, key=lambda a: a.travel_time):
                arrival  = self.scene_step + math.ceil(arr.travel_time)
                is_cap   = (arr.owner != dest_planet.owner)
                # Project target garrison to arrival time so garrison_on_arrival is accurate.
                if is_cap:
                    prod_during_travel = dest_planet.production * math.ceil(arr.travel_time) if dest_planet.owner != -1 else 0.0
                    garrison_at_arrival = dest_planet.ships + prod_during_travel
                    gar_on_arr = max(0.0, arr.ships - garrison_at_arrival)
                else:
                    gar_on_arr = arr.ships
                # Only the first friendly fleet to a capture target claims the production bonus.
                # Without this guard, two friendly fleets to the same planet each get
                # production_id set, and warchest_score counts that planet's income twice.
                if is_cap and arr.owner == self.player:
                    if dest_planet.id in friendly_capture_claimed:
                        is_cap = False   # treat as reinforcement for scoring purposes
                        gar_on_arr = arr.ships
                    else:
                        friendly_capture_claimed.add(dest_planet.id)
                fleet = WarchestFleet(
                    owner=arr.owner,
                    destination_id=dest_planet.id,
                    fleet_size=arr.ships,
                    garrison_on_arrival=gar_on_arr,
                    arrival_turn=arrival,
                    is_capture=is_cap,
                    production_id=dest_planet.id if is_cap else -1,
                )
                if arr.owner == self.player:
                    friendly_fleets.append(fleet)
                else:
                    enemy_fleets.append(fleet)

        return WarchestState(
            turn=self.scene_step,
            garrison=garrison,
            production=production,
            ownership=ownership,
            friendly_fleets=friendly_fleets,
            enemy_fleets=enemy_fleets,
            committed_ids=set(),
        )

    def warchest_source_would_be_lost(
        self,
        state: WarchestState,
        src_id: int,
        ships_to_send: float,
        until_turn: int,
    ) -> bool:
        """Return True if the source planet will fall to enemy before ships are launched."""
        inbound = [
            (f.fleet_size, f.arrival_turn)
            for f in state.enemy_fleets
            if f.destination_id == src_id and f.arrival_turn <= until_turn
        ]
        if not inbound:
            return False
        garrison = state.garrison.get(src_id, 0.0) - ships_to_send
        prod     = state.production.get(src_id, 0.0)
        prev_t   = state.turn
        # Group arrivals by turn so simultaneous enemy fleets are resolved together.
        arrivals_by_turn: dict[int, float] = defaultdict(float)
        for fleet_size, arrival in inbound:
            arrivals_by_turn[arrival] += fleet_size
        for arrival in sorted(arrivals_by_turn):
            garrison += prod * (arrival - prev_t)
            garrison -= arrivals_by_turn[arrival]
            prev_t    = arrival
            if garrison < 0:
                return True
        return False

    def warchest_assign_fleet(
        self,
        state: WarchestState,
        target: HPlanet,
        horizon: int,
    ) -> dict[int, Assignment]:
        candidates = sorted(
            [edge
             for edge in self.inbound_edges.get(target, [])
             if state.ownership.get(edge.planet.id) == self.player
             and edge.planet.id not in state.committed_ids
             and state.garrison.get(edge.planet.id, 0) > 0],
            key=lambda e: e.travel,   # closest first
        )

        if not candidates:
            return {}

        # Phase 1: find minimum source set (closest first) that overcomes the target garrison
        # projected forward to arrival time (enemy/neutral planets produce during travel).
        if state.ownership.get(target.id) == self.player:
            # Defense: simulate each enemy arrival sequentially, crediting production between
            # arrivals. Summing all enemy ships and projecting the garrison only to the earliest
            # arrival is wrong: it ignores that the defender keeps producing between sequential
            # enemy arrivals, systematically overstating the threat and over-committing defenders.
            enemy_inbound = sorted(
                [f for f in state.enemy_fleets if f.destination_id == target.id],
                key=lambda f: f.arrival_turn,
            )
            cur_garrison = state.garrison.get(target.id, 0.0)
            prod_rate = state.production.get(target.id, 0.0)
            prev_t = state.turn
            for f in enemy_inbound:
                cur_garrison += prod_rate * (f.arrival_turn - prev_t)
                cur_garrison = max(0.0, cur_garrison - f.fleet_size)
                prev_t = f.arrival_turn
            # arrival_garrison is the net deficit after all enemy arrivals (positive = we lost ships).
            # If cur_garrison > 0, the planet survived; no reinforcement needed.
            if cur_garrison > 0:
                return {}  # projected garrison already holds; no reinforcement needed
            arrival_garrison = -cur_garrison + 1  # ships needed to recover the planet
        else:
            # Offense: project target garrison to earliest possible arrival (closest source travel).
            # This is a conservative lower-bound threshold: Phase 1 accumulates sources until the
            # fleet exceeds this estimate. After Phase 1 we re-check against the actual latest_arrival
            # garrison, because adding more sources pushes latest_arrival later and the target
            # accumulates more production during the extra turns.
            #
            # NOTE: inbound_edges stores raw distance, not travel turns. Compute the first
            # candidate's actual intercept time to get a correct turn estimate.
            first_src = candidates[0].planet
            first_ships = int(state.garrison.get(first_src.id, 1)) or 1
            orb0 = self.orbital_info.get(first_src)
            if orb0 is not None:
                _cx = _cy = CENTER
                _a0 = orb0.initial_angle + self.angular_velocity * state.turn
                _sx0, _sy0 = _cx + orb0.r * math.cos(_a0), _cy + orb0.r * math.sin(_a0)
            else:
                _sx0, _sy0 = first_src.x, first_src.y
            ic0 = self.intercept_planet(_sx0, _sy0, target, first_ships)
            min_travel = ic0.travel if math.isfinite(ic0.travel) else candidates[0].travel
            prod_rate = state.production.get(target.id, 0.0) if state.ownership.get(target.id) != -1 else 0.0
            arrival_garrison = state.garrison.get(target.id, 0.0) + prod_rate * math.ceil(min_travel)

        selected: list[tuple[HPlanet, float, float, int]] = []   # (src, ships_now, angle, arrival_turn_if_launched_now)
        total_ships  = 0.0
        latest_arrival = 0

        for edge in candidates:
            src = edge.planet
            ships = int(state.garrison.get(src.id, 0))
            if ships <= 0:
                continue
            # Use orbital position at state.turn so Phase 1 and Phase 2 (launch_turn=state.turn+delay)
            # agree on source position when delay=0. Without this, arrival computed here differs from
            # arrival2 in Phase 2, causing Phase 2 to discard the source (arrival2 != latest_arrival).
            orb_src = self.orbital_info.get(src)
            if orb_src is not None:
                cx = cy = CENTER
                a_now = orb_src.initial_angle + self.angular_velocity * state.turn
                sx1, sy1 = cx + orb_src.r * math.cos(a_now), cy + orb_src.r * math.sin(a_now)
            else:
                sx1, sy1 = src.x, src.y
            ic = self.intercept_planet(sx1, sy1, target, ships)
            if not math.isfinite(ic.travel):
                continue
            if self.first_planet_hit(sx1, sy1, ic.angle, ships, src) is not target:
                continue
            arrival = state.turn + math.ceil(ic.travel)
            if arrival > horizon:
                continue
            selected.append((src, ships, ic.angle, arrival))
            total_ships    += ships
            latest_arrival  = max(latest_arrival, arrival)
            if total_ships > arrival_garrison:
                break

        if total_ships <= arrival_garrison:
            return {}  # can't win even with all viable sources

        # Re-check against the actual synchronized arrival garrison. When multiple sources are
        # needed, latest_arrival is the slowest source's arrival turn, which can be several turns
        # later than min_travel. The target produces ships during those extra turns.
        if state.ownership.get(target.id) != self.player:
            actual_prod = state.production.get(target.id, 0.0) if state.ownership.get(target.id) != -1 else 0.0
            actual_arrival_garrison = (state.garrison.get(target.id, 0.0) +
                                       actual_prod * (latest_arrival - state.turn))
            if total_ships <= actual_arrival_garrison:
                return {}  # fleet sufficient at min_travel but not at actual latest_arrival

        # Second-enemy-arrival guard: on neutral targets, if enemy fleets arrive at or within
        # 1 turn after our fleet, we'd be fighting the battle winner rather than the original
        # garrison. The +1 tolerance covers the engine's swept-collision handling which can
        # resolve arrivals off by one turn.
        if state.ownership.get(target.id) == -1:
            enemy_arrivals = [f.arrival_turn for f in state.enemy_fleets
                              if f.destination_id == target.id]
            if enemy_arrivals and latest_arrival <= max(enemy_arrivals) + 1:
                return {}

        # Phase 2: synchronize — each source launches so it arrives at latest_arrival.
        # Sources with shorter travel times get a delay; during the delay they accumulate
        # production but may also take enemy fleet damage — simulate both.
        assignment: dict[int, Assignment] = {}

        for src, ships_now, _, arrival_now in selected:
            delay       = latest_arrival - arrival_now   # turns this source must wait
            launch_turn = state.turn + delay

            # Projected garrison: simulate production and enemy arrivals at source during delay.
            proj_garrison = float(state.garrison.get(src.id, 0))
            prod_src = state.production.get(src.id, 0.0)
            prev_t = state.turn
            inbound_at_src = sorted(
                [(f.fleet_size, f.arrival_turn) for f in state.enemy_fleets
                 if f.destination_id == src.id and state.turn < f.arrival_turn <= launch_turn],
                key=lambda x: x[1]
            )
            for fleet_sz, arr_t in inbound_at_src:
                proj_garrison += prod_src * (arr_t - prev_t)
                proj_garrison = max(0.0, proj_garrison - fleet_sz)  # damage; capture caught by Phase 3
                prev_t = arr_t
            proj_garrison += prod_src * (launch_turn - prev_t)
            projected = max(1, int(proj_garrison))

            # Recompute intercept from source position at launch_turn (handles orbiting planets).
            orb = self.orbital_info.get(src)
            if orb is not None:
                cx = cy = CENTER
                a  = orb.initial_angle + self.angular_velocity * launch_turn
                sx, sy = cx + orb.r * math.cos(a), cy + orb.r * math.sin(a)
            else:
                sx, sy = src.x, src.y

            ic2 = self.intercept_planet(sx, sy, target, projected)
            if not math.isfinite(ic2.travel):
                continue
            # Re-validate path from the delayed launch position (source may have rotated).
            if self.first_planet_hit(sx, sy, ic2.angle, projected, src) is not target:
                continue
            arrival2 = launch_turn + math.ceil(ic2.travel)
            # Enforce exact synchronization: all sources must share the same arrival turn.
            # A 1-turn drift would mean the first source fights the garrison alone before
            # the second arrives, violating the synchronized-arrival invariant (§6.1).
            if arrival2 != latest_arrival:
                continue
            assignment[src.id] = Assignment(projected, launch_turn, arrival2)

        if not assignment:
            return {}

        # Phase 3: safety check — ensure source survives until its launch_turn.
        for src_id, a in list(assignment.items()):
            if self.warchest_source_would_be_lost(state, src_id, a.ships, a.launch_turn):
                del assignment[src_id]

        total_sync = sum(a.ships for a in assignment.values())
        if total_sync <= arrival_garrison:
            return {}  # safety filter removed too many sources

        # Phase 4: trim excess — keep half the surplus on the first-added (highest-delay) source
        # so it retains some garrison while still sending enough to win.
        # selected is ordered closest-first (ascending arrival_now = descending delay), so the
        # first key inserted into assignment has the largest delay and the most accumulated ships.
        # Use ceiling division so odd excess values are fully resolved (excess=1 -> trim 1).
        excess = total_sync - arrival_garrison - 1
        if excess > 0:
            first_id = list(assignment.keys())[0]
            a = assignment[first_id]
            trimmed = max(1, int(a.ships - (excess + 1) // 2))
            assignment[first_id] = Assignment(trimmed, a.launch_turn, a.arrival_turn)

        return assignment

    def warchest_execute(
        self,
        state: WarchestState,
        target: HPlanet,
        assignment: dict[int, Assignment],
    ) -> WarchestState:
        total_ships    = 0.0
        latest_arrival = 0

        for source_id, a in sorted(
            assignment.items(), key=lambda x: x[1].launch_turn
        ):
            # Advance to one turn before launch so production for launch_turn hasn't run yet.
            if a.launch_turn - 1 > state.turn:
                self.warchest_advance(state, a.launch_turn - 1)
            # Deduct ships before this turn's production runs.
            state.garrison[source_id] = max(0.0, state.garrison.get(source_id, 0.0) - a.ships)
            state.committed_ids.add(source_id)
            # Now advance through launch_turn: applies production, resolves any arrivals.
            self.warchest_advance(state, a.launch_turn)
            total_ships    += a.ships
            latest_arrival  = max(latest_arrival, a.arrival_turn)

        # All ships arrive simultaneously at latest_arrival.
        # Project target garrison forward to arrival time — enemy planets produce while we travel.
        is_capture = state.ownership.get(target.id) != self.player
        target_garrison_now = state.garrison.get(target.id, 0.0)
        if is_capture:
            turns_to_arrival = latest_arrival - state.turn
            target_prod = state.production.get(target.id, 0.0) if state.ownership.get(target.id) != -1 else 0.0
            target_garrison_at_arrival = target_garrison_now + target_prod * turns_to_arrival
            garrison_on_arrival = max(0.0, total_ships - target_garrison_at_arrival)
            if garrison_on_arrival <= 0:
                # Attack fails: ships were deducted from sources but don't capture anything.
                # Advance to latest_arrival before returning so warchest_score sees the correct
                # remaining horizon. Without this, state.turn is stuck at the last launch_turn,
                # inflating `remaining = horizon - state.turn` and making failed branches score
                # higher than branches that successfully advance to the arrival turn.
                self.warchest_advance(state, latest_arrival)
                return state
        else:
            garrison_on_arrival = total_ships

        state.friendly_fleets.append(WarchestFleet(
            owner=self.player,
            destination_id=target.id,
            fleet_size=total_ships,
            garrison_on_arrival=garrison_on_arrival,
            arrival_turn=latest_arrival,
            is_capture=is_capture,
            production_id=target.id if is_capture else -1,
        ))

        # Advance to latest_arrival so subsequent DFS levels can use the capture result
        # (ownership updated, production credited) as a source for follow-on attacks.
        # This is correct even for the scoring function: warchest_score reads live ownership
        # and won't double-penalise a planet that has already been captured.
        self.warchest_advance(state, latest_arrival)

        return state

    def warchest_score(self, state: WarchestState, horizon: int) -> float:
        remaining = max(0, horizon - state.turn)
        total = 0.0

        for pid, owner in state.ownership.items():
            prod     = state.production.get(pid, 0)
            garrison = state.garrison.get(pid, 0)
            if owner == self.player:
                total += garrison + prod * remaining
            elif owner != -1:
                total -= (garrison + prod * remaining) * self.ENEMY_PROD_WEIGHT

        for f in state.friendly_fleets:
            total += f.garrison_on_arrival
            if f.is_capture and f.production_id >= 0:
                prod = state.production.get(f.production_id, 0)
                total += prod * max(0, horizon - f.arrival_turn)

        # Penalize known enemy fleets inbound to our planets (static approximation).
        # Without this, the DFS cannot distinguish states where our planets are under attack.
        # Only apply to fleets that haven't arrived yet (arrival_turn > state.turn) and whose
        # destination is still owned by us. An enemy fleet whose destination we captured during
        # the DFS sequence will find a friendly planet on arrival and reinforce us — don't penalize.
        for f in state.enemy_fleets:
            if f.arrival_turn <= state.turn:
                continue  # already resolved by warchest_advance; don't double-count
            if state.ownership.get(f.destination_id) != self.player:
                continue  # not our planet at this point in the simulation
            garrison = state.garrison.get(f.destination_id, 0.0)
            prod     = state.production.get(f.destination_id, 0.0)
            turns_until_arrival = max(0, f.arrival_turn - state.turn)
            garrison_at_arrival = garrison + prod * turns_until_arrival
            remaining_after = max(0, horizon - f.arrival_turn)
            if f.fleet_size >= garrison_at_arrival:
                # Enemy likely captures: penalise the production income we'll lose on both sides.
                total -= prod * remaining_after * (1.0 + self.ENEMY_PROD_WEIGHT)
            else:
                # Enemy damages but doesn't capture: penalise ships lost.
                total -= f.fleet_size

        return total

    def warchest_upper_bound(
        self,
        state: WarchestState,
        remaining: list[HPlanet],
        horizon: int,
    ) -> float:
        bound = self.warchest_score(state, horizon)

        # Available ships from uncommitted owned planets
        ship_budget = sum(
            state.garrison.get(pid, 0)
            for pid, owner in state.ownership.items()
            if owner == self.player and pid not in state.committed_ids
        )

        for p in sorted(remaining, key=lambda p: p.production, reverse=True):
            if state.ownership.get(p.id) == self.player:
                continue
            capture_cost = state.garrison.get(p.id, 0) + 1
            if capture_cost > ship_budget:
                continue
            # Production benefit starts at capture, not now: use warchest_earliest_capture
            # which computes the true turn-based arrival time (inbound_edges holds distance, not turns).
            earliest_turn = self.warchest_earliest_capture(state, p, horizon)
            production_turns = max(0, horizon - earliest_turn) if math.isfinite(earliest_turn) else 0
            prod_multiplier = (1.0 + self.ENEMY_PROD_WEIGHT) if self.solo_endgame else 1.0
            net_gain = p.production * production_turns * prod_multiplier - capture_cost
            if net_gain > 0:
                bound      += net_gain
                ship_budget -= capture_cost

        return bound

    def warchest_earliest_capture(
        self,
        state: WarchestState,
        target: HPlanet,
        horizon: int,
    ) -> float:
        """Earliest turn a single source, or the two closest combined sources, can capture.

        Single-source check first. If no single source has enough ships, a two-source
        fallback checks whether any pair of nearby sources can together beat the garrison.
        Without the two-source check, heavily-garrisoned enemy planets are silently excluded
        from the candidate list even though `warchest_assign_fleet` would win them with two sources.
        """
        garrison_size = state.garrison.get(target.id, target.ships)
        best = math.inf
        viable: list[tuple[int, float]] = []  # (arrival_turn, ships)

        for edge in self.inbound_edges.get(target, []):
            src = edge.planet
            if state.ownership.get(src.id) != self.player:
                continue
            ships = state.garrison.get(src.id, 0)
            if ships <= 0:
                continue
            ic = self.intercept_planet(src.x, src.y, target, int(ships))
            if not math.isfinite(ic.travel):
                continue
            arrival = state.turn + math.ceil(ic.travel)
            if arrival > horizon:
                continue
            viable.append((arrival, ships))
            if ships > garrison_size:
                best = min(best, arrival)

        # Multi-source fallback: find the earliest synchronized arrival where sources
        # accumulated in arrival order together exceed the garrison.
        # NOTE: accumulate a running SUM, not a running max. max_ships_seen would only
        # find pairs where one large prior source + current > garrison, silently missing
        # 3+-source combinations (e.g. three sources of 40 vs garrison 100).
        if best == math.inf and len(viable) >= 2:
            viable.sort(key=lambda x: x[0])
            total_ships_seen = 0.0
            for arr_j, ships_j in viable:
                total_ships_seen += ships_j
                if total_ships_seen > garrison_size:
                    best = min(best, arr_j)
                    break

        return best

    def warchest_candidates(
        self,
        initial: WarchestState,
        horizon: int,
    ) -> list[HPlanet]:
        defense = []
        for p in self.owned_planets:
            enemy_inbound = [f for f in initial.enemy_fleets if f.destination_id == p.id]
            if not enemy_inbound:
                continue
            # Quick simulation: will we lose this planet?
            trial = warchest_state_copy(initial)
            last_enemy_arrival = max(f.arrival_turn for f in enemy_inbound)
            self.warchest_advance(trial, min(last_enemy_arrival, horizon))
            if trial.ownership.get(p.id) != self.player:
                defense.append(p)

        # Planet IDs already committed to capture by an in-flight friendly fleet.
        # These are handled; re-targeting them wastes DFS slots and can emit redundant moves.
        already_captured_ids = {f.destination_id for f in initial.friendly_fleets if f.is_capture}

        offense = []
        for p in self.planets:
            if p.owner == self.player or p.id in self.comet_ids:
                continue
            if p.id in already_captured_ids:
                continue
            if not self.inbound_edges.get(p):
                continue
            ct = self.warchest_earliest_capture(initial, p, horizon)
            if not math.isfinite(ct):
                continue
            prod_multiplier = (1.0 + self.ENEMY_PROD_WEIGHT) if self.solo_endgame else 1.0
            ship_cost = self.ENEMY_ATTACK_WEIGHT * p.ships if self.solo_endgame else p.ships
            gain = p.production * (horizon - ct) * prod_multiplier - ship_cost
            if gain > 0:
                offense.append((p, gain))

        offense.sort(key=lambda x: x[1], reverse=True)
        return (defense + [p for p, _ in offense])[:self.MAX_CANDIDATES]

    def warchest_emit_moves(self, sequence) -> FleetOrders:
        moves: FleetOrders = []
        planet_by_id = {p.id: p for p in self.planets}
        for target_planet, assignment in sequence:
            # Collect sources that launch this turn vs deferred sources.
            immediate: list[tuple[int, Assignment]] = []
            deferred:  list[tuple[int, Assignment]] = []
            for source_id, a in assignment.items():
                (immediate if a.launch_turn == self.scene_step else deferred).append((source_id, a))

            # If there are deferred sources, the plan needs them to complete the attack.
            # Emitting only the immediate subset would send an under-strength fleet that
            # arrives alone and loses, while also marking the target as in-flight so future
            # candidates exclude it — preventing the deferred sources from ever launching.
            # Guard: only emit immediate sources if they are sufficient on their own.
            if deferred:
                immediate_ships = sum(a.ships for _, a in immediate)
                # Garrison at the arrival turn of the immediate sources (all share arrival_turn).
                if immediate:
                    arr_turn = immediate[0][1].arrival_turn
                    prod_rate = target_planet.production if target_planet.owner != -1 else 0.0
                    garrison_at_arrival = target_planet.ships + prod_rate * (arr_turn - self.scene_step)
                    if immediate_ships <= garrison_at_arrival:
                        continue  # immediate subset can't win alone; skip this turn, re-plan next

            for source_id, a in immediate:
                src = planet_by_id.get(source_id)
                if src is None:
                    continue
                ic = self.intercept_planet(src.x, src.y, target_planet, a.ships)
                if not math.isfinite(ic.travel):
                    continue
                if self.first_planet_hit(src.x, src.y, ic.angle, a.ships, src) is not target_planet:
                    continue
                moves.append(FleetOrder(source_id, ic.angle, a.ships))
        return moves

    def warchest_dfs(self, state, remaining, sequence, horizon, t0, best):
        score = self.warchest_score(state, horizon)
        if score > best[0]:
            best[0] = score
            best[1] = list(sequence)

        if time.perf_counter() - t0 > self.DFS_EARLY_EXIT_S:
            return  # soft cutoff; return best found so far

        if not remaining:
            return

        if self.warchest_upper_bound(state, remaining, horizon) <= best[0]:
            return  # prune: branch cannot improve best

        for i, planet in enumerate(remaining):
            assignment = self.warchest_assign_fleet(state, planet, horizon)
            if not assignment:
                continue

            next_state = self.warchest_execute(warchest_state_copy(state), planet, assignment)
            self.warchest_dfs(
                next_state,
                remaining[:i] + remaining[i+1:],
                sequence + [(planet, assignment)],
                horizon, t0, best,
            )

    def run_unified_search(self, horizon: int) -> FleetOrders:
        self._warchest_committed_ids = set()
        enemy_players = {p.owner for p in self.enemy_planets}
        self.solo_endgame = (len(enemy_players) == 1)
        initial    = self.warchest_initial_state()

        # Pre-advance past any in-flight fleet arrivals that have already happened by the
        # current turn so the DFS baseline and every branch are scored in the same
        # post-arrival reference frame. Without this, warchest_score(initial) is inflated
        # by unresolved fleet bonuses, causing the DFS to always prefer the empty sequence
        # over any profitable plan.
        # Cap at scene_step: future arrivals (arrival_turn > scene_step) are already
        # represented as pending WarchestFleet entries and must not shift state.turn forward,
        # which would make warchest_assign_fleet set launch_turn > scene_step for all
        # sources and cause warchest_emit_moves to suppress every move.
        past_arrivals = [f.arrival_turn for f in initial.friendly_fleets + initial.enemy_fleets
                         if f.arrival_turn <= self.scene_step]
        if past_arrivals:
            self.warchest_advance(initial, max(past_arrivals))

        candidates = self.warchest_candidates(initial, horizon)

        best = [self.warchest_score(initial, horizon), []]
        t0   = time.perf_counter()
        self.warchest_dfs(initial, candidates, [], horizon, t0, best)

        # Only mark sources that launch this turn as committed; deferred sources still sit at home
        # and must remain available to send_reinforcements on turns when they don't launch.
        self._warchest_committed_ids = {
            src_id
            for _, assignment in best[1]
            for src_id, a in assignment.items()
            if a.launch_turn == self.scene_step
        }
        return self.warchest_emit_moves(best[1])

    def drain_comets(self) -> FleetOrders:
        orders: FleetOrders = []
        for comet in self.owned_comets:
            if comet.ships <= 0:
                continue
            # Find nearest non-comet planet within reach
            best_dst, best_travel = None, math.inf
            for p in self.planets:
                ic = self.intercept_planet(comet.x, comet.y, p, int(comet.ships))
                if math.isfinite(ic.travel) and ic.travel < best_travel:
                    best_travel = ic.travel
                    best_dst = p
            if best_dst is None:
                continue
            ships = int(comet.ships)
            ic = self.intercept_planet(comet.x, comet.y, best_dst, ships)
            if not math.isfinite(ic.travel):
                continue
            if self.first_planet_hit(comet.x, comet.y, ic.angle, ships, comet) is not best_dst:
                continue
            orders.append(FleetOrder(comet.id, ic.angle, ships))
        return orders

    # ------------------------------------------------------------------
    # viz

    @dataclass(slots=True)
    class ArrowEndpoints:
        sx: float
        sy: float
        ex: float
        ey: float

    @staticmethod
    def viz_arrow_endpoints(
        px: float, py: float, tx: float, ty: float,
        origin_radius: float, target_radius: float,
    ) -> ArrowEndpoints:
        dx, dy = tx - px, ty - py
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return ArrowEndpoints(px, py, tx, ty)
        ux, uy = dx / d, dy / d
        return ArrowEndpoints(
            px + ux * origin_radius, py + uy * origin_radius,
            tx - ux * target_radius, ty - uy * target_radius,
        )

    def viz_proximity_graph(self, show_inbound: bool = True) -> None:
        """Draw directed edges from each planet's current position to the target's future_pos."""
        edge_map = self.inbound_edges if show_inbound else self.outbound_edges
        direction = 'inbound' if show_inbound else 'outbound'

        on_screen: list[str] = []
        for p in sorted(self.planets, key=lambda p: p.id):
            if p.owner == -1:
                continue
            neighbors = edge_map.get(p, [])
            if not neighbors:
                continue
            px, py = p.x, p.y
            neighbor_strs: list[str] = []
            for edge in neighbors:
                fp = self.future_pos[edge.planet]
                ae = self.viz_arrow_endpoints(px, py, fp.x, fp.y, p.radius, edge.planet.radius)
                color = '#ff8844' if p.owner == 1 else '#22aaff'
                viz.add_arrow(self.scene_step, ae.sx, ae.sy, ae.ex, ae.ey, color=color, width=1, length_frac=1.0, head_size=5)
                neighbor_strs.append(f'P{edge.planet.id}({edge.travel:.0f})')
            on_screen.append(f'  P{p.id}: [{", ".join(neighbor_strs)}]')

        edge_count = sum(len(v) for v in edge_map.values())
        header = f'{direction}_edges:'
        viz.add_text(self.scene_step, header + '\n' + '\n'.join(on_screen))

    def viz_reinforcement_targets(self) -> None:
        """Draw reinforcement_target arrows from each owned planet's current pos to target's current pos."""
        lines: list[str] = []
        for p in sorted(self.owned_planets, key=lambda p: p.id):
            if p.reinforcement_target is None:
                continue
            px, py = p.x, p.y
            tx, ty = p.reinforcement_target.x, p.reinforcement_target.y
            ae = self.viz_arrow_endpoints(px, py, tx, ty, p.radius, p.reinforcement_target.radius)
            viz.add_arrow(self.scene_step, ae.sx, ae.sy, ae.ex, ae.ey, color='#44ff88', width=1, length_frac=1.0, head_size=5)
            lines.append(f'  P{p.id} -> P{p.reinforcement_target.id}')

        if lines:
            viz.add_text(self.scene_step, f'reinforcement_targets ({len(lines)}):\n' + '\n'.join(lines))
        else:
            viz.add_text(self.scene_step, 'reinforcement_targets: none')

    def viz_destination_list(self) -> None:
        """Draw a line from each fleet to its destination planet's arrival position."""
        lines = []
        for planet, arrivals in self.destination_list.items():
            for arr in arrivals:
                color = '#44ff88' if arr.owner == 0 else '#ff8844'
                viz.add_line(self.scene_step, arr.src_x, arr.src_y, arr.arrival_x, arr.arrival_y, color=color, width=1)
                lines.append(f' {arr.owner} ({arr.ships}) -> P{planet.id}({planet.ships}) t={round(arr.travel_time)} v={fleet_speed(arr.ships):.2f}')
        if lines:
            viz.add_text(self.scene_step, 'dest_list:\n' + '\n'.join(lines))

    # ------------------------------------------------------------------

    def main(self, obs: dict[str, Any]) -> list[Any]:
        viz.record(obs)
        _t0 = time.perf_counter()

        self.player           = obs['player']
        self.scene_step       = obs['step'] - 1
        self.angular_velocity = obs['angular_velocity']

        comet_ids             = set(obs['comet_planet_ids'])
        self.comet_ids        = comet_ids
        self.all_bodies       = [HPlanet(*p) for p in obs['planets']]
        self.planets          = [p for p in self.all_bodies if p.id not in comet_ids]
        self.owned_comets     = [p for p in self.all_bodies
                                 if p.id in comet_ids and p.owner == self.player]
        self.owned_planets    = [p for p in self.planets if p.owner == self.player]
        self.enemy_planets    = [p for p in self.planets
                                 if p.owner not in (-1, self.player)]
        self.fleets           = [Fleet(*f) for f in obs['fleets']]

        if not self.enemy_planets:
            return []

        self.build_orbital_info(obs['initial_planets'])
        self.build_proximity_graph()
        self.build_destination_list()
        self.build_reinforcement_targets()

        horizon = min(self.scene_step + self.WARCHEST_LOOK_AHEAD, 500)
        moves   = self.run_unified_search(horizon)

        reinforcement_orders = self.send_reinforcements()
        reinforcement_orders = [o for o in reinforcement_orders
                                if o.planet_id not in self._warchest_committed_ids]
        moves.extend(reinforcement_orders)

        comet_orders = self.drain_comets()
        moves.extend(comet_orders)

        elapsed_ms = (time.perf_counter() - _t0) * 1000
        viz.add_text(self.scene_step, f'Hellburner ms: {elapsed_ms:.2f}ms')

        # DO NOT DELETE:
        #self.viz_proximity_graph(False) # inbound: True, outbound: False.
        #self.viz_reinforcement_targets()
        #self.viz_destination_list()
        #viz.add_text(self.scene_step, 'moves: ' + str(moves))

        return moves


def agent(obs: dict[str, Any]) -> list[Any]:
    _agent = Hellburner()
    try:
        return [[o.planet_id, o.angle, o.ships] for o in _agent.main(obs)]
    except Exception:
        import traceback
        tb = traceback.format_exc()
        try:
            viz.add_text(_agent.scene_step, tb)
        except Exception:
            pass
        return []


def make_agent(**overrides):
    """Return an agent function with the given constants overridden.

    Example:
        challenger = make_agent(REINFORCEMENT_SIZE=12, GARRISON_SIZE=8)
    """
    def _agent(obs: dict[str, Any]) -> list[Any]:
        h = Hellburner()
        for k, v in overrides.items():
            setattr(h, k, v)
        try:
            return [[o.planet_id, o.angle, o.ships] for o in h.main(obs)]
        except Exception:
            return []
    return _agent
