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
def viz_save():
    viz.save('/mnt/c/Users/ajohn/Downloads/orbitwars_viz.html')

class HPlanet:
    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id = id; self.owner = owner; self.x = x; self.y = y
        self.radius = radius; self.ships = ships; self.production = production
        self.reinforcement_target: 'HPlanet | None' = None  # nearest owned planet on shortest path to front

# HPlanet -> (orbital_radius, initial_angle) if the planet orbits the sun, else None
OrbitalInfo = dict[HPlanet, tuple[float, float] | None]
# HPlanets rotated by LOOK_AHEAD
FuturePos = dict[HPlanet, tuple[float, float]]
# dst -> [(src, travel_steps)]: directed graph; src departs now, dst is its intercept position
ProximityGraph = dict[HPlanet, list[tuple[HPlanet, float]]]
# HPlanet -> [(owner, ships, travel_time, src_x, src_y, arrival_x, arrival_y)]
DestinationList = dict[HPlanet, list[tuple[int, float, float, float, float, float, float]]]
# [planet_id, angle, ships]
FleetOrders = list[list]


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

SHIP_SPEED_MAX = 6.0  # matches configuration.shipSpeed default

MAX_DISTANCE = 35
LOOK_AHEAD = 10
REINFORCEMENT_SIZE = 10
GARRISON_SIZE = 10

# Search budget
TIME_BUDGET_S       = 0.800  # hard wall; return best-found-so-far if exceeded
DFS_EARLY_EXIT_S    = 0.080  # soft cutoff inside DFS loop (leaves room for emit/reinforce)

# Lookahead
UNIFIED_LOOK_AHEAD  = 25     # horizon = min(scene_step + UNIFIED_LOOK_AHEAD, 500)

# Candidate generation
MAX_CANDIDATES      = 6      # hard cap on DFS branching factor (reduce to 4 if timing spikes)

# Scoring
ENEMY_WEIGHT        = 0.8    # penalty multiplier for enemy production (tune toward 1.0)


def fleet_speed(ships: int | float) -> float:
    """Mirror the engine's speed formula exactly."""
    return min(SHIP_SPEED_MAX, 1.0 + (SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5)


def warchest_copy(state: 'WarchestState') -> 'WarchestState':
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

    def build_orbital_info(self, initial_planets: list[Any]) -> None:
        """Return dict mapping Planet -> (r, initial_angle) if orbiting, else None."""
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
                self.orbital_info[p] = (r, math.atan2(ip[3] - cy, ip[2] - cx))
            else:
                self.orbital_info[p] = None

    def build_proximity_graph(self) -> None:
        """Build directed adjacency list: dst -> [(src, travel_steps)].

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
                r, ia = orb
                a = ia + self.angular_velocity * (self.scene_step + 1 + LOOK_AHEAD)
                self.future_pos[p] = (cx + r * math.cos(a), cy + r * math.sin(a))
            else:
                self.future_pos[p] = (p.x, p.y)

        self.inbound_edges = {p: [] for p in self.planets}
        for src in self.planets:
            for dst in self.planets:
                if dst is src:
                    continue
                travel = distance((src.x, src.y), self.future_pos[dst])
                if travel <= MAX_DISTANCE:
                    self.inbound_edges[dst].append((src, travel))

        # self.outbound_edges[p] = [(dst, travel)] — keyed by source, complement of the inbound-keyed inbound_edges.
        self.outbound_edges = {p: [] for p in self.planets}
        for dst, inbound in self.inbound_edges.items():
            for src, travel in inbound:
                self.outbound_edges[src].append((dst, travel))

    def intercept_planet(
        self,
        sx: float, sy: float, target: HPlanet, ships: int | float,
        tol: float = 1e-6, max_iters: int = 30,
    ) -> tuple[float, float, float, float]:
        """Aim angle from (sx, sy) toward where target will be when a fleet arrives.
        Returns (angle, intercept_x, intercept_y, travel_steps).
        """
        speed = fleet_speed(ships)
        orb = self.orbital_info[target]
        if orb is None:
            tx, ty = target.x, target.y
            travel = distance((sx, sy), (tx, ty)) / speed
        else:
            cx = cy = CENTER
            r, ia = orb
            # Seed: straight-line travel time to the planet's current position.
            travel = distance((sx, sy), (target.x, target.y)) / speed
            for _ in range(max_iters):
                a = ia + self.angular_velocity * (self.scene_step + travel - 0.5)
                new_tx, new_ty = cx + r * math.cos(a), cy + r * math.sin(a)
                new_travel = distance((sx, sy), (new_tx, new_ty)) / speed
                # Damp update: average old and new travel to suppress oscillation.
                new_travel = 0.5 * (travel + new_travel - 0.5)
                if abs(new_travel - travel) < tol:
                    travel = new_travel
                    break
                travel = new_travel
            else:
                # Diverged: fleet too slow to catch this planet's orbital speed.
                return 0.0, target.x, target.y, math.inf
            # Recompute final position from converged travel so tx/ty/angle are consistent.
            a = ia + self.angular_velocity * (self.scene_step + travel - 0.5)
            tx, ty = cx + r * math.cos(a), cy + r * math.sin(a)
        angle = math.atan2(ty - sy, tx - sx)
        return angle, tx, ty, travel

    def first_planet_hit(self, sx: float, sy: float, angle: float, ships: int | float, source: HPlanet) -> HPlanet | None:
        """Return the first planet a fleet launched from (sx, sy) at `angle` would hit, or None.
        Returns None if the path crosses the sun before any planet is hit."""
        best = None
        best_t = float('inf')
        for planet in self.all_bodies:   # ← was self.planets; CRITICAL: comets must block paths
            if planet is source:
                continue
            needed_angle, px, py, travel = self.intercept_planet(sx, sy, planet, ships)
            dist = distance((sx, sy), (px, py))
            if dist < planet.radius:
                half_cone = math.pi
            else:
                half_cone = math.asin(min(1.0, planet.radius / dist))
            delta = abs(math.atan2(math.sin(angle - needed_angle), math.cos(angle - needed_angle)))
            if math.isfinite(travel) and delta <= half_cone and travel < best_t:
                best_t = travel
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
        Populates self.destination_list: Planet -> list of (owner, ships, t, src_x, src_y, arrival_x, arrival_y).
        t is continuous time in turns.
        """
        self.destination_list = defaultdict(list)
        for fleet in self.fleets:
            best = None
            best_t = float('inf')
            for planet in self.all_bodies:   # ← was self.planets; CRITICAL: comets must block paths
                needed_angle, px, py, travel = self.intercept_planet(
                    fleet.x, fleet.y, planet, fleet.ships
                )
                dist = distance((fleet.x, fleet.y), (px, py))
                if dist < planet.radius:
                    half_cone = math.pi
                else:
                    half_cone = math.asin(min(1.0, planet.radius / dist))
                delta = abs(math.atan2(math.sin(fleet.angle - needed_angle),
                                       math.cos(fleet.angle - needed_angle)))
                if math.isfinite(travel) and delta <= half_cone and travel < best_t:
                    best_t = travel
                    best = (planet, travel, px, py)
            if best is not None:
                planet, travel, px, py = best
                self.destination_list[planet].append((fleet.owner, fleet.ships, travel, fleet.x, fleet.y, px, py))

    def build_reinforcement_targets(self) -> None:
        front_line = {
            p for p in self.owned_planets
            if any(src.owner != self.player for src, _ in self.inbound_edges[p])
            or any(dst.owner != self.player for dst, _ in self.outbound_edges[p])
        }

        # BFS hop-distance from every owned node to nearest frontline planet,
        # traversing only owned-planet edges (frontline nodes are sinks, not sources).
        hops_to_front: dict[HPlanet, int] = {p: 0 for p in front_line}
        queue: list[HPlanet] = list(front_line)
        head = 0
        while head < len(queue):
            node = queue[head]; head += 1
            for src, _ in self.inbound_edges[node]:
                if src.owner != self.player or src in hops_to_front:
                    continue
                hops_to_front[src] = hops_to_front[node] + 1
                queue.append(src)

        for p in self.owned_planets:
            p.reinforcement_target = None
            if p in front_line:
                continue

            direct_front = [
                dst for dst, _ in self.outbound_edges[p]
                if dst in front_line
            ]
            if direct_front:
                p.reinforcement_target = min(direct_front, key=lambda d: d.ships)
                continue

            # No direct edge to a frontline planet: pick the direct neighbor with
            # fewest hops to the front, breaking ties by fewest ships at destination.
            reachable = [
                dst for dst, _ in self.outbound_edges[p]
                if dst.owner == self.player and dst not in front_line and dst in hops_to_front
            ]
            if reachable:
                p.reinforcement_target = min(reachable, key=lambda d: (hops_to_front[d], d.ships))

    def send_reinforcements(self) -> FleetOrders:
        """ Allows sending by an intermediate planet if in the way. """
        orders: FleetOrders = []
        for p in self.owned_planets:
            if p.reinforcement_target is None:
                continue
            if p.ships < (REINFORCEMENT_SIZE + GARRISON_SIZE):
                continue
            has_enemy_incoming = any(
                src.owner != self.player
                for src, _ in self.inbound_edges.get(p, []) )
            if has_enemy_incoming:
                continue
            target = p.reinforcement_target
            ships = int(p.ships - GARRISON_SIZE)
            angle, ix, iy, travel = self.intercept_planet(p.x, p.y, target, ships)
            if not math.isfinite(travel):
                continue
            orders.append([p.id, angle, ships])
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
            arrivals_by_dest: dict[int, list[tuple[int, float]]] = defaultdict(list)
            remaining_friendly = []
            for f in state.friendly_fleets:
                if f.arrival_turn == t:
                    arrivals_by_dest[f.destination_id].append((f.owner, f.fleet_size))
                else:
                    remaining_friendly.append(f)
            state.friendly_fleets = remaining_friendly

            remaining_enemy = []
            for f in state.enemy_fleets:
                if f.arrival_turn == t:
                    arrivals_by_dest[f.destination_id].append((f.owner, f.fleet_size))
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
        arrivals: list[tuple[int, float]],
    ) -> None:
        # Stage 1: arriving fleets fight each other (garrison is separate)
        owner_ships: dict[int, float] = defaultdict(float)
        for owner, ships in arrivals:
            owner_ships[owner] += ships

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
            for owner, ships, t, _, _, _, _ in sorted(arrivals, key=lambda a: a[2]):
                arrival  = self.scene_step + math.ceil(t)
                is_cap   = (owner != dest_planet.owner)
                # Project target garrison to arrival time so garrison_on_arrival is accurate.
                if is_cap:
                    garrison_at_arrival = dest_planet.ships + dest_planet.production * math.ceil(t)
                    gar_on_arr = max(0.0, ships - garrison_at_arrival)
                else:
                    gar_on_arr = ships
                # Only the first friendly fleet to a capture target claims the production bonus.
                # Without this guard, two friendly fleets to the same planet each get
                # production_id set, and warchest_score counts that planet's income twice.
                if is_cap and owner == self.player:
                    if dest_planet.id in friendly_capture_claimed:
                        is_cap = False   # treat as reinforcement for scoring purposes
                        gar_on_arr = ships
                    else:
                        friendly_capture_claimed.add(dest_planet.id)
                fleet = WarchestFleet(
                    owner=owner,
                    destination_id=dest_planet.id,
                    fleet_size=ships,
                    garrison_on_arrival=gar_on_arr,
                    arrival_turn=arrival,
                    is_capture=is_cap,
                    production_id=dest_planet.id if is_cap else -1,
                )
                if owner == self.player:
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
    ) -> dict:
        candidates = sorted(
            [(src, travel)
             for src, travel in self.inbound_edges.get(target, [])
             if state.ownership.get(src.id) == self.player
             and src.id not in state.committed_ids
             and state.garrison.get(src.id, 0) > 0],
            key=lambda x: x[1],   # closest first
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
            first_src = candidates[0][0]
            first_ships = int(state.garrison.get(first_src.id, 1)) or 1
            _, _, _, min_travel_turns = self.intercept_planet(
                first_src.x, first_src.y, target, first_ships
            )
            min_travel = min_travel_turns if math.isfinite(min_travel_turns) else candidates[0][1]
            arrival_garrison = (state.garrison.get(target.id, 0.0) +
                                state.production.get(target.id, 0.0) * math.ceil(min_travel))

        selected = []   # (src, ships_now, angle, arrival_turn_if_launched_now)
        total_ships  = 0.0
        latest_arrival = 0

        for src, _ in candidates:
            ships = int(state.garrison.get(src.id, 0))
            if ships <= 0:
                continue
            angle, ix, iy, t = self.intercept_planet(src.x, src.y, target, ships)
            if not math.isfinite(t):
                continue
            if self.first_planet_hit(src.x, src.y, angle, ships, src) is not target:
                continue
            arrival = state.turn + math.ceil(t)
            if arrival > horizon:
                continue
            selected.append((src, ships, angle, arrival))
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
            actual_arrival_garrison = (state.garrison.get(target.id, 0.0) +
                                       state.production.get(target.id, 0.0) * (latest_arrival - state.turn))
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
        assignment: dict[int, tuple[float, int, int]] = {}

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
                r, ia = orb
                a  = ia + self.angular_velocity * (launch_turn - 0.5)
                sx, sy = cx + r * math.cos(a), cy + r * math.sin(a)
            else:
                sx, sy = src.x, src.y

            angle2, ix2, iy2, t2 = self.intercept_planet(sx, sy, target, projected)
            if not math.isfinite(t2):
                continue
            # Re-validate path from the delayed launch position (source may have rotated).
            if self.first_planet_hit(sx, sy, angle2, projected, src) is not target:
                continue
            arrival2 = launch_turn + math.ceil(t2)
            # Enforce exact synchronization: all sources must share the same arrival turn.
            # A 1-turn drift would mean the first source fights the garrison alone before
            # the second arrives, violating the synchronized-arrival invariant (§6.1).
            if arrival2 != latest_arrival:
                continue
            assignment[src.id] = (projected, launch_turn, arrival2)

        if not assignment:
            return {}

        # Phase 3: safety check — ensure source survives until its launch_turn.
        for src_id, (ships, launch_turn, _) in list(assignment.items()):
            if self.warchest_source_would_be_lost(state, src_id, ships, launch_turn):
                del assignment[src_id]

        total_sync = sum(ships for ships, _, _ in assignment.values())
        if total_sync <= arrival_garrison:
            return {}  # safety filter removed too many sources

        # Phase 4: trim excess — halve the overshoot on the first-added (highest-delay) source.
        # selected is ordered closest-first (ascending arrival_now = descending delay), so the
        # first key inserted into assignment has the largest delay and the most accumulated ships
        # to spare.
        excess = total_sync - arrival_garrison - 1
        if excess > 1:
            first_id = list(assignment.keys())[0]
            ships, lt, at = assignment[first_id]
            trimmed = max(1, int(ships - excess // 2))
            assignment[first_id] = (trimmed, lt, at)

        return assignment

    def warchest_execute(
        self,
        state: WarchestState,
        target: HPlanet,
        assignment: dict,  # {source_id: (ships, launch_turn, arrival_turn)}
    ) -> WarchestState:
        total_ships    = 0.0
        latest_arrival = 0

        for source_id, (ships, launch_turn, arrival_turn) in sorted(
            assignment.items(), key=lambda x: x[1][1]   # sort ascending by launch_turn
        ):
            # Advance to one turn before launch so production for launch_turn hasn't run yet.
            if launch_turn - 1 > state.turn:
                self.warchest_advance(state, launch_turn - 1)
            # Deduct ships before this turn's production runs.
            state.garrison[source_id] = max(0.0, state.garrison.get(source_id, 0.0) - ships)
            state.committed_ids.add(source_id)
            # Now advance through launch_turn: applies production, resolves any arrivals.
            self.warchest_advance(state, launch_turn)
            total_ships    += ships
            latest_arrival  = max(latest_arrival, arrival_turn)

        # All ships arrive simultaneously at latest_arrival.
        # Project target garrison forward to arrival time — enemy planets produce while we travel.
        is_capture = state.ownership.get(target.id) != self.player
        target_garrison_now = state.garrison.get(target.id, 0.0)
        if is_capture:
            turns_to_arrival = latest_arrival - state.turn
            target_garrison_at_arrival = (target_garrison_now
                                          + state.production.get(target.id, 0.0) * turns_to_arrival)
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
                total -= (garrison + prod * remaining) * ENEMY_WEIGHT

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
                total -= prod * remaining_after * (1.0 + ENEMY_WEIGHT)
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
            net_gain = p.production * production_turns - capture_cost
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
        viable = []  # (arrival_turn, ships) for all sources with a valid path

        for src, _ in self.inbound_edges.get(target, []):
            if state.ownership.get(src.id) != self.player:
                continue
            ships = state.garrison.get(src.id, 0)
            if ships <= 0:
                continue
            _, _, _, t = self.intercept_planet(src.x, src.y, target, int(ships))
            if not math.isfinite(t):
                continue
            arrival = state.turn + math.ceil(t)
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
            trial = warchest_copy(initial)
            last_enemy_arrival = max(f.arrival_turn for f in enemy_inbound)
            self.warchest_advance(trial, min(last_enemy_arrival, horizon))
            if trial.ownership.get(p.id) != self.player:
                defense.append(p)

        offense = []
        for p in self.planets:
            if p.owner == self.player or p.id in self.comet_ids:
                continue
            if not self.inbound_edges.get(p):
                continue
            ct = self.warchest_earliest_capture(initial, p, horizon)
            if not math.isfinite(ct):
                continue
            gain = p.production * (horizon - ct) - p.ships
            if gain > 0:
                offense.append((p, gain))

        offense.sort(key=lambda x: x[1], reverse=True)
        return (defense + [p for p, _ in offense])[:MAX_CANDIDATES]

    def warchest_emit_moves(self, sequence) -> FleetOrders:
        moves = []
        planet_by_id = {p.id: p for p in self.planets}
        for target_planet, assignment in sequence:
            for source_id, (ships, launch_turn, arrival_turn) in assignment.items():
                if launch_turn != self.scene_step:
                    continue  # deferred; will be re-planned next turn
                src = planet_by_id.get(source_id)
                if src is None:
                    continue
                angle, ix, iy, travel = self.intercept_planet(src.x, src.y, target_planet, ships)
                if not math.isfinite(travel):
                    continue
                if self.first_planet_hit(src.x, src.y, angle, ships, src) is not target_planet:
                    continue
                moves.append([source_id, angle, ships])
        return moves

    def warchest_dfs(self, state, remaining, sequence, horizon, t0, best):
        score = self.warchest_score(state, horizon)
        if score > best[0]:
            best[0] = score
            best[1] = list(sequence)

        if time.perf_counter() - t0 > DFS_EARLY_EXIT_S:
            return  # soft cutoff; return best found so far

        if not remaining:
            return

        if self.warchest_upper_bound(state, remaining, horizon) <= best[0]:
            return  # prune: branch cannot improve best

        for i, planet in enumerate(remaining):
            assignment = self.warchest_assign_fleet(state, planet, horizon)
            if not assignment:
                continue

            next_state = self.warchest_execute(warchest_copy(state), planet, assignment)
            self.warchest_dfs(
                next_state,
                remaining[:i] + remaining[i+1:],
                sequence + [(planet, assignment)],
                horizon, t0, best,
            )

    def run_unified_search(self, horizon: int) -> FleetOrders:
        self._warchest_committed_ids = set()
        initial    = self.warchest_initial_state()
        candidates = self.warchest_candidates(initial, horizon)

        best = [self.warchest_score(initial, horizon), []]
        t0   = time.perf_counter()
        self.warchest_dfs(initial, candidates, [], horizon, t0, best)

        # Only mark sources that launch this turn as committed; deferred sources still sit at home
        # and must remain available to send_reinforcements on turns when they don't launch.
        self._warchest_committed_ids = {
            src_id
            for _, assignment in best[1]
            for src_id, (_, launch_turn, _) in assignment.items()
            if launch_turn == self.scene_step
        }
        return self.warchest_emit_moves(best[1])

    def drain_comets(self) -> FleetOrders:
        orders = []
        for comet in self.owned_comets:
            if comet.ships <= 0:
                continue
            # Find nearest non-comet planet within reach
            best_dst, best_travel = None, math.inf
            for p in self.planets:
                _, _, _, t = self.intercept_planet(comet.x, comet.y, p, int(comet.ships))
                if math.isfinite(t) and t < best_travel:
                    best_travel = t
                    best_dst = p
            if best_dst is None:
                continue
            ships = int(comet.ships)
            angle, ix, iy, travel = self.intercept_planet(comet.x, comet.y, best_dst, ships)
            if not math.isfinite(travel):
                continue
            if self.first_planet_hit(comet.x, comet.y, angle, ships, comet) is not best_dst:
                continue
            orders.append([comet.id, angle, ships])
        return orders

    # ------------------------------------------------------------------
    # viz

    @staticmethod
    def viz_arrow_endpoints(
        px: float, py: float, tx: float, ty: float,
        origin_radius: float, target_radius: float,
    ) -> tuple[float, float, float, float]:
        dx, dy = tx - px, ty - py
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return px, py, tx, ty
        ux, uy = dx / d, dy / d
        sx, sy = px + ux * origin_radius, py + uy * origin_radius
        ex, ey = tx - ux * target_radius, ty - uy * target_radius
        return sx, sy, ex, ey

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
            for neighbor, t in neighbors:
                tx, ty = self.future_pos[neighbor]
                sx, sy, ex, ey = self.viz_arrow_endpoints(px, py, tx, ty, p.radius, neighbor.radius)
                color = '#ff8844' if p.owner == 1 else '#22aaff'
                viz.add_arrow(self.scene_step, sx, sy, ex, ey, color=color, width=1, length_frac=1.0, head_size=5)
                neighbor_strs.append(f'P{neighbor.id}({t:.0f})')
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
            sx, sy, ex, ey = self.viz_arrow_endpoints(px, py, tx, ty, p.radius, p.reinforcement_target.radius)
            viz.add_arrow(self.scene_step, sx, sy, ex, ey, color='#44ff88', width=1, length_frac=1.0, head_size=5)
            lines.append(f'  P{p.id} -> P{p.reinforcement_target.id}')

        if lines:
            viz.add_text(self.scene_step, f'reinforcement_targets ({len(lines)}):\n' + '\n'.join(lines))
        else:
            viz.add_text(self.scene_step, 'reinforcement_targets: none')

    def viz_destination_list(self) -> None:
        """Draw a line from each fleet to its destination planet's arrival position."""
        lines = []
        for planet, arrivals in self.destination_list.items():
            for owner, ships, t, sx, sy, px, py in arrivals:
                color = '#44ff88' if owner == 0 else '#ff8844'
                viz.add_line(self.scene_step, sx, sy, px, py, color=color, width=1)
                lines.append(f' {owner} ({ships}) -> P{planet.id}({planet.ships}) t={round(t)} v={fleet_speed(ships):.2f}')
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

        horizon = min(self.scene_step + UNIFIED_LOOK_AHEAD, 500)
        moves   = self.run_unified_search(horizon)

        reinforcement_orders = self.send_reinforcements()
        reinforcement_orders = [o for o in reinforcement_orders
                                if o[0] not in self._warchest_committed_ids]
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
        return _agent.main(obs)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        try:
            viz.add_text(_agent.scene_step, tb)
        except Exception:
            pass
        return []
