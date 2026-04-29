import heapq
import math
import time
import sys
from collections import defaultdict
from typing import Any

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Fleet, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS,
    distance, point_to_segment_distance
)

class HPlanet:
    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id = id; self.owner = owner; self.x = x; self.y = y
        self.radius = radius; self.ships = ships; self.production = production
        self.reinforcement_target: 'HPlanet | None' = None  # nearest owned planet on shortest path to front

OrbitalInfo = dict[HPlanet, tuple[float, float] | None]
FuturePos = dict[HPlanet, tuple[float, float]]
# dst -> [(src, travel_steps)]: directed graph; src departs now, dst is its intercept position
ProximityGraph = dict[HPlanet, list[tuple['HPlanet', float]]]
# HPlanet -> [(owner, ships, travel_time, src_x, src_y, arrival_x, arrival_y)]
DestinationList = dict[HPlanet, list[tuple[int, float, float, float, float, float, float]]]
# (target planet, heuristic value, fleet orders, intercepts)
# intercepts is parallel to fleet_orders: list of (ix, iy, travel) pre-computed at plan time
MoveOrders = tuple[HPlanet | None, int, list[list], list[tuple]]

sys.path.insert(0, '/home/t/orbitwars')
from visualizer import Visualizer
viz = Visualizer()
def viz_save():
    viz.save('/mnt/c/Users/ajohn/Downloads/orbitwars_viz.html')


MAX_DISTANCE = 30
LOOK_AHEAD = 15
SHIP_SPEED_MAX = 6.0  # matches configuration.shipSpeed default

def fleet_speed(ships: int | float) -> float:
    """Mirror the engine's speed formula exactly."""
    return min(SHIP_SPEED_MAX, 1.0 + (SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5)


class Hellburner:
    def __init__(self):

        self.player: int = 0
        self.scene_step: int = 0
        self.angular_velocity: float = 0.0
        self.planets: list[HPlanet] = []
        self.owned_planets: list[HPlanet] = []
        self.enemy_planets: list[HPlanet] = []
        self.fleets: list[Fleet] = []
        self.orbital_info: OrbitalInfo = {}
        self.inbound_edges: ProximityGraph = {}
        self.outbound_edges: ProximityGraph = {}
        self.future_pos: FuturePos = {}
        self.destination_list: DestinationList = {}

    def build_orbital_info(self, initial_planets: list[Any]) -> None:
        """Return dict mapping Planet -> (r, initial_angle) if orbiting, else None."""
        cx = cy = CENTER
        ip_by_id = {ip[0]: ip for ip in initial_planets}
        self.orbital_info = {}
        for p in self.planets:
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
        (used only for visualization; actual intercepts are computed per-src in evaluate_frontline_strategy).
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

        # front-line: owned planets that have at least one enemy/neutral inbound or outbound edge
        front_line = {
            p for p in self.owned_planets
            if any(src.owner != self.player for src, _ in self.inbound_edges[p])
            or any(dst.owner != self.player for dst, _ in self.outbound_edges[p])
        }

        for p in self.owned_planets:
            p.reinforcement_target = None
            if p in front_line:
                continue  # only rear planets get a reinforcement target
            # Dijkstra through owned-planet subgraph to nearest front-line planet.
            dist: dict[HPlanet, float] = {p: 0.0}
            prev: dict[HPlanet, HPlanet | None] = {p: None}
            heap: list[tuple[float, int, HPlanet]] = [(0.0, id(p), p)]
            heapq.heapify(heap)
            found_front: HPlanet | None = None
            while heap:
                d, _, node = heapq.heappop(heap)
                if d > dist.get(node, float('inf')):
                    continue
                if node in front_line:
                    found_front = node
                    break
                for dst, travel in self.outbound_edges[node]:
                    if dst.owner != self.player:
                        continue  # stay within owned subgraph
                    nd = d + travel
                    if nd < dist.get(dst, float('inf')):
                        dist[dst] = nd
                        prev[dst] = node
                        heapq.heappush(heap, (nd, id(dst), dst))
            if found_front is None:
                continue
            # Walk back to the first hop after p
            node = found_front
            while prev.get(node) is not p:
                node = prev[node]  # type: ignore[assignment]
                if node is None:
                    break
            if node is not None and node is not p and node.owner == self.player:
                p.reinforcement_target = node

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
            neighbors = edge_map.get(p, [])
            if not neighbors:
                continue
            px, py = p.x, p.y
            neighbor_strs: list[str] = []
            for neighbor, t in neighbors:
                tx, ty = self.future_pos[neighbor]
                sx, sy, ex, ey = self.viz_arrow_endpoints(px, py, tx, ty, p.radius, neighbor.radius)
                viz.add_arrow(self.scene_step, sx, sy, ex, ey, color='#22aaff', width=1, length_frac=1.0, head_size=5)
                neighbor_strs.append(f'P{neighbor.id}({t:.0f})')
            on_screen.append(f'  P{p.id}: [{", ".join(neighbor_strs)}]')

        edge_count = sum(len(v) for v in edge_map.values())
        header = f'{direction}_edges: {len(self.planets)} planets, {edge_count} directed edges'
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
            viz.add_arrow(self.scene_step, sx, sy, ex, ey, color='#22aaff', width=1, length_frac=1.0, head_size=5)
            lines.append(f'  P{p.id} -> P{p.reinforcement_target.id}')

        if lines:
            viz.add_text(self.scene_step, f'reinforcement_targets ({len(lines)}):\n' + '\n'.join(lines))
        else:
            viz.add_text(self.scene_step, 'reinforcement_targets: none')

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
        for planet in self.planets:
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
            for planet in self.planets:
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

    def simulate_planet_timeline(self, planet: HPlanet, destination_list: DestinationList) -> tuple[int, float]:
        """Simulate planet ownership/production over time given a list of inbound fleets.
        All arrivals at the same integer turn are resolved simultaneously (highest stack wins).
        Returns (final_owner, excess_ships) where excess_ships is the surplus in the last entry in destination_list.
        """
        cur_owner = planet.owner
        entries = destination_list.get(planet)
        if not bool(entries):
            return cur_owner, 0

        buckets = defaultdict(list)
        for owner, ships, t, _, _, _, _ in entries:
            turn = max(1, math.ceil(t))
            buckets[turn].append((owner, ships))

        last_ships, last_t = entries[-1][1], entries[-1][2]
        last_turn = max(1, math.ceil(last_t))

        cur_ships = float(planet.ships)
        prod = planet.production
        cur_t = 0
        # minimum margin by which the player survived each fight after the last entry landed
        excess_ships = float('inf')

        for turn in sorted(buckets):
            elapsed = turn - cur_t
            if elapsed > 0:
                if cur_owner == self.player:
                    cur_ships += prod * elapsed
                elif cur_owner != -1:
                    cur_ships += prod * elapsed
            cur_t = turn

            owner_ships = defaultdict(float)
            for owner, ships in buckets[turn]:
                owner_ships[owner] += ships

            if owner_ships:
                sorted_owners = sorted(owner_ships.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_owners) == 1:
                    survivor_owner, survivor_ships = sorted_owners[0]
                else:
                    top_owner, top_ships = sorted_owners[0]
                    second_ships = sorted_owners[1][1]
                    survivor_ships = top_ships - second_ships
                    survivor_owner = top_owner if survivor_ships > 0 else -1

                if survivor_ships > 0:
                    if survivor_owner == cur_owner:
                        cur_ships += survivor_ships
                    else:
                        cur_ships -= survivor_ships
                        if cur_ships < 0:
                            cur_owner = survivor_owner
                            cur_ships = abs(cur_ships)

            if turn >= last_turn:
                # track the narrowest margin by which we stayed in control
                margin = cur_ships if cur_owner == self.player else 0.0
                excess_ships = min(excess_ships, margin)

        if excess_ships == float('inf'):
            excess_ships = 0.0
        # excess can't exceed what the last entry actually sent
        excess_ships = min(excess_ships, last_ships)

        return cur_owner, excess_ships

    def evaluate_frontline_strategy(self, target: HPlanet) -> tuple[list[list], list[tuple], bool]:
        """Find the set of nearby ships needed to attack or reinforce a target.
        Returns (fleet_orders, intercepts, battle_won).
        intercepts is parallel to fleet_orders: list of (ix, iy, travel) pre-computed at plan time.
        """
        possible_origins = sorted(
            [(src, travel) for src, travel in self.inbound_edges.get(target, [])
                if src.owner == self.player], key=lambda x: x[1])

        fleet_orders: list[list] = []
        intercepts: list[tuple] = []
        trial_destination_list = {k: list(v) for k, v in self.destination_list.items()}
        trial_destination_list.setdefault(target, [])
        battle_won = False
        for neighbor, _ in possible_origins:
            if neighbor.ships == 0:
                continue
            ships_to_send = int(neighbor.ships)
            angle, ix, iy, travel = self.intercept_planet(neighbor.x, neighbor.y, target, ships_to_send)

            if not math.isfinite(travel):
                continue
            if self.first_planet_hit(neighbor.x, neighbor.y, angle, ships_to_send, neighbor) is not target:
                continue

            trial_destination_list[target].append((self.player, ships_to_send, travel, neighbor.x, neighbor.y, ix, iy))
            fleet_orders.append([neighbor.id, angle, ships_to_send])
            intercepts.append((ix, iy, travel))
            trial_end_owner, excess_ships = self.simulate_planet_timeline(target, trial_destination_list)
            if trial_end_owner == self.player:
                keep = int(excess_ships // 2)
                ships_to_send = max(1, ships_to_send - keep)
                angle, ix, iy, travel = self.intercept_planet(neighbor.x, neighbor.y, target, ships_to_send)
                if not math.isfinite(travel):
                    battle_won = True
                    break
                trial_destination_list[target][-1] = (self.player, ships_to_send, travel, neighbor.x, neighbor.y, ix, iy)
                fleet_orders[-1] = [neighbor.id, angle, ships_to_send]
                intercepts[-1] = (ix, iy, travel)
                battle_won = True
                break

        return fleet_orders, intercepts, battle_won

    def evaluate_move_orders(self) -> MoveOrders:
        """Score every reachable planet and pick the best destination."""
        best_move_orders: MoveOrders = (None, -65535, [], [])

        for target in self.planets:
            if not bool(self.inbound_edges.get(target)):
                continue # effectively unreachable

            # is owned
            if (target.owner == self.player):
                if not bool(self.destination_list.get(target)):
                    continue # no incoming

                end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
                threatened = (end_owner != self.player)
                if not threatened:
                    continue

                fleet_orders, intercepts, battle_won = self.evaluate_frontline_strategy(target)

                if not battle_won:
                    continue  # can't save it; skip for now

                value = target.production
                _, best_value, best_orders, _ = best_move_orders
                if (value > best_value or
                        (value == best_value and len(fleet_orders) < len(best_orders))):
                    best_move_orders = (target, value, fleet_orders, intercepts)

            # not owned
            else:
                end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
                if end_owner == self.player:
                    continue  # already won by in-flight fleets

                fleet_orders, intercepts, battle_won = self.evaluate_frontline_strategy(target)

                if not battle_won:
                    continue

                value = target.production
                if (target.owner == -1):
                    value = value - 1

                _, best_value, best_orders, _ = best_move_orders
                if (value > best_value or
                        (value == best_value and len(fleet_orders) < len(best_orders))):
                    best_move_orders = (target, value, fleet_orders, intercepts)

        return best_move_orders

    def viz_orders(self, best: MoveOrders) -> None:
        """Visualize evaluate_move_orders result: target ring, order arrows, text summary."""
        target_planet, attack_value, fleet_orders, intercepts = best
        if target_planet is None:
            return

        planet_by_id = {p.id: p for p in self.planets}
        viz.add_label(self.scene_step, target_planet.x, target_planet.y,
                           f'P{target_planet.id} val={attack_value}', color='#ffff44')

        lines = [f'orders -> P{target_planet.id} (val={attack_value}):']
        for (from_id, angle, ships), (ix, iy, travel) in zip(fleet_orders, intercepts):
            src = planet_by_id[from_id]
            viz.add_line(self.scene_step, src.x, src.y, ix, iy, color='#ffff44', width=2)
            lines.append(f'  P{from_id}({src.ships}sh) -> {int(ships)}sh angle={angle:.3f} t={travel:.1f}')

        viz.add_text(self.scene_step, '\n'.join(lines))

    def commit_move_orders(self, move: MoveOrders) -> None:
        target, _, fleet_orders, intercepts = move

        for (from_id, _, ships), (ix, iy, travel) in zip(fleet_orders, intercepts):
            src = next((p for p in self.planets if p.id == from_id), None)
            if src is None:
                continue
            src.ships = max(0, src.ships - ships)
            self.destination_list.setdefault(target, [])
            self.destination_list[target].append((self.player, ships, travel, src.x, src.y, ix, iy))

    def main(self, obs: dict[str, Any]) -> list[Any]:
        viz.record(obs)
        _t0 = time.perf_counter()

        self.player = obs['player']
        self.scene_step = obs['step'] - 1
        self.angular_velocity = obs['angular_velocity']

        comet_ids = set(obs['comet_planet_ids'])
        planets_and_comets = [HPlanet(*p) for p in obs['planets']]
        self.planets = [p for p in planets_and_comets if p.id not in comet_ids]
        self.owned_planets = [p for p in self.planets if p.owner == self.player]
        self.enemy_planets = [p for p in self.planets if p.owner != self.player]
        self.fleets = [Fleet(*f) for f in obs['fleets']]

        if not self.enemy_planets:
            return []

        self.build_orbital_info(obs.get('initial_planets', []))
        self.build_proximity_graph()
        self.build_destination_list()

        moves = []
        while True:
            move_orders = self.evaluate_move_orders()
            target_planet, _, fleet_orders, _ = move_orders
            if target_planet is None:
                break
            #self.viz_orders(move_orders)
            self.commit_move_orders(move_orders)
            moves.extend(fleet_orders)

        elapsed_ms = (time.perf_counter() - _t0) * 1000
        viz.add_text(self.scene_step, f'hellburner ms: {elapsed_ms:.2f}ms')
        #self.viz_proximity_graph(False) # inbound: True, outbound: False.
        self.viz_reinforcement_targets()
        #self.viz_destination_list()
        #viz.add_text(self.scene_step, 'moves: ' + str(moves))

        return moves


def hellburner(obs: dict[str, Any]) -> list[Any]:
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
