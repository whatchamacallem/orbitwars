import math
import time
import sys
from collections import defaultdict
from typing import Any

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT,
    distance, point_to_segment_distance
)

OrbitalInfo = dict[Planet, tuple[float, float] | None]
FuturePos = dict[Planet, tuple[float, float]]
ProximityGraph = dict[Planet, list[tuple[Planet, float]]]
# Planet -> [(owner, ships, travel_time, src_x, src_y, arrival_x, arrival_y)]
DestinationList = dict[Planet, list[tuple[int, float, float, float, float, float, float]]]

sys.path.insert(0, '/home/t/orbitwars')
from visualizer import Visualizer
viz = Visualizer()
def viz_save():
    viz.save('/mnt/c/Users/ajohn/Downloads/orbitwars_viz.html')


MAX_DISTANCE = 30
LOOK_AHEAD = 15
SHIP_SPEED_MAX = 6.0
EVAL_HORIZON = 40

def fleet_speed(ships: int | float) -> float:
    """Mirror the engine's speed formula exactly."""
    return min(SHIP_SPEED_MAX, 1.0 + (SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5)


class Hellburner:
    def __init__(self):

        self.player: int = 0
        self.scene_step: int = 0
        self.angular_velocity: float = 0.0
        self.planets: list[Planet] = []
        self.owned_planets: list[Planet] = []
        self.enemy_planets: list[Planet] = []
        self.fleets: list[Fleet] = []
        self.orbital_info: OrbitalInfo = {}
        self.proximity_graph: ProximityGraph = {}
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
        """Build adjacency list: planet -> list of (neighbor, dist) within MAX_DISTANCE.

        scene_step is the rotation index the planets are currently at (= obs.step - 1).
        Orbiting planets are projected LOOK_AHEAD turns into the future.
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

        self.proximity_graph = {p: [] for p in self.planets}
        for i, a in enumerate(self.planets):
            ax, ay = self.future_pos[a]
            for b in self.planets[i + 1:]:
                bx, by = self.future_pos[b]
                dist = distance((ax, ay), (bx, by))
                if dist <= MAX_DISTANCE:
                    self.proximity_graph[a].append((b, dist))
                    self.proximity_graph[b].append((a, dist))

    def viz_proximity_graph(self) -> None:
        """Draw proximity_graph edges and future-position planet labels onto the visualizer frame."""
        moving = {p for p in self.planets if self.future_pos[p] != (p.x, p.y)}
        seen_edges = set()
        for p, neighbors in self.proximity_graph.items():
            fpx, fpy = self.future_pos[p]
            if p in moving:
                viz.add_label(self.scene_step, fpx, fpy, f'P{p.id}', color='#22ffcc')
            for neighbor, _ in neighbors:
                edge = (min(p.id, neighbor.id), max(p.id, neighbor.id))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                nx, ny = self.future_pos[neighbor]
                viz.add_line(self.scene_step, fpx, fpy, nx, ny, color='#22aaff', width=1)

    def intercept_planet(
        self,
        sx: float, sy: float, target: Planet, ships: int | float,
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
            # Recompute final position from converged travel so tx/ty/angle are consistent.
            a = ia + self.angular_velocity * (self.scene_step + travel - 0.5)
            tx, ty = cx + r * math.cos(a), cy + r * math.sin(a)
        angle = math.atan2(ty - sy, tx - sx)
        return angle, tx, ty, travel

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
                if delta <= half_cone and travel < best_t:
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

    def is_destination(self, planet: Planet) -> bool:
        return bool(self.destination_list.get(planet))

    def simulate_planet_timeline(self, planet: Planet, destination_list: DestinationList) -> tuple[int, float]:
        """Simulate planet ownership/production over time given a list of inbound fleets.
        All arrivals at the same integer turn are resolved simultaneously (highest stack wins).
        Returns (final_owner, final_ships).
        """
        buckets = defaultdict(list)
        for owner, ships, t, _, _, _, _ in destination_list.get(planet, []):
            turn = max(1, math.ceil(t))
            if turn <= EVAL_HORIZON:
                buckets[turn].append((owner, ships))

        cur_owner = planet.owner
        cur_ships = float(planet.ships)
        prod = planet.production
        cur_t = 0

        for turn in sorted(buckets):
            elapsed = turn - cur_t
            if elapsed > 0:
                if cur_owner == self.player:
                    cur_ships += prod * elapsed
                elif cur_owner != -1:
                    cur_ships += prod * elapsed
            cur_t = turn

            # Step 1: sum arriving fleet ships per owner (garrison is NOT in this pool)
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

                # Step 3: survivor fights garrison (or reinforces if same owner)
                if survivor_ships > 0:
                    if survivor_owner == cur_owner:
                        cur_ships += survivor_ships
                    else:
                        cur_ships -= survivor_ships
                        if cur_ships < 0:
                            cur_owner = survivor_owner
                            cur_ships = abs(cur_ships)

        return cur_owner, cur_ships

    def evaluate_destinations(self) -> tuple[Planet | None, int, list[list]]:
        """Score every reachable planet and pick the best destination."""
        best_planet = None
        best_value = -65535
        best_orders: list[list] = []

        for target in self.planets:
            is_owned = (target.owner == self.player)

            if is_owned:
                if not self.is_destination(target):
                    continue
                end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
                threatened = (end_owner != self.player)
                if not threatened:
                    continue

                # Find nearby owned planets sorted by proximity_graph distance, then
                # try adding possible reinforcements one by one (closest first)
                # until the planet is saved.
                possible_reinforcements = sorted(
                    [(neighbor, dist) for neighbor, dist in self.proximity_graph.get(target, [])
                        if neighbor.owner == self.player], key=lambda x: x[1])

                rescue_orders: list[list] = []
                trial_destination_list = {k: list(v) for k, v in self.destination_list.items()}
                saved = False
                for neighbor, _ in possible_reinforcements:
                    if neighbor.ships == 0:
                        continue
                    ships_to_send = int(neighbor.ships)
                    angle, ix, iy, travel = self.intercept_planet(
                        neighbor.x, neighbor.y, target, ships_to_send)

                    trial_destination_list[target].append((self.player, ships_to_send, travel, neighbor.x, neighbor.y, ix, iy))
                    rescue_orders.append([neighbor.id, angle, ships_to_send])
                    trial_end_owner, _ = self.simulate_planet_timeline(target, trial_destination_list)
                    if trial_end_owner == self.player:
                        saved = True
                        break

                if not saved:
                    continue  # can't save it; skip for now

                value = target.production
                if (value > best_value or
                        (value == best_value and len(rescue_orders) < len(best_orders))):
                    best_planet = target
                    best_value = value
                    best_orders = rescue_orders

            is_neutral = (target.owner == -1)

            # Accumulate ships from multiple sources (fastest-arriving first)

            # Don't attack neutral planets that can't be taken outright

            # Simulate with fleets added

        return best_planet, best_value, best_orders

    def viz_orders(self, best_planet: 'Planet | None', best_value: int, best_orders: list[list]) -> None:
        """Visualize evaluate_destinations result: target ring, order arrows, text summary."""
        if best_planet is None or not best_orders:
            return

        planet_by_id = {p.id: p for p in self.planets}
        viz.add_label(self.scene_step, best_planet.x, best_planet.y,
                           f'P{best_planet.id} val={best_value}', color='#ffff44')

        lines = [f'orders -> P{best_planet.id} (val={best_value}):']
        for from_id, angle, ships in best_orders:
            src = planet_by_id[from_id]
            _, ix, iy, travel = self.intercept_planet(src.x, src.y, best_planet, ships)
            viz.add_line(self.scene_step, src.x, src.y, ix, iy, color='#ffff44', width=2)
            lines.append(f'  P{from_id}({src.ships}sh) -> {int(ships)}sh angle={angle:.3f} t={travel:.1f}')

        viz.add_text(self.scene_step, '\n'.join(lines))

    def exec_snipe(self, moves: list[Any]) -> None:
        best_travel = float('inf')
        best_mine = None
        best_enemy = None
        for mine in self.owned_planets:
            for enemy in self.enemy_planets:
                _, _, _, travel = self.intercept_planet(mine.x, mine.y, enemy, int(mine.ships))
                if travel < best_travel:
                    best_travel = travel
                    best_mine = mine
                    best_enemy = enemy
        if best_mine is None or best_mine.ships < 10:
            reason = 'no owned planet' if best_mine is None else f'P{best_mine.id} ships={best_mine.ships} < 5'
            #viz.add_text(self.scene_step, f'snipe: skip ({reason})')
            return
        ships = int(best_mine.ships)
        angle, ix, iy, travel = self.intercept_planet(best_mine.x, best_mine.y, best_enemy, ships)
        moves.append([best_mine.id, angle, ships])
        #viz.add_text(self.scene_step, f'snipe: P{best_mine.id}({ships}sh) -> P{best_enemy.id}({best_enemy.ships}sh) angle={angle:.3f} travel={travel:.1f}t')
        #viz.add_line(self.scene_step, best_mine.x, best_mine.y, ix, iy, color='#ff44ff', width=2)

    def main(self, obs: dict[str, Any]) -> list[Any]:
        viz.record(obs)
        _t0 = time.perf_counter()

        self.player = obs['player']
        self.scene_step = obs['step'] - 1
        self.angular_velocity = obs['angular_velocity']

        comet_ids = set(obs['comet_planet_ids'])
        planets_and_comets = [Planet(*p) for p in obs['planets']]
        self.planets = [p for p in planets_and_comets if p.id not in comet_ids]
        self.owned_planets = [p for p in self.planets if p.owner == self.player]
        self.enemy_planets = [p for p in self.planets if p.owner != self.player]
        self.fleets = [Fleet(*f) for f in obs['fleets']]

        if not self.enemy_planets:
            return []

        self.build_orbital_info(obs.get('initial_planets', []))
        self.build_proximity_graph()
        self.build_destination_list()

        best_planet, best_value, best_orders = self.evaluate_destinations()

        elapsed_ms = (time.perf_counter() - _t0) * 1000
        viz.add_text(self.scene_step, f'hellburner ms: {elapsed_ms:.2f}ms')
        self.viz_orders(best_planet, best_value, best_orders)
        #self.viz_proximity_graph()
        #self.viz_destination_list()

        moves = []
        if best_planet is not None:
            moves = best_orders
        else:
            self.exec_snipe(moves)

        #viz.add_text(self.scene_step, 'moves: ' + str(moves))
        return moves


def hellburner(obs: dict[str, Any]) -> list[Any]:
    _agent = Hellburner()
    return _agent.main(obs)
