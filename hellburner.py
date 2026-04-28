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
# Planet -> [(fleet, travel_time, arrival_x, arrival_y)]
DestinationList = dict[Planet, list[tuple[Fleet, float, float, float]]]

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


def build_orbital_info(planets: list[Planet], initial_planets: list[Any]) -> OrbitalInfo:
    """Return dict mapping Planet -> (r, initial_angle) if orbiting, else None."""
    cx = cy = CENTER
    ip_by_id = {ip[0]: ip for ip in initial_planets}
    orbital_info = {}
    for p in planets:
        r = distance((p.x, p.y), (cx, cy))
        if r + p.radius < ROTATION_RADIUS_LIMIT and p.id in ip_by_id:
            ip = ip_by_id[p.id]
            orbital_info[p] = (r, math.atan2(ip[3] - cy, ip[2] - cx))
        else:
            orbital_info[p] = None
    return orbital_info


def build_proximity_graph(
    planets: list[Planet], orbital_info: OrbitalInfo,
    angular_velocity: float, scene_step: int,
) -> tuple[ProximityGraph, FuturePos]:
    """Build adjacency list: planet -> list of (neighbor, dist) within MAX_DISTANCE.

    scene_step is the rotation index the planets are currently at (= obs.step - 1).
    Orbiting planets are projected LOOK_AHEAD turns into the future.
    """
    cx = cy = CENTER

    future_pos = {}
    for p in planets:
        orb = orbital_info[p]
        if orb is not None:
            r, ia = orb
            a = ia + angular_velocity * (scene_step + 1 + LOOK_AHEAD)
            future_pos[p] = (cx + r * math.cos(a), cy + r * math.sin(a))
        else:
            future_pos[p] = (p.x, p.y)

    proximity_graph = {p: [] for p in planets}
    for i, a in enumerate(planets):
        ax, ay = future_pos[a]
        for b in planets[i + 1:]:
            bx, by = future_pos[b]
            dist = distance((ax, ay), (bx, by))
            if dist <= MAX_DISTANCE:
                proximity_graph[a].append((b, dist))
                proximity_graph[b].append((a, dist))
    return proximity_graph, future_pos


def viz_proximity_graph(
    viz: Visualizer, scene_step: int, planets: list[Planet],
    proximity_graph: ProximityGraph, future_pos: FuturePos,
) -> None:
    """Draw proximity_graph edges and future-position planet labels onto the visualizer frame."""
    moving = {p for p in planets if future_pos[p] != (p.x, p.y)}
    seen_edges = set()
    for p, neighbors in proximity_graph.items():
        fpx, fpy = future_pos[p]
        if p in moving:
            viz.add_label(scene_step, fpx, fpy, f'P{p.id}', color='#22ffcc')
        for nb, dist in neighbors:
            edge = (min(p.id, nb.id), max(p.id, nb.id))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            nx, ny = future_pos[nb]
            viz.add_line(scene_step, fpx, fpy, nx, ny, color='#22aaff', width=1)


def intercept_planet(
    sx: float, sy: float, target: Planet, orbital_info: OrbitalInfo,
    angular_velocity: float, scene_step: int, ships: int | float,
    tol: float = 1e-6, max_iters: int = 30,
) -> tuple[float, float, float, float]:
    """Aim angle from (sx, sy) toward where target will be when a fleet arrives.
    Returns (angle, intercept_x, intercept_y, travel_steps).
    """
    speed = fleet_speed(ships)
    orb = orbital_info[target]
    if orb is None:
        tx, ty = target.x, target.y
        travel = distance((sx, sy), (tx, ty)) / speed
    else:
        cx = cy = CENTER
        r, ia = orb
        # Seed: straight-line travel time to the planet's current position.
        travel = distance((sx, sy), (target.x, target.y)) / speed
        for _ in range(max_iters):
            a = ia + angular_velocity * (scene_step + travel - 0.5)
            new_tx, new_ty = cx + r * math.cos(a), cy + r * math.sin(a)
            new_travel = distance((sx, sy), (new_tx, new_ty)) / speed
            # Damp update: average old and new travel to suppress oscillation.
            new_travel = 0.5 * (travel + new_travel - 0.5)
            if abs(new_travel - travel) < tol:
                travel = new_travel
                break
            travel = new_travel
        # Recompute final position from converged travel so tx/ty/angle are consistent.
        a = ia + angular_velocity * (scene_step + travel - 0.5)
        tx, ty = cx + r * math.cos(a), cy + r * math.sin(a)
    angle = math.atan2(ty - sy, tx - sx)
    return angle, tx, ty, travel


def build_destination_list(
    fleets: list[Fleet], planets: list[Planet], orbital_info: OrbitalInfo,
    angular_velocity: float, scene_step: int,
) -> DestinationList:
    """For each fleet, find the first planet it is on an interception course for.
    Returns: dict mapping Planet -> list of (Fleet, t, arrival_x, arrival_y).
             t is continuous time in turns.
    """
    destination_list = defaultdict(list)
    for fleet in fleets:
        best = None
        best_t = float('inf')
        for planet in planets:
            needed_angle, px, py, travel = intercept_planet(
                fleet.x, fleet.y, planet, orbital_info, angular_velocity, scene_step, fleet.ships
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
            destination_list[planet].append((fleet, travel, px, py))
    return destination_list


def viz_destination_list(viz: Visualizer, scene_step: int, destination_list: DestinationList) -> None:
    """Draw a line from each fleet to its destination planet's arrival position."""
    lines = []
    for planet, arrivals in destination_list.items():
        for fleet, t, px, py in arrivals:
            color = '#44ff88' if fleet.owner == 0 else '#ff8844'
            viz.add_line(scene_step, fleet.x, fleet.y, px, py, color=color, width=1)
            lines.append(f' {fleet.owner} F{fleet.id} P{fleet.from_planet_id}({fleet.ships}) -> P{planet.id}({planet.ships}) t={round(t)} v={fleet_speed(fleet.ships):.2f}')
    if lines:
        viz.add_text(scene_step, 'dest_list:\n' + '\n'.join(lines))


def simulate_planet_timeline(planet: Planet, destination_list: DestinationList, player: int) -> tuple[int, float]:
    """Simulate planet ownership/production over time given a list of inbound fleets.
    destination_list: dict mapping Planet -> list of (Fleet, t, arrival_x, arrival_y).
    All arrivals at the same integer turn are resolved simultaneously (highest stack wins).
    Returns (final_owner, final_ships).
    """
    buckets = defaultdict(list)
    for fleet, t, _, _ in destination_list.get(planet, []):
        owner, ships = fleet.owner, fleet.ships
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
            if cur_owner == player:
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

    remaining = EVAL_HORIZON - cur_t

    return cur_owner, cur_ships


def evaluate_destinations(
    player: int, planets: list[Planet], proximity_graph: ProximityGraph,
    destination_list: DestinationList, orbital_info: OrbitalInfo,
    angular_velocity: float, scene_step: int,
) -> tuple[Planet | None, float, list[Any]]:
    """Score every reachable planet and pick the best destination. """

    best_planet = None
    best_score = -float('inf')
    best_orders = []

    for target in planets:
        is_owned = (target.owner == player)

        if is_owned:
            end_owner, end_ships = simulate_planet_timeline(target, destination_list, player)
            threatened = (end_owner != player)
            if not threatened:
                continue

        is_neutral = (target.owner == -1)


        # Accumulate ships from multiple sources (fastest-arriving first)

        # Don't attack neutral planets that can't be taken outright

        # Simulate with fleets added

    return best_planet, best_score, best_orders


def exec_snipe(
    scene_step: int, owned_planets: list[Planet], enemy_planets: list[Planet],
    moves: list[Any], orbital_info: OrbitalInfo, angular_velocity: float,
) -> None:
    best_travel = float('inf')
    best_mine = None
    best_enemy = None
    for mine in owned_planets:
        for enemy in enemy_planets:
            _, _, _, travel = intercept_planet(mine.x, mine.y, enemy, orbital_info, angular_velocity, scene_step, mine.ships)
            if travel < best_travel:
                best_travel = travel
                best_mine = mine
                best_enemy = enemy
    if best_mine is None or best_mine.ships < 10:
        reason = 'no owned planet' if best_mine is None else f'P{best_mine.id} ships={best_mine.ships} < 5'
        viz.add_text(scene_step, f'snipe: skip ({reason})')
        return
    ships = best_mine.ships
    angle, ix, iy, travel = intercept_planet(best_mine.x, best_mine.y, best_enemy, orbital_info, angular_velocity, scene_step, ships=ships)
    moves.append([best_mine.id, angle, ships])
    viz.add_text(scene_step, f'snipe: P{best_mine.id}({ships}sh) -> P{best_enemy.id}({best_enemy.ships}sh) angle={angle:.3f} travel={travel:.1f}t')
    viz.add_line(scene_step, best_mine.x, best_mine.y, ix, iy, color='#ff44ff', width=2)


def hellburner(obs: dict[str, Any]) -> list[Any]:
    viz.record(obs)
    _t0 = time.perf_counter()

    player = obs['player']
    scene_step = obs['step'] - 1
    angular_velocity = obs['angular_velocity']

    comet_ids = set(obs['comet_planet_ids'])
    planets_and_comets = [Planet(*p) for p in obs['planets']]
    planets = [p for p in planets_and_comets if p.id not in comet_ids]
    owned_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner != player]
    fleets = [Fleet(*f) for f in obs['fleets']]

    if not enemy_planets:
        return []

    orbital_info = build_orbital_info(planets, obs.get('initial_planets', []))
    proximity_graph, future_pos = build_proximity_graph(planets, orbital_info, angular_velocity, scene_step)
    destination_list = build_destination_list(fleets, planets, orbital_info, angular_velocity, scene_step)

    #evaluate_destinations(player, planets, proximity_graph, destination_list,
    #                       orbital_info, angular_velocity, scene_step)

    elapsed_ms = (time.perf_counter() - _t0) * 1000
    viz.add_text(scene_step, f'hellburner ms: {elapsed_ms:.2f}ms')
    #viz_proximity_graph(viz, scene_step, planets, proximity_graph, future_pos)
    viz_destination_list(viz, scene_step, destination_list)
    #viz_evaluate_destinations(viz, scene_step, scores_by_planet, best_planet, best_score,
    #                           best_orders, future_pos, orbital_info, angular_velocity)

    moves = []
    exec_snipe(scene_step, owned_planets, enemy_planets, moves, orbital_info, angular_velocity)
    return moves
