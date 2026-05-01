# Hellburner v2: Warchest

## 1. Problem Statement

The current bot has two mutually exclusive planners:

| | `run_early_game` | `evaluate_move_orders` |
|---|---|---|
| **Horizon** | 30 turns lookahead | 0 turns (current state only) |
| **Search** | DFS with branch-and-bound | Greedy single-pass |
| **Sources per target** | One (single best source) | Many (accumulates by proximity) |
| **Enemy modeling** | None | In-flight fleets only, no prediction |
| **Defense** | None | Yes, via `simulate_planet_timeline` |
| **Economic scoring** | Yes (production × time) | No (production as flat value) |
| **Active turns** | Steps 0–49, 1v1 only | Steps 50+ |

The cutoff at `EARLY_ROUNDS=50` is arbitrary. We want a single planner that has:

- Multi-target sequencing (DFS) with multi-source attacks per target (synchronized arrival)
- Enemy fleet tracking and defensive response baked into the search
- Production-weighted economic scoring throughout the game
- A time-bounded search that degrades gracefully to the best plan found
- Correct comet handling: comets block fleet paths, own ships are drained from them, they are never targets

---

## 2. Module-Level Configuration

All tuning knobs live here. Nothing is hardcoded in algorithm bodies.

```python
# Search budget
TIME_BUDGET_S       = 0.800  # hard wall; return best-found-so-far if exceeded
DFS_EARLY_EXIT_S    = 0.080  # soft cutoff inside DFS loop (leaves room for emit/reinforce)

# Lookahead
UNIFIED_LOOK_AHEAD  = 25     # horizon = min(scene_step + UNIFIED_LOOK_AHEAD, 500)

# Candidate generation
MAX_CANDIDATES      = 6      # hard cap on DFS branching factor (reduce to 4 if timing spikes)

# Scoring
ENEMY_WEIGHT        = 0.8    # penalty multiplier for enemy production (tune toward 1.0)

# Post-pass thresholds (unchanged from baseline)
GARRISON_SIZE       = 10
REINFORCEMENT_SIZE  = 10
```

`TIME_BUDGET_S` is the wall-clock limit for the entire `run_unified_search` call.
`DFS_EARLY_EXIT_S` is checked inside the DFS recursion and must be < `TIME_BUDGET_S`;
the gap covers `warchest_emit_moves`, `send_reinforcements`, and `drain_comets`.

**No cross-round caching.** Every call to `main()` recomputes all data structures from
first principles: planet positions orbit each turn, so any intercept or garrison computed
last turn is stale. `build_orbital_info`, `build_proximity_graph`, `build_destination_list`,
and `warchest_initial_state` all run fresh each frame. This is by design and stays that way.

---

## 3. New Data Structures

### 3.1 `WarchestFleet`

```python
@dataclass(slots=True)
class WarchestFleet:
    owner: int
    destination_id: int
    fleet_size: float
    garrison_on_arrival: float   # estimated net ships after battle resolution
    arrival_turn: int
    is_capture: bool             # True if destination is not currently owned by owner
    production_id: int = -1      # planet whose production to add after capture (-1 = none)
```

Tracks friendly AND enemy fleets in the same structure so `warchest_advance` can resolve them in one pass.

### 3.2 `WarchestState`

```python
@dataclass(slots=True)
class WarchestState:
    turn: int
    garrison: dict[int, float]    # planet_id -> current ships on planet
    production: dict[int, float]  # planet_id -> ships/turn (non-comet planets only)
    ownership: dict[int, int]     # planet_id -> owner id (-1, 0, 1, ...)
    friendly_fleets: list[WarchestFleet]
    enemy_fleets: list[WarchestFleet]
    committed_ids: set[int]       # planet IDs already used as sources this planning pass
```

Key points:

- Tracks **enemy fleets** (sourced from `self.destination_list` at search init)
- Tracks **committed_ids** to prevent double-sending from the same source
- **ownership** dict replaces `owned` set (needed to model enemy captures too)

---

## 4. Algorithm: `run_unified_search()`

Replaces both `run_early_game` and the `while True: evaluate_move_orders()` loop.

### 4.1 Entry Point

```
1. Build candidates (attack targets; defense as optional candidate — see §10)
2. Construct initial WarchestState from current observation + destination_list
3. Run DFS with DFS_EARLY_EXIT_S budget
4. Emit moves whose launch_turn == scene_step
5. Run send_reinforcements() post-pass for uncommitted planets
6. Run drain_comets() post-pass for owned comets
```

### 4.2 Candidate Generation

**Offense candidates**: reachable enemy or neutral (non-comet) planets, pre-sorted by `initial_gain`:

```
initial_gain(p) = p.production × (horizon - earliest_capture_turn) - p.ships
```

Keep positive-gain candidates only, up to `MAX_CANDIDATES`. With branching factor ≤6 and DFS depth ≤4,
max nodes before pruning: 6×5×4×3 = 360. At ~0.3ms per node: ~108ms worst-case, heavily pruned in practice.
In theory, tight `warchest_upper_bound` pruning should keep actual timing well under 100ms.

**Defense candidates**: prepended to the list (not mandatory; see §10).

### 4.3 Initial State Construction

```python
def warchest_initial_state(self) -> WarchestState:
    garrison  = {p.id: float(p.ships)      for p in self.planets}
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
```

### 4.4 DFS Core

```python
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
```

---

## 5. State Operations

### 5.1 `warchest_advance(state, to_turn)` — Turn Order

Production is applied **before** combat each turn, matching the rules turn order
(Production → Fleet Movement → Combat):

```python
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
```

### 5.2 `warchest_resolve_battle(state, planet_id, arrivals)`

Two-stage resolution matching the actual rules. The garrison is **not** part of the initial
fleet fight — only arriving fleets duel each other; the survivor then fights the garrison.

```python
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
```

### 5.3 `warchest_execute(state, target, assignment)` — SYNCHRONIZED ARRIVAL

All sources are timed to arrive at the **same turn** (the latest individual arrival).
Earlier sources delay their launch; during the wait they accumulate production.

**Turn-order correctness**: ships are deducted **before** production runs for the launch turn.
`warchest_advance` applies production then resolves fleets for each turn. To avoid the launch
turn's production being credited before the deduction, `warchest_execute` first advances to
`launch_turn - 1`, then deducts ships, then calls `warchest_advance` for `launch_turn` itself
(which applies production and resolves any arrivals at that turn). This correctly models
"ships leave at the start of the turn, production runs after":

```python
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
```

After `warchest_execute` returns, `state.turn == latest_arrival`. The DFS continues
planning from the arrival turn, which means follow-on attacks can use the captured
planet as a source and the scoring function sees the correct post-capture ownership.

**Depth impact**: with `UNIFIED_LOOK_AHEAD=25` and typical travel of 6–10 turns,
advancing to `latest_arrival` leaves 15–19 turns of horizon after the first capture.
A second capture at 6–10 turns later still leaves horizon for a third level. DFS depth
remains 2–3 in practice, which is sufficient. If timing analysis shows depth 1 dominates,
reduce `UNIFIED_LOOK_AHEAD` before reducing `MAX_CANDIDATES`.

**Synchronized sources with different launch turns**: `warchest_execute` processes sources
in ascending `launch_turn` order. Each source advances the shared state to `launch_turn - 1`,
deducts ships, then advances through `launch_turn`. Because state is shared and advanced
incrementally, production and enemy arrivals between source launches are applied exactly once
in chronological order. This is correct: a source launching at t=8 sees the garrison as it
actually stands after production and combat through t=8.

### 5.4 `warchest_copy(state)` — Optimized State Copy

```python
def warchest_copy(state: WarchestState) -> WarchestState:
    return WarchestState(
        turn=state.turn,
        garrison=dict(state.garrison),
        production=state.production,              # read-only during search; share reference
        ownership=dict(state.ownership),
        friendly_fleets=list(state.friendly_fleets),  # WarchestFleet is immutable
        enemy_fleets=state.enemy_fleets,              # never mutated during search; share reference
        committed_ids=set(state.committed_ids),
    )
```

`production` and `enemy_fleets` are never mutated during DFS — reference-share them.
Only `garrison`, `ownership`, `friendly_fleets`, and `committed_ids` need new containers.

**Why `enemy_fleets` sharing is safe**: `warchest_advance` filters resolved fleets by
reassigning `state.enemy_fleets = remaining_enemy` (a new list), not by mutating the existing
list in place. Each copy therefore gets its own rebind without touching the shared original.
Do **not** change this to an in-place `.remove()` or `.pop()` — that would corrupt sibling DFS branches.

---

## 6. Multi-Source Fleet Assignment (`warchest_assign_fleet`) — Synchronized

### 6.1 Overview

Returns `{source_id: (ships, launch_turn, arrival_turn)}` or `{}` if the target is unwinnable.

All returned entries share the **same `arrival_turn`** (synchronized). Earlier-arriving sources
receive a delayed `launch_turn` so they arrive simultaneously with the slowest source.
During the delay they accumulate production.

**Key correctness invariants:**

- Phase 1 threshold uses the **projected arrival garrison** (current + production × travel), not the static current garrison. Using the static garrison causes committed ship counts that are too low, producing captured-at-0-garrison fleet records that `warchest_score` falsely credits with full production income.
- A **second-enemy-arrival guard** is enforced on neutral targets: if enemy fleets arrive at or after our fleet, we'd fight the battle winner rather than the original garrison. We return `{}` and let the DFS skip this target.
- Phase 2 projected garrison simulation accounts for **enemy fleet arrivals at the source** during the delay, not just raw production.

### 6.2 Implementation

```python
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
        min_travel = candidates[0][1]  # inbound_edges already sorted closest-first
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
    # resolve arrivals off by one turn (mirrors the existing hellburner.py check at line 456).
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
        if self._warchest_source_would_be_lost(state, src_id, ships, launch_turn):
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


def _warchest_source_would_be_lost(
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
```

### 6.3 Defense Mode

When `warchest_assign_fleet` is called on an **owned** planet (defense candidate from `warchest_candidates`),
the same logic applies: "target_garrison" is the net enemy force arriving, sources are other
owned planets within range, the fleet reinforces (is_capture=False). The scoring function
naturally rewards keeping the planet.

---

## 7. Performance Model: No Cache, Aggressive Pruning

Every turn recomputes from scratch — no cross-round cache of any kind. Planet positions change
each turn, so cached intercepts are stale. With `MAX_CANDIDATES=6`, the DFS makes ~100–300
`intercept_planet` calls per turn. Each call is ~0.05ms; total intercept cost ≈ 5–15ms.
Tight `warchest_upper_bound` pruning should keep total DFS time well under 100ms.
`DFS_EARLY_EXIT_S` (default 0.080) serves as a safety net, not the expected exit path.

**Staying within budget — primary control levers (in order):**

| Lever | Effect | When to use |
|---|---|---|
| `MAX_CANDIDATES` | Hard cap on DFS branching | Default 6; reduce to 4 if timing spikes |
| `warchest_upper_bound` tightness | Prunes branches early | Default tight bound (§8.2) |
| `UNIFIED_LOOK_AHEAD` | Less `warchest_advance` work per node | Reduce if still slow |
| `DFS_EARLY_EXIT_S` | Returns best-found-so-far | Always present as safety net |

If a turn exceeds 100ms, tighten pruning (fewer candidates or tighter horizon), not caching.

---

## 8. Scoring Function

### 8.1 `warchest_score(state, horizon)`

Scores without advancing state (approximation). No double-counting: `garrison` is the current
state; `friendly_fleets` contains future arrivals not yet applied to garrisons.

```python
def warchest_score(self, state: WarchestState, horizon: int) -> float:
    remaining = max(0, horizon - state.turn)
    total = 0.0

    for pid, owner in state.ownership.items():
        prod    = state.production.get(pid, 0)
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
```

`ENEMY_WEIGHT = 0.8`. The game is won by **total ships at step 500**, so every ship the enemy
produces is a ship you don't have. Capturing an enemy planet is a 2× swing (you gain income,
they lose income), meaning the theoretically correct weight is 1.0. Start at 0.8 to avoid
over-aggressiveness against strongly defended planets; tune up toward 1.0 if the bot
under-attacks enemies and wastes turns on neutrals instead.

### 8.2 `warchest_upper_bound(state, remaining, horizon)`

Subtracts the capture cost (garrison ships spent), caps by available ship budget, and counts
production only from the **estimated arrival turn** (not from `state.turn`). Counting from
`state.turn` overestimates each capture by up to `production × travel_turns`, which with
`MAX_CANDIDATES=8` and travel of 8 turns could add phantom ship headroom, preventing
almost all pruning.

```python
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
        # Production benefit starts at capture, not now: use closest reachable owned source
        # to estimate arrival turn.
        min_travel = min(
            (t for src, t in self.inbound_edges.get(p, [])
             if state.ownership.get(src.id) == self.player),
            default=float(horizon - state.turn),
        )
        production_turns = max(0, horizon - state.turn - math.ceil(min_travel))
        net_gain = p.production * production_turns - capture_cost
        if net_gain > 0:
            bound      += net_gain
            ship_budget -= capture_cost

    return bound
```

---

## 9. Enemy Fleet Modeling

### 9.1 Known Enemy Fleets (from `destination_list`)

Loaded into `WarchestState.enemy_fleets` at init. `warchest_advance` resolves them
alongside friendly fleets. This is the only source of enemy fleet information.

### 9.2 No Enemy Prediction

Predicting attacks that don't materialize wastes defensive ships. Use only confirmed
in-flight enemy fleets from `destination_list`.

### 9.3 Second-Enemy-Arrival Guard

When a neutral planet has enemy fleets inbound, we must not arrive at or within 1 turn after
the last enemy arrival. If we do, we risk fighting the enemy battle winner (who may have more
ships than the original garrison) rather than the static garrison we planned against. The +1
turn buffer accounts for the engine's swept-collision handling, which can resolve fleet
arrivals off by one turn (the same tolerance used in the existing `evaluate_frontline_strategy`
check: `math.ceil(travel) <= second_enemy_arrival + 1`).

**Implemented in `warchest_assign_fleet` (§6.2)**: after Phase 1 computes `latest_arrival`,
if the target is neutral (`ownership == -1`) and any enemy fleet arrives at `>= latest_arrival - 1`,
the function returns `{}`. The DFS naturally skips this target or tries it later once the
timing is safe.

---

## 10. Defense Strategy: Offense Is the Best Defense

Defense is **not** a mandatory pre-pass. It enters the DFS as a regular candidate.

**Rationale:** production capacity is the only thing that matters over 500 turns. If the cost
to defend a planet is greater than the production gain from attacking an enemy planet instead,
attacking is the correct choice. The `warchest_score` function makes this comparison
automatically — losing a planet drops garrison + prod × remaining; capturing an enemy planet
adds prod × remaining and removes ENEMY_WEIGHT × their production. The DFS explores both
and picks the highest score.

### 10.1 Defense Candidate Detection

Owned planets under known enemy attack are prepended to the candidate list to improve DFS
ordering (explored first). They are still subject to the normal DFS pruning.

```python
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
        ct = self._warchest_earliest_capture(initial, p, horizon)
        if not math.isfinite(ct):
            continue
        gain = p.production * (horizon - ct) - p.ships
        if gain > 0:
            offense.append((p, gain))

    offense.sort(key=lambda x: x[1], reverse=True)
    return (defense + [p for p, _ in offense])[:MAX_CANDIDATES]


def _warchest_earliest_capture(
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
```

---

## 11. Comet Handling

Comets are **not** strategic targets. They are temporary bodies with production=1 that expire
mid-game. Capturing a comet is wasted effort. However:

1. **Path blocking**: comets occupy space and can intercept fleet trajectories.
   `first_planet_hit` must check comets so we don't accidentally send a fleet into one.
2. **Owned comet draining**: if we own a comet, drain its ships to a nearby non-comet planet
   before it expires.

### 11.1 Body Sets

```python
# In main():
comet_ids = set(obs['comet_planet_ids'])
self.all_bodies  = [HPlanet(*p) for p in obs['planets']]   # all bodies, for path-blocking
self.planets     = [p for p in self.all_bodies if p.id not in comet_ids]
self.owned_comets = [p for p in self.all_bodies
                     if p.id in comet_ids and p.owner == self.player]
self.comet_ids   = comet_ids  # store for filtering in warchest_initial_state / warchest_candidates
```

### 11.2 Comet Orbital Info

Comets follow **elliptical waypoint paths**, not circular orbits. `intercept_planet` and
`orbital_info` assume circular orbits. `build_orbital_info` must iterate `self.all_bodies`
and register comets as static bodies (`None`) — treating them at their current position is a
conservative approximation sufficient for collision avoidance:

```python
def build_orbital_info(self, initial_planets: list[Any]) -> None:
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
```

### 11.3 Path Collision Fix

`first_planet_hit` and `build_destination_list` must iterate `self.all_bodies` so that
fleet paths crossing a comet are correctly detected:

```python
def first_planet_hit(self, sx, sy, angle, ships, source):
    for planet in self.all_bodies:   # ← was self.planets; CRITICAL: comets must block paths
        ...
```

`build_destination_list` likewise iterates `self.all_bodies` for the inner planet loop
so incoming fleets that hit a comet are correctly tracked (then filtered out in
`warchest_initial_state` via `if dest_planet.id in self.comet_ids: continue`).

**Implementation note**: Both changes are easy to miss because `self.planets` is used
everywhere else. The current `hellburner.py` has the bug at lines 282 and 311 — both
must be updated before any warchest code is added.

### 11.4 Owned Comet Drain Post-Pass

```python
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
```

---

## 12. Reinforcement Post-Pass

`send_reinforcements()` is kept as-is as a post-pass. Filter out planets whose source ID
appears in `self._warchest_committed_ids` so we don't double-move from a planet that already sent
ships via the DFS search.

DFS commits entire planets (all ships sent in one move per source), so the committed_ids
filter is sufficient to prevent double-sends.

---

## 13. Move Emission (`warchest_emit_moves`)

Only emit moves with `launch_turn == scene_step`. Uses live `intercept_planet` and
`first_planet_hit` (no cache lookups):

```python
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
```

**Deferred moves**: a launch at `t+3` is not stored. The next turn's DFS rediscovers it
because the same state analysis produces the same assignment (given no new enemy fleets).
Re-planning from scratch is always correct.

---

## 14. Updated `main()`

```python
def main(self, obs: dict) -> list:
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

    self.build_orbital_info(obs.get('initial_planets', []))
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
    viz.add_text(self.scene_step, f'TES ms: {elapsed_ms:.2f}ms')
    return moves
```

---

## 15. Performance Budget Analysis

| Operation | Est. Time | Notes |
|-----------|-----------|-------|
| `build_*` helpers | ~5ms | Unchanged from baseline |
| `warchest_initial_state` | <1ms | Dict construction |
| `warchest_candidates` (trial advances) | ~2ms | O(owned_planets × enemy_fleets) |
| `warchest_assign_fleet` per call | ~0.3ms | `intercept_planet` × ≤5 sources |
| DFS (≤360 nodes, pruned to ~60–120) | ~20–40ms | Dominates; tighten candidates if needed |
| `send_reinforcements` | <1ms | Unchanged |
| `drain_comets` | <1ms | O(owned_comets × planets) |
| **Total** | **~30–50ms** | Target <100ms; hard wall `TIME_BUDGET_S=0.800` |

---

## 16. What to Keep Unchanged

- `HPlanet`, `fleet_speed`, `intercept_planet` — unchanged
- `first_planet_hit` — **MUST** change `self.planets` → `self.all_bodies` (§11.3); easy to miss
- `build_orbital_info` — iterate `self.all_bodies`; register comets as `None` (static) (§11.2)
- `build_proximity_graph` — unchanged (operates on `self.planets` only; comets not in graph)
- `build_destination_list` — **MUST** change inner planet loop to `self.all_bodies` (§11.3); easy to miss
- `simulate_planet_timeline` — kept; used by `send_reinforcements` indirectly; DFS uses its own `warchest_advance`/`warchest_resolve_battle`
- `build_reinforcement_targets`, `send_reinforcements` — unchanged
- All `viz_*` methods — unchanged

---

## 17. Implementation Order

1. Update `main()`: add `all_bodies`, `owned_comets`, `comet_ids`; remove old branching
2. Update `build_orbital_info` to iterate `self.all_bodies` and register comets as `None` (§11.2)
3. Update `first_planet_hit` and `build_destination_list` to iterate `self.all_bodies` (§11.3)
4. Add module-level constants: `TIME_BUDGET_S`, `DFS_EARLY_EXIT_S`, `UNIFIED_LOOK_AHEAD`, `MAX_CANDIDATES`, `ENEMY_WEIGHT`, `GARRISON_SIZE`, `REINFORCEMENT_SIZE`
5. Add `WarchestFleet`, `WarchestState` dataclasses; add `self._warchest_committed_ids: set[int] = set()` to `__init__`
6. Implement `warchest_copy`
7. Implement `warchest_advance` (production-first order)
8. Implement `warchest_resolve_battle` (two-stage: fleet-fight then garrison-fight)
9. Implement `warchest_initial_state` (skip comet destinations; project garrison to arrival time)
10. Implement `_warchest_source_would_be_lost` (simultaneous arrivals grouped by turn)
11. Implement `warchest_assign_fleet` (arrival-garrison threshold; second-enemy-arrival guard on neutral targets; Phase 2 enemy-damage projection; synchronized arrival; first_planet_hit re-check + drift enforcement; safety-checked; excess-trimmed vs arrival garrison)
12. Implement `warchest_execute` (advance-to-launch-minus-1 then deduct then advance-through-launch; arrival-time garrison projection; early-return on garrison_on_arrival ≤ 0; advance to latest_arrival at end)
13. Implement `warchest_score` (with enemy in-flight attack penalty), `warchest_upper_bound` (arrival-turn production estimate; ship-budget cap)
14. Implement `_warchest_earliest_capture` (single-source + two-source fallback), `warchest_candidates` (defense prepended, offense sorted)
15. Implement `warchest_emit_moves` (live intercept, no cache)
16. Implement `warchest_dfs` and `run_unified_search`
17. Implement `drain_comets`
18. Wire into `main()`, remove `run_early_game` / `evaluate_move_orders`
19. Add viz hooks: candidates list, assignment per node, score at each node, timing
20. Tune `UNIFIED_LOOK_AHEAD`, `ENEMY_WEIGHT`, `MAX_CANDIDATES` empirically vs. baseline
