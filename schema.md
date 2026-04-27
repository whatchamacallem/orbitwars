# Orbit Wars Observation Schema

## Top-level fields

| Field | Type | Description |
|---|---|---|
| `player` | int | Your player index (0-3) |
| `step` | int | Current turn number |
| `remainingOverageTime` | float | Seconds of overage time left |
| `angular_velocity` | float | Rotation speed of the map (radians/step) |
| `next_fleet_id` | int | ID that will be assigned to the next fleet launched |
| `planets` | list | Current planet states (see below) |
| `initial_planets` | list | Planet states at game start (same schema as planets) |
| `fleets` | list | In-flight fleets (see below) |
| `comets` | list | Active comets (see below) |
| `comet_planet_ids` | list[int] | Planet IDs reserved for comet landings |

---

## Planet `[id, owner, x, y, growth_rate, ships, size]`

| Index | Field | Type | Notes |
|---|---|---|---|
| 0 | `id` | int | Unique planet ID |
| 1 | `owner` | int | -1 = neutral, 0 = player 0, 1 = player 1 |
| 2 | `x` | float | X position (rotates each step) |
| 3 | `y` | float | Y position (rotates each step) |
| 4 | `growth_rate` | float | Ships produced per step when owned |
| 5 | `ships` | int | Current ship garrison |
| 6 | `size` | int | Planet size (affects conquest threshold?) |

Planets in `initial_planets` use the same schema but positions are fixed (pre-rotation).  
Comet placeholder planets start at `x=-99, y=-99` until the comet arrives.

---

## Fleet `[id, owner, x, y, angle, speed, ships]`

| Index | Field | Type | Notes |
|---|---|---|---|
| 0 | `id` | int | Unique fleet ID |
| 1 | `owner` | int | 0 or 1 |
| 2 | `x` | float | Current X position |
| 3 | `y` | float | Current Y position |
| 4 | `angle` | float | Heading in radians |
| 5 | `speed` | int | Units moved per step |
| 6 | `ships` | int | Ships in this fleet |

---

## Action format

Return a list of moves. Each move: `[planet_id, angle, ships]`

| Field | Type | Description |
|---|---|---|
| `planet_id` | int | ID of the planet launching the fleet |
| `angle` | float | Direction to send fleet (radians) |
| `ships` | int | Number of ships to send |

---

## Comet

| Field | Type | Description |
|---|---|---|
| `planet_ids` | list[int] | IDs of planets this comet will create/visit |
| `path_index` | int | Current position along the path |
| `paths` | list[list[[x,y]]] | One path per planet_id; sequence of [x,y] waypoints |

Comets move along their path each step, creating capturable planets as they go.
