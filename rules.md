# Orbit Wars

Conquer planets rotating around a sun in continuous 2D space.

A real-time strategy game for 2 or 4 players. The player with the most total ships (on planets + in fleets) after 500 turns wins.

## Board Layout

- **Board:** 100x100 continuous space, origin at top-left.
- **Sun:** Centered at (50, 50) with radius 10. Fleets that cross the sun are destroyed.

## Planets

`[id, owner, x, y, radius, ships, production]`

| Index | Field | Type | Notes |
|---|---|---|---|
| 0 | `id` | int | Unique planet ID |
| 1 | `owner` | int | Player ID (0-3), or -1 for neutral |
| 2 | `x` | float | X position |
| 3 | `y` | float | Y position |
| 4 | `radius` | float | `1 + ln(production)` — larger planets produce more |
| 5 | `ships` | int | Current garrison; starts 5–99 (skewed low) |
| 6 | `production` | int | Ships generated per turn when owned (1–5) |

### Planet Types

- **Orbiting:** Planets whose `orbital_radius + planet_radius < 50` rotate around the sun at a constant angular velocity (0.025–0.05 rad/turn, randomized per game). Use `initial_planets` and `angular_velocity` to predict positions.
- **Static:** Planets further from the center do not rotate.

## Fleets

`[id, owner, x, y, angle, from_planet_id, ships]`

| Index | Field | Type | Notes |
|---|---|---|---|
| 0 | `id` | int | Unique fleet ID |
| 1 | `owner` | int | Player ID (0-3) |
| 2 | `x` | float | Current X position |
| 3 | `y` | float | Current Y position |
| 4 | `angle` | float | Direction of travel in radians |
| 5 | `from_planet_id` | int | Planet this fleet was launched from |
| 6 | `ships` | int | Number of ships (does not change during travel) |

### Fleet Speed

```python
speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5
```

- 1 ship moves at 1.0 units/turn; approaches max speed (default 6.0) at 1000 ships.
- ~500 ships ≈ speed 5.

### Fleet Movement

Fleets travel in a straight line. A fleet is removed if it goes out of bounds, crosses the sun, or collides with a planet (triggering combat). Collision detection is continuous — the full path segment is checked each turn.

## Comets

Comets spawn in groups of 4 (one per quadrant) at steps 50, 150, 250, 350, and 450. They follow highly elliptical orbits and are removed (with garrisoned ships) when they leave the board.

- **Radius:** 1.0. **Production:** 1 ship/turn when owned.
- **Starting ships:** Random, skewed low (min of 4 rolls from 1–99); all 4 in a group share the same count.
- **Speed:** `cometSpeed` (default 4.0 units/turn).
- Comets appear in `planets` and follow all normal rules (capture, production, launch, combat).
- Check `comet_planet_ids` to identify which planet IDs are comets.
- Comets are removed before fleet launches, so you cannot launch from a departing comet.
- `comets[].paths` gives the full trajectory; `comets[].path_index` gives the current position.

## Turn Order

1. **Comet expiration:** Remove comets that have left the board.
2. **Comet spawning:** Spawn new comet groups at designated steps.
3. **Fleet launch:** Process all player actions, creating new fleets.
4. **Production:** All owned planets (including comets) generate ships.
5. **Fleet movement:** Move all fleets; check out-of-bounds, sun, and planet collisions. Hits are queued for combat.
6. **Planet rotation & comet movement:** Orbiting planets rotate; comets advance. Fleets swept by a moving planet/comet are queued for combat.
7. **Combat resolution:** Resolve all queued combats.

## Combat

1. Arriving fleets are grouped by owner; ships from the same owner are summed.
2. The largest force fights the second largest; the difference survives.
3. If a surviving attacker is the **same owner** as the planet, ships are added to the garrison. If a **different owner**, survivors fight the garrison — if they exceed it the planet is captured and the surplus becomes the new garrison.
4. If two attackers tie, all attacking ships are destroyed.

## Scoring and Termination

The game ends at step 500, or when only one player (or zero) remains with planets or fleets.

**Final score** = ships on owned planets + ships in owned fleets. Highest score wins.

## Observation Reference

| Field | Type | Description |
|---|---|---|
| `planets` | list | Current planet states — see Planets schema |
| `fleets` | list | Active fleets — see Fleets schema |
| `player` | int | Your player ID (0-3) |
| `step` | int | Current turn number |
| `angular_velocity` | float | Planet rotation speed (radians/turn) |
| `initial_planets` | list | Planet positions at game start (same schema as `planets`) |
| `comets` | list | Active comet group data (`planet_ids`, `paths`, `path_index`) |
| `comet_planet_ids` | `[int, ...]` | Planet IDs that are comets |
| `next_fleet_id` | int | ID assigned to the next launched fleet |
| `remainingOverageTime` | float | Remaining overage time budget (seconds) |

## Action Format

Return a list of moves per turn. Each move: `[from_planet_id, direction_angle, num_ships]`.

| Field | Type | Description |
|---|---|---|
| `from_planet_id` | int | ID of a planet you own |
| `direction_angle` | float | Angle in radians (0 = right, π/2 = down) |
| `num_ships` | int | Ships to send (cannot exceed planet's current garrison) |

Return `[]` to take no action.
