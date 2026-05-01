"""
Regression tests for hellburner.py — run with: python3 hellburner_tests.py
Covers run_unified_search() and everything it calls.
"""
import math
import sys
import time
import unittest

sys.path.insert(0, '/home/t/orbitwars')

from hellburner import (
    Hellburner, WarchestFleet, WarchestState,
    warchest_copy, fleet_speed, UNIFIED_LOOK_AHEAD, LOOK_AHEAD,
    MAX_DISTANCE, GARRISON_SIZE, REINFORCEMENT_SIZE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_obs(
    planets,
    initial_planets=None,
    fleets=None,
    comet_ids=None,
    player=0,
    step=1,
    angular_velocity=0.0,
):
    """Build a minimal obs dict. initial_planets defaults to planets."""
    return {
        'player': player,
        'step': step,
        'angular_velocity': angular_velocity,
        'planets': planets,
        'initial_planets': initial_planets if initial_planets is not None else planets,
        'fleets': fleets or [],
        'comet_planet_ids': comet_ids or [],
        'comets': [],
        'next_fleet_id': len(fleets or []),
        'remainingOverageTime': 60.0,
    }


def run(obs):
    """Construct a fresh Hellburner and call main()."""
    h = Hellburner()
    moves = h.main(obs)
    return h, moves


def make_warchest_state(garrison, production, ownership,
                        friendly=None, enemy=None, turn=0):
    return WarchestState(
        turn=turn,
        garrison=dict(garrison),
        production=dict(production),
        ownership=dict(ownership),
        friendly_fleets=list(friendly or []),
        enemy_fleets=list(enemy or []),
        committed_ids=set(),
    )


# ---------------------------------------------------------------------------
# 1. fleet_speed
# ---------------------------------------------------------------------------

class TestFleetSpeed(unittest.TestCase):
    def test_one_ship_minimum_speed(self):
        """A fleet of 1 ship travels at speed 1.0 (the minimum per the formula)."""
        self.assertAlmostEqual(fleet_speed(1), 1.0)

    def test_1000_ships_maximum_speed(self):
        """A fleet of 1000 ships hits the configured maximum speed of 6.0."""
        self.assertAlmostEqual(fleet_speed(1000), 6.0)

    def test_100_ships_intermediate(self):
        """100 ships produces a known intermediate speed value (~3.72)."""
        self.assertAlmostEqual(fleet_speed(100), 3.721655, places=4)

    def test_speed_increases_with_fleet_size(self):
        """Larger fleets are strictly faster than smaller ones."""
        self.assertLess(fleet_speed(10), fleet_speed(100))
        self.assertLess(fleet_speed(100), fleet_speed(1000))


# ---------------------------------------------------------------------------
# 2. intercept_planet
# ---------------------------------------------------------------------------

class TestInterceptPlanet(unittest.TestCase):
    def setUp(self):
        # p0 at (15,30) and p1 at (15,60): both static (r≈49, r+2>50),
        # distance=30, no sun obstruction along the vertical left wall.
        obs = make_obs(
            [[0, 0, 15.0, 30.0, 2.0, 100, 3],
             [1, 1, 15.0, 60.0, 2.0, 20, 2]],
        )
        self.h, _ = run(obs)
        self.p0, self.p1 = self.h.planets[0], self.h.planets[1]

    def test_static_intercept_angle(self):
        """Shooting from p0 straight up at a static p1 directly above returns angle π/2."""
        angle, ix, iy, t = self.h.intercept_planet(
            self.p0.x, self.p0.y, self.p1, 100
        )
        self.assertAlmostEqual(angle, math.pi / 2, places=4)

    def test_static_intercept_position_matches_target(self):
        """The predicted intercept position equals the static target's position."""
        angle, ix, iy, t = self.h.intercept_planet(
            self.p0.x, self.p0.y, self.p1, 100
        )
        self.assertAlmostEqual(ix, 15.0, places=2)
        self.assertAlmostEqual(iy, 60.0, places=2)

    def test_static_intercept_travel_positive(self):
        """Travel time to a reachable static planet is finite and positive."""
        _, _, _, t = self.h.intercept_planet(
            self.p0.x, self.p0.y, self.p1, 100
        )
        self.assertGreater(t, 0)
        self.assertTrue(math.isfinite(t))

    def test_travel_decreases_with_more_ships(self):
        """A larger fleet reaches the same target in fewer turns because it travels faster."""
        _, _, _, t_small = self.h.intercept_planet(
            self.p0.x, self.p0.y, self.p1, 1
        )
        _, _, _, t_large = self.h.intercept_planet(
            self.p0.x, self.p0.y, self.p1, 1000
        )
        self.assertGreater(t_small, t_large)


# ---------------------------------------------------------------------------
# 3. first_planet_hit
# ---------------------------------------------------------------------------

class TestFirstPlanetHit(unittest.TestCase):
    def setUp(self):
        obs = make_obs(
            [[0, 0, 15.0, 30.0, 2.0, 100, 3],
             [1, 1, 15.0, 60.0, 2.0, 20, 2]],
        )
        self.h, _ = run(obs)
        self.p0, self.p1 = self.h.planets[0], self.h.planets[1]

    def test_clear_shot_hits_target(self):
        """A fleet aimed directly at p1 with no obstacles returns p1 as the first hit."""
        angle, _, _, _ = self.h.intercept_planet(
            self.p0.x, self.p0.y, self.p1, 100
        )
        hit = self.h.first_planet_hit(self.p0.x, self.p0.y, angle, 100, self.p0)
        self.assertIs(hit, self.p1)

    def test_sun_blocking_returns_none(self):
        """A shot fired east along y=50 crosses the sun at (50,50) and returns None."""
        hit = self.h.first_planet_hit(30.0, 50.0, 0.0, 100, self.p0)
        self.assertIsNone(hit)

    def test_comet_blocks_path(self):
        """A comet sitting between the source and target intercepts the fleet before the target."""
        obs = make_obs(
            [[0, 0, 15.0, 30.0, 2.0, 100, 3],
             [1, 1, 15.0, 60.0, 2.0, 20, 2],
             [2, 0, 15.0, 45.0, 1.0, 5, 1]],
            comet_ids=[2],
        )
        h, _ = run(obs)
        angle, _, _, _ = h.intercept_planet(
            h.planets[0].x, h.planets[0].y, h.planets[1], 100
        )
        hit = h.first_planet_hit(
            h.planets[0].x, h.planets[0].y, angle, 100, h.planets[0]
        )
        self.assertIsNotNone(hit)
        self.assertIsNot(hit, h.planets[1])


# ---------------------------------------------------------------------------
# 4. build_orbital_info
# ---------------------------------------------------------------------------

class TestBuildOrbitalInfo(unittest.TestCase):
    def test_outer_planet_is_static(self):
        """A planet at (2,50) has orbital_radius=48, 48+2=50 which is not < ROTATION_RADIUS_LIMIT,
        so it is classified as static (orbital_info entry is None)."""
        obs = make_obs([[0, 0, 2.0, 50.0, 2.0, 50, 3],
                        [1, 1, 35.0, 50.0, 2.0, 20, 2]],
                       angular_velocity=0.03)
        h, _ = run(obs)
        self.assertIsNone(h.orbital_info[h.planets[0]])

    def test_inner_planet_orbits(self):
        """A planet at (35,50) has orbital_radius=15, 15+2=17 < 50, so it orbits and
        orbital_info stores (radius=15, initial_angle)."""
        obs = make_obs([[0, 0, 2.0, 50.0, 2.0, 50, 3],
                        [1, 1, 35.0, 50.0, 2.0, 20, 2]],
                       angular_velocity=0.03)
        h, _ = run(obs)
        orb = h.orbital_info[h.planets[1]]
        self.assertIsNotNone(orb)
        r, ia = orb
        self.assertAlmostEqual(r, 15.0, places=4)

    def test_comet_registered_as_static(self):
        """Comets follow elliptical paths so they are always registered as static (None)
        regardless of their distance from the centre."""
        obs = make_obs([[0, 0, 2.0, 50.0, 2.0, 100, 3],
                        [1, 1, 30.0, 50.0, 2.0, 20, 2],
                        [2, 0, 35.0, 35.0, 1.0, 10, 1]],
                       comet_ids=[2], angular_velocity=0.03)
        h, _ = run(obs)
        comet = next(b for b in h.all_bodies if b.id == 2)
        self.assertIsNone(h.orbital_info[comet])


# ---------------------------------------------------------------------------
# 5. build_proximity_graph
# ---------------------------------------------------------------------------

class TestBuildProximityGraph(unittest.TestCase):
    def test_planets_within_range_have_edges(self):
        """Two planets separated by 30 units (< MAX_DISTANCE=35) appear in each other's
        inbound_edges."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 50, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        p0, p1 = h.planets[0], h.planets[1]
        self.assertTrue(any(src is p0 for src, _ in h.inbound_edges[p1]))
        self.assertTrue(any(src is p1 for src, _ in h.inbound_edges[p0]))

    def test_planets_out_of_range_have_no_edges(self):
        """Two planets 88 units apart (> MAX_DISTANCE=35) have empty inbound_edges."""
        obs = make_obs([[0, 0, 2.0, 50.0, 2.0, 100, 3],
                        [1, 1, 90.0, 50.0, 2.0, 20, 2]])
        h, _ = run(obs)
        p0, p1 = h.planets[0], h.planets[1]
        self.assertEqual(h.inbound_edges[p0], [])
        self.assertEqual(h.inbound_edges[p1], [])

    def test_outbound_is_complement_of_inbound(self):
        """If p1 is reachable from p0, then p1 appears in p0's outbound_edges."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 50, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        p0, p1 = h.planets[0], h.planets[1]
        self.assertTrue(any(dst is p1 for dst, _ in h.outbound_edges[p0]))

    def test_comets_excluded_from_planet_graph(self):
        """Comets are not nodes in inbound_edges; they only act as path obstacles."""
        obs = make_obs([[0, 0, 2.0, 50.0, 2.0, 100, 3],
                        [1, 1, 30.0, 50.0, 2.0, 20, 2],
                        [2, 0, 15.0, 25.0, 1.0, 10, 1]],
                       comet_ids=[2])
        h, _ = run(obs)
        comet_ids_in_graph = {p.id for p in h.inbound_edges}
        self.assertNotIn(2, comet_ids_in_graph)


# ---------------------------------------------------------------------------
# 6. warchest_advance / warchest_resolve_battle
# ---------------------------------------------------------------------------

class TestWarchestAdvance(unittest.TestCase):
    def _make_h(self):
        h = Hellburner()
        h.player = 0
        return h

    def test_production_accumulates(self):
        """Each owned planet earns production ships every turn; after 4 turns garrison
        increases by production * 4."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 10.0}, production={0: 3.0},
            ownership={0: 0},
        )
        h.warchest_advance(state, 4)
        self.assertAlmostEqual(state.garrison[0], 10.0 + 3.0 * 4)

    def test_friendly_fleet_captures_enemy(self):
        """A friendly fleet of 35 arriving at t=5 versus enemy garrison of 20 + 2*5=30
        wins by 5, captures the planet, and leaves garrison=5."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 50.0, 1: 20.0},
            production={0: 3.0, 1: 2.0},
            ownership={0: 0, 1: 1},
            friendly=[WarchestFleet(owner=0, destination_id=1, fleet_size=35.0,
                                    garrison_on_arrival=0.0, arrival_turn=5,
                                    is_capture=True, production_id=1)],
        )
        h.warchest_advance(state, 5)
        self.assertEqual(state.ownership[1], 0)
        self.assertAlmostEqual(state.garrison[1], 5.0)
        self.assertAlmostEqual(state.garrison[0], 50.0 + 3.0 * 5)

    def test_exact_tie_defender_keeps_with_zero(self):
        """Fleet=30 vs garrison 20+2*5=30 is an exact tie; per the rules the defender
        keeps the planet with 0 garrison."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 50.0, 1: 20.0},
            production={0: 3.0, 1: 2.0},
            ownership={0: 0, 1: 1},
            friendly=[WarchestFleet(owner=0, destination_id=1, fleet_size=30.0,
                                    garrison_on_arrival=0.0, arrival_turn=5,
                                    is_capture=True, production_id=1)],
        )
        h.warchest_advance(state, 5)
        self.assertEqual(state.ownership[1], 1)   # defender keeps
        self.assertAlmostEqual(state.garrison[1], 0.0)

    def test_enemy_captures_our_planet(self):
        """An enemy fleet of 40 arriving at t=5 versus our garrison 20+2*5=30 captures
        the planet and leaves 10 ships for the enemy."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 20.0, 1: 50.0},
            production={0: 2.0, 1: 3.0},
            ownership={0: 0, 1: 1},
            enemy=[WarchestFleet(owner=1, destination_id=0, fleet_size=40.0,
                                 garrison_on_arrival=0.0, arrival_turn=5,
                                 is_capture=True, production_id=0)],
        )
        h.warchest_advance(state, 5)
        self.assertEqual(state.ownership[0], 1)
        self.assertAlmostEqual(state.garrison[0], 10.0)

    def test_advance_noop_to_same_turn(self):
        """Advancing to the current turn (to_turn == state.turn) makes no changes."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 20.0}, production={0: 3.0}, ownership={0: 0},
        )
        h.warchest_advance(state, 0)
        self.assertAlmostEqual(state.garrison[0], 20.0)

    def test_reinforcement_same_owner_adds_ships(self):
        """A friendly fleet arriving at a planet we already own adds ships to the
        garrison without changing ownership."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 10.0, 1: 20.0},
            production={0: 0.0, 1: 0.0},
            ownership={0: 0, 1: 0},
            friendly=[WarchestFleet(owner=0, destination_id=1, fleet_size=15.0,
                                    garrison_on_arrival=15.0, arrival_turn=3,
                                    is_capture=False, production_id=-1)],
        )
        h.warchest_advance(state, 3)
        self.assertAlmostEqual(state.garrison[1], 35.0)
        self.assertEqual(state.ownership[1], 0)


# ---------------------------------------------------------------------------
# 7. warchest_score
# ---------------------------------------------------------------------------

class TestWarchestScore(unittest.TestCase):
    def _make_h(self):
        h = Hellburner()
        h.player = 0
        return h

    def test_owned_planet_scores_garrison_plus_future_production(self):
        """Score for a single owned planet is garrison + production * remaining_turns.
        With garrison=50, production=3, horizon=25: 50 + 3*25 = 125."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 50.0}, production={0: 3.0}, ownership={0: 0},
        )
        score = h.warchest_score(state, 25)
        self.assertAlmostEqual(score, 125.0)

    def test_enemy_planet_penalised(self):
        """Enemy planets are subtracted at ENEMY_WEIGHT=0.8.
        owned: 50+3*25=125; enemy: -(20+2*25)*0.8=−56; total=69."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 50.0, 1: 20.0},
            production={0: 3.0, 1: 2.0},
            ownership={0: 0, 1: 1},
        )
        score = h.warchest_score(state, 25)
        self.assertAlmostEqual(score, 69.0)

    def test_enemy_inbound_fleet_penalises_score(self):
        """An enemy fleet large enough to capture our planet reduces the score below
        the score of the same state without the inbound threat."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 10.0}, production={0: 2.0}, ownership={0: 0},
            enemy=[WarchestFleet(owner=1, destination_id=0, fleet_size=100.0,
                                 garrison_on_arrival=0.0, arrival_turn=5,
                                 is_capture=True, production_id=0)],
        )
        score_with_threat = h.warchest_score(state, 25)
        state_clean = make_warchest_state(
            garrison={0: 10.0}, production={0: 2.0}, ownership={0: 0},
        )
        score_clean = h.warchest_score(state_clean, 25)
        self.assertLess(score_with_threat, score_clean)

    def test_score_increases_after_capture(self):
        """Capturing an enemy planet raises the score because we gain its production
        and the enemy penalty disappears."""
        h = self._make_h()
        before = make_warchest_state(
            garrison={0: 50.0, 1: 20.0},
            production={0: 3.0, 1: 2.0},
            ownership={0: 0, 1: 1},
        )
        after = make_warchest_state(
            garrison={0: 20.0, 1: 5.0},
            production={0: 3.0, 1: 2.0},
            ownership={0: 0, 1: 0},
        )
        self.assertGreater(
            h.warchest_score(after, 25),
            h.warchest_score(before, 25),
        )


# ---------------------------------------------------------------------------
# 8. warchest_source_would_be_lost
# ---------------------------------------------------------------------------

class TestSourceWouldBeLost(unittest.TestCase):
    def _make_h(self):
        h = Hellburner()
        h.player = 0
        return h

    def test_lost_when_garrison_insufficient(self):
        """After deducting 5 ships to send, remaining garrison is 15; with production 2/turn
        it reaches 21 at t=3, which is less than the enemy fleet of 25, so the planet is lost."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 20.0}, production={0: 2.0}, ownership={0: 0},
            enemy=[WarchestFleet(owner=1, destination_id=0, fleet_size=25.0,
                                 garrison_on_arrival=0.0, arrival_turn=3,
                                 is_capture=True, production_id=0)],
        )
        self.assertTrue(h.warchest_source_would_be_lost(state, 0, 5.0, 5))

    def test_not_lost_when_garrison_survives(self):
        """Sending 0 ships leaves garrison at 20; with production it reaches 26 at t=3,
        which beats the enemy fleet of 25, so the planet survives."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 20.0}, production={0: 2.0}, ownership={0: 0},
            enemy=[WarchestFleet(owner=1, destination_id=0, fleet_size=25.0,
                                 garrison_on_arrival=0.0, arrival_turn=3,
                                 is_capture=True, production_id=0)],
        )
        self.assertFalse(h.warchest_source_would_be_lost(state, 0, 0.0, 5))

    def test_no_enemy_inbound_never_lost(self):
        """With no enemy fleets inbound the function always returns False."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 5.0}, production={0: 0.0}, ownership={0: 0},
        )
        self.assertFalse(h.warchest_source_would_be_lost(state, 0, 4.0, 10))

    def test_enemy_after_until_turn_ignored(self):
        """An enemy fleet that arrives after until_turn is not considered; the planet
        is safe within the window even though it would eventually fall."""
        h = self._make_h()
        state = make_warchest_state(
            garrison={0: 5.0}, production={0: 0.0}, ownership={0: 0},
            enemy=[WarchestFleet(owner=1, destination_id=0, fleet_size=100.0,
                                 garrison_on_arrival=0.0, arrival_turn=10,
                                 is_capture=True, production_id=0)],
        )
        self.assertFalse(h.warchest_source_would_be_lost(state, 0, 0.0, 5))


# ---------------------------------------------------------------------------
# 9. warchest_assign_fleet
# ---------------------------------------------------------------------------

class TestWarchestAssignFleet(unittest.TestCase):
    def _setup(self, planets, angular_velocity=0.0):
        obs = make_obs(planets, angular_velocity=angular_velocity)
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        return h, initial, horizon

    def test_single_source_wins(self):
        """100 ships from p0 can overcome p1's garrison of 20 plus production during travel;
        the assignment includes p0, sets launch_turn=0, and has a positive arrival_turn."""
        h, initial, horizon = self._setup(
            [[0, 0, 15.0, 30.0, 2.0, 100, 3],
             [1, 1, 15.0, 60.0, 2.0, 20, 2]]
        )
        assignment = h.warchest_assign_fleet(initial, h.planets[1], horizon)
        self.assertIn(0, assignment)
        ships, launch_turn, arrival_turn = assignment[0]
        self.assertGreater(ships, 0)
        self.assertEqual(launch_turn, 0)
        self.assertGreater(arrival_turn, 0)

    def test_no_source_returns_empty(self):
        """No owned planet is within MAX_DISTANCE of the enemy, so the assignment is empty."""
        h, initial, horizon = self._setup(
            [[0, 0, 2.0, 50.0, 2.0, 100, 3],
             [1, 1, 90.0, 50.0, 2.0, 20, 2]]
        )
        assignment = h.warchest_assign_fleet(initial, h.planets[1], horizon)
        self.assertEqual(assignment, {})

    def test_insufficient_ships_returns_empty(self):
        """Our planet has only 5 ships, which cannot overcome the enemy garrison of 100
        plus production during travel time, so the assignment is empty."""
        h, initial, horizon = self._setup(
            [[0, 0, 15.0, 30.0, 2.0, 5, 3],
             [1, 1, 15.0, 60.0, 2.0, 100, 2]]
        )
        assignment = h.warchest_assign_fleet(initial, h.planets[1], horizon)
        self.assertEqual(assignment, {})

    def test_defense_returns_empty_when_garrison_holds(self):
        """When called on an owned planet under threat, if the projected garrison survives
        all enemy arrivals without help (50+3*5=65 > enemy fleet 5) the function returns {}."""
        h, initial, horizon = self._setup(
            [[0, 0, 15.0, 30.0, 2.0, 50, 3],
             [1, 1, 15.0, 60.0, 2.0, 20, 2]]
        )
        initial.enemy_fleets.append(
            WarchestFleet(owner=1, destination_id=0, fleet_size=5.0,
                          garrison_on_arrival=0.0, arrival_turn=5,
                          is_capture=True, production_id=0)
        )
        assignment = h.warchest_assign_fleet(initial, h.planets[0], horizon)
        self.assertEqual(assignment, {})

    def test_committed_source_excluded(self):
        """A source whose id is in committed_ids is not eligible; if that is the only
        viable source the assignment is empty."""
        h, initial, horizon = self._setup(
            [[0, 0, 15.0, 30.0, 2.0, 100, 3],
             [1, 1, 15.0, 60.0, 2.0, 20, 2]]
        )
        initial.committed_ids.add(0)
        assignment = h.warchest_assign_fleet(initial, h.planets[1], horizon)
        self.assertEqual(assignment, {})

    def test_sun_blocking_returns_empty(self):
        """p0 at (30,50) to p1 at (70,50) passes through the sun at (50,50);
        first_planet_hit returns None so the assignment is empty."""
        h, initial, horizon = self._setup(
            [[0, 0, 30.0, 50.0, 2.0, 100, 3],
             [1, 1, 70.0, 50.0, 2.0, 20, 2]]
        )
        assignment = h.warchest_assign_fleet(initial, h.planets[1], horizon)
        self.assertEqual(assignment, {})

    def test_multi_source_combined(self):
        """When two owned planets are both in range and individually sufficient, at least
        one is selected and the total ships assigned is positive."""
        h, initial, horizon = self._setup(
            [[0, 0, 2.0, 40.0, 2.0, 80, 3],
             [1, 0, 2.0, 62.0, 2.0, 80, 3],
             [2, 1, 30.0, 50.0, 2.0, 40, 3]]
        )
        assignment = h.warchest_assign_fleet(initial, h.planets[2], horizon)
        self.assertNotEqual(assignment, {})
        total = sum(s for s, _, _ in assignment.values())
        self.assertGreater(total, 0)

    def test_all_sources_share_same_arrival_turn(self):
        """The synchronized-arrival invariant: every source in the assignment must have
        the same arrival_turn so they fight the garrison together."""
        h, initial, horizon = self._setup(
            [[0, 0, 2.0, 40.0, 2.0, 80, 3],
             [1, 0, 2.0, 62.0, 2.0, 80, 3],
             [2, 1, 30.0, 50.0, 2.0, 40, 3]]
        )
        assignment = h.warchest_assign_fleet(initial, h.planets[2], horizon)
        arrival_turns = {at for _, _, at in assignment.values()}
        self.assertEqual(len(arrival_turns), 1)


# ---------------------------------------------------------------------------
# 10. warchest_upper_bound
# ---------------------------------------------------------------------------

class TestWarchestUpperBound(unittest.TestCase):
    def test_upper_bound_at_least_base_score(self):
        """The upper bound is always >= the current score (adding captures can only help)."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        cands = h.warchest_candidates(initial, horizon)
        ub = h.warchest_upper_bound(initial, cands, horizon)
        base = h.warchest_score(initial, horizon)
        self.assertGreaterEqual(ub, base)

    def test_upper_bound_exceeds_base_when_capture_profitable(self):
        """When a candidate planet has positive net gain (production * remaining > capture cost),
        the upper bound exceeds the base score so the DFS branch is not pruned."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        cands = h.warchest_candidates(initial, horizon)
        ub = h.warchest_upper_bound(initial, cands, horizon)
        base = h.warchest_score(initial, horizon)
        self.assertGreater(ub, base)

    def test_upper_bound_empty_remaining_equals_score(self):
        """With no remaining candidates the upper bound equals the base score exactly
        (no additional gains are possible)."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        ub = h.warchest_upper_bound(initial, [], horizon)
        base = h.warchest_score(initial, horizon)
        self.assertAlmostEqual(ub, base)


# ---------------------------------------------------------------------------
# 11. warchest_earliest_capture
# ---------------------------------------------------------------------------

class TestWarchestEarliestCapture(unittest.TestCase):
    def test_single_source_sufficient(self):
        """100 ships from p0 beats p1's garrison alone; the earliest capture turn
        is finite and equals the ceil of the intercept travel time (~8 turns)."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        ec = h.warchest_earliest_capture(initial, h.planets[1], horizon)
        self.assertTrue(math.isfinite(ec))
        self.assertAlmostEqual(ec, 8.0)

    def test_unreachable_returns_inf(self):
        """When no owned planet is within MAX_DISTANCE, earliest capture is math.inf."""
        obs = make_obs([[0, 0, 2.0, 50.0, 2.0, 100, 3],
                        [1, 1, 90.0, 50.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        ec = h.warchest_earliest_capture(initial, h.planets[1], horizon)
        self.assertEqual(ec, math.inf)

    def test_multi_source_fallback(self):
        """Two sources of 80 each cannot individually beat garrison 100, but combined
        they can; the multi-source fallback returns a finite earliest capture turn."""
        obs = make_obs([[0, 0, 2.0, 40.0, 2.0, 80, 3],
                        [1, 0, 2.0, 62.0, 2.0, 80, 3],
                        [2, 1, 30.0, 50.0, 2.0, 100, 3]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        ec = h.warchest_earliest_capture(initial, h.planets[2], horizon)
        self.assertTrue(math.isfinite(ec))


# ---------------------------------------------------------------------------
# 12. warchest_candidates
# ---------------------------------------------------------------------------

class TestWarchestCandidates(unittest.TestCase):
    def test_profitable_enemy_included(self):
        """A reachable enemy planet with positive net gain (production * remaining > garrison)
        appears in the candidate list."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        cands = h.warchest_candidates(initial, horizon)
        self.assertIn(h.planets[1], cands)

    def test_owned_planets_not_in_offense_candidates(self):
        """Our own planets are never targets in the offense candidate list."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        cands = h.warchest_candidates(initial, horizon)
        for p in cands:
            self.assertNotEqual(initial.ownership.get(p.id), h.player)

    def test_defense_candidate_prepended(self):
        """An owned planet that will be lost to an in-flight enemy fleet is added as a
        defense candidate; the candidate list is non-empty."""
        obs = make_obs(
            [[0, 0, 2.0, 30.0, 2.0, 50, 3],
             [1, 0, 30.0, 30.0, 2.0, 15, 2],
             [2, 1, 60.0, 30.0, 2.0, 60, 2]],
            fleets=[[0, 1, 55.0, 30.0, math.pi, 2, 40]],
        )
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        cands = h.warchest_candidates(initial, horizon)
        self.assertGreater(len(cands), 0)

    def test_comet_excluded_from_candidates(self):
        """Comets are never attack targets; an enemy-owned comet does not appear in
        the candidate list even if it is reachable."""
        obs = make_obs([[0, 0, 2.0, 50.0, 2.0, 100, 3],
                        [1, 1, 30.0, 50.0, 2.0, 20, 2],
                        [2, 1, 20.0, 30.0, 1.0, 5, 1]],
                       comet_ids=[2])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        cands = h.warchest_candidates(initial, horizon)
        self.assertNotIn(2, [p.id for p in cands])

    def test_unprofitable_target_excluded(self):
        """An enemy planet whose garrison is so large that production * remaining_turns
        cannot cover the capture cost is excluded from candidates."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 99, 1]])  # production=1, huge garrison
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        cands = h.warchest_candidates(initial, horizon)
        self.assertNotIn(h.planets[1], cands)


# ---------------------------------------------------------------------------
# 13. warchest_execute
# ---------------------------------------------------------------------------

class TestWarchestExecute(unittest.TestCase):
    def test_execute_captures_and_advances_turn(self):
        """After executing a successful assignment the target planet is owned by the player
        and state.turn has advanced to the arrival turn."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        p1 = h.planets[1]
        assignment = h.warchest_assign_fleet(initial, p1, horizon)
        state = warchest_copy(initial)
        after = h.warchest_execute(state, p1, assignment)
        self.assertEqual(after.ownership[p1.id], h.player)
        arrival = list(assignment.values())[0][2]
        self.assertEqual(after.turn, arrival)

    def test_execute_deducts_ships_from_source(self):
        """Ships sent are removed from the source planet's garrison before the fleet departs."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        p1 = h.planets[1]
        assignment = h.warchest_assign_fleet(initial, p1, horizon)
        state = warchest_copy(initial)
        h.warchest_execute(state, p1, assignment)
        self.assertLess(state.garrison[0], initial.garrison[0])

    def test_execute_adds_source_to_committed(self):
        """After execution every source planet id appears in state.committed_ids to prevent
        the reinforcement pass from double-moving it."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        p1 = h.planets[1]
        assignment = h.warchest_assign_fleet(initial, p1, horizon)
        state = warchest_copy(initial)
        h.warchest_execute(state, p1, assignment)
        for src_id in assignment:
            self.assertIn(src_id, state.committed_ids)

    def test_failed_attack_still_advances_turn(self):
        """Even when the attacking fleet is too small to capture (garrison_on_arrival <= 0),
        state.turn still advances to the arrival turn so downstream scoring is correct."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        p1 = h.planets[1]
        assignment = {0: (1, 0, 8)}   # 1 ship cannot beat garrison 20+
        state = warchest_copy(initial)
        after = h.warchest_execute(state, p1, assignment)
        self.assertEqual(after.turn, 8)
        self.assertEqual(after.ownership[p1.id], 1)


# ---------------------------------------------------------------------------
# 14. warchest_dfs / run_unified_search
# ---------------------------------------------------------------------------

class TestRunUnifiedSearch(unittest.TestCase):
    def test_basic_attack_produces_move(self):
        """Two static planets on the left wall (no sun obstruction): p0 with 100 ships
        attacks p1 (20 ships). Exactly one move is returned, from p0, aimed upward (π/2)."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, moves = run(obs)
        self.assertEqual(len(moves), 1)
        planet_id, angle, ships = moves[0]
        self.assertEqual(planet_id, 0)
        self.assertGreater(ships, 0)
        self.assertAlmostEqual(angle, math.pi / 2, places=4)

    def test_no_enemy_returns_empty(self):
        """When all planets are owned by the player (no enemy_planets), main() short-circuits
        and returns an empty move list."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 50, 3],
                        [1, 0, 15.0, 60.0, 2.0, 30, 2]])
        _, moves = run(obs)
        self.assertEqual(moves, [])

    def test_enemy_out_of_range_returns_empty(self):
        """When the only enemy planet is more than MAX_DISTANCE away, no edges exist and
        run_unified_search finds no profitable attacks."""
        obs = make_obs([[0, 0, 2.0, 50.0, 2.0, 100, 3],
                        [1, 1, 90.0, 50.0, 2.0, 20, 2]])
        _, moves = run(obs)
        self.assertEqual(moves, [])

    def test_orbiting_enemy_produces_move(self):
        """p0 is static, p1 orbits at r=15; intercept_planet correctly predicts p1's future
        position and run_unified_search emits one move from p0."""
        obs = make_obs([[0, 0, 2.0, 50.0, 2.0, 100, 3],
                        [1, 1, 35.0, 50.0, 2.0, 20, 2]],
                       angular_velocity=0.03)
        _, moves = run(obs)
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0][0], 0)

    def test_sun_blocked_path_no_move(self):
        """p0 at (30,50) and p1 at (70,50) are on either side of the sun; the direct path
        is blocked so no warchest attack move is emitted for p0."""
        obs = make_obs([[0, 0, 30.0, 50.0, 2.0, 100, 3],
                        [1, 1, 70.0, 50.0, 2.0, 20, 2]])
        _, moves = run(obs)
        warchest_moves = [m for m in moves if m[0] == 0]
        self.assertEqual(warchest_moves, [])

    def test_committed_ids_set_after_search(self):
        """Every source planet that emits a move this turn must appear in
        _warchest_committed_ids so the reinforcement pass skips it."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, moves = run(obs)
        if moves:
            src_ids = {m[0] for m in moves}
            self.assertTrue(src_ids.issubset(h._warchest_committed_ids))

    def test_score_improves_after_dfs(self):
        """The best score found by warchest_dfs exceeds the base score of the initial
        state, confirming the DFS explored and improved on the do-nothing baseline."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        base_score = h.warchest_score(initial, horizon)
        cands = h.warchest_candidates(initial, horizon)
        best = [base_score, []]
        h.warchest_dfs(initial, cands, [], horizon, time.perf_counter(), best)
        self.assertGreater(best[0], base_score)


# ---------------------------------------------------------------------------
# 15. warchest_emit_moves
# ---------------------------------------------------------------------------

class TestWarchestEmitMoves(unittest.TestCase):
    def test_only_current_turn_moves_emitted(self):
        """A move whose launch_turn is in the future (deferred) must not appear in the
        output; deferred moves are re-planned next turn from scratch."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        p1 = h.planets[1]
        assignment = h.warchest_assign_fleet(initial, p1, horizon)
        src_id = list(assignment.keys())[0]
        ships, lt, at = assignment[src_id]
        assignment[src_id] = (ships, h.scene_step + 3, at)  # push launch into future
        moves = h.warchest_emit_moves([(p1, assignment)])
        self.assertEqual(moves, [])

    def test_current_turn_move_emitted(self):
        """A move whose launch_turn equals scene_step is included in the output with
        the correct source planet id."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        p1 = h.planets[1]
        assignment = h.warchest_assign_fleet(initial, p1, horizon)
        moves = h.warchest_emit_moves([(p1, assignment)])
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0][0], 0)


# ---------------------------------------------------------------------------
# 16. Reinforcement post-pass
# ---------------------------------------------------------------------------

class TestSendReinforcements(unittest.TestCase):
    def test_backline_reinforces_frontline(self):
        """p0 (backline, 50 ships) has p1 as its reinforcement target because p1 is on the
        frontline adjacent to enemy p2; p0 is included in the move list."""
        obs = make_obs([[0, 0, 2.0, 30.0, 2.0, 50, 3],
                        [1, 0, 30.0, 30.0, 2.0, 15, 2],
                        [2, 1, 60.0, 30.0, 2.0, 30, 2]])
        h, moves = run(obs)
        src_ids = {m[0] for m in moves}
        self.assertIn(0, src_ids)

    def test_reinforcement_target_set_correctly(self):
        """p0's reinforcement_target is p1 (the frontline planet closest to the enemy)."""
        obs = make_obs([[0, 0, 2.0, 30.0, 2.0, 50, 3],
                        [1, 0, 30.0, 30.0, 2.0, 15, 2],
                        [2, 1, 60.0, 30.0, 2.0, 30, 2]])
        h, _ = run(obs)
        p0 = next(p for p in h.owned_planets if p.id == 0)
        self.assertIsNotNone(p0.reinforcement_target)
        self.assertEqual(p0.reinforcement_target.id, 1)

    def test_committed_source_not_double_moved(self):
        """A planet that was committed by the warchest DFS search must not also appear
        in the reinforcement pass; every planet id should appear at most once in moves."""
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        h, moves = run(obs)
        src_counts = {}
        for m in moves:
            src_counts[m[0]] = src_counts.get(m[0], 0) + 1
        for src_id, count in src_counts.items():
            self.assertEqual(count, 1, f'planet {src_id} moved {count} times')


# ---------------------------------------------------------------------------
# 17. drain_comets
# ---------------------------------------------------------------------------

class TestDrainComets(unittest.TestCase):
    def test_owned_comet_drains_to_nearest_planet(self):
        """An owned comet at (15,25) drains all 30 ships to the nearest non-comet planet;
        exactly one move is emitted for the comet planet id with ships=30."""
        obs = make_obs(
            [[0, 0, 2.0, 40.0, 2.0, 100, 3],
             [1, 1, 30.0, 50.0, 2.0, 20, 2],
             [2, 0, 15.0, 25.0, 1.0, 30, 1]],
            comet_ids=[2],
        )
        _, moves = run(obs)
        comet_moves = [m for m in moves if m[0] == 2]
        self.assertEqual(len(comet_moves), 1)
        self.assertEqual(comet_moves[0][2], 30)

    def test_unowned_comet_not_drained(self):
        """An enemy-owned comet is not in owned_comets so drain_comets emits no move for it."""
        obs = make_obs(
            [[0, 0, 2.0, 40.0, 2.0, 100, 3],
             [1, 1, 30.0, 50.0, 2.0, 20, 2],
             [2, 1, 15.0, 25.0, 1.0, 30, 1]],
            comet_ids=[2],
        )
        _, moves = run(obs)
        comet_moves = [m for m in moves if m[0] == 2]
        self.assertEqual(comet_moves, [])

    def test_comet_not_a_warchest_target(self):
        """Comets are filtered out in warchest_candidates; an enemy comet never appears
        in the DFS candidate list regardless of its position."""
        obs = make_obs(
            [[0, 0, 2.0, 40.0, 2.0, 100, 3],
             [1, 1, 30.0, 50.0, 2.0, 20, 2],
             [2, 1, 15.0, 40.0, 1.0, 5, 1]],
            comet_ids=[2],
        )
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        horizon = min(h.scene_step + UNIFIED_LOOK_AHEAD, 500)
        cands = h.warchest_candidates(initial, horizon)
        self.assertNotIn(2, [p.id for p in cands])


# ---------------------------------------------------------------------------
# 18. build_destination_list (fleet tracking)
# ---------------------------------------------------------------------------

class TestBuildDestinationList(unittest.TestCase):
    def test_in_flight_enemy_fleet_tracked(self):
        """An enemy fleet at (55,30) heading west (angle=π) toward p1 at (30,30) is found
        by build_destination_list; destination_list[p1] contains one entry with owner=1
        and fleet_size=40."""
        obs = make_obs(
            [[0, 0, 2.0, 30.0, 2.0, 50, 3],
             [1, 0, 30.0, 30.0, 2.0, 15, 2],
             [2, 1, 60.0, 30.0, 2.0, 60, 2]],
            fleets=[[0, 1, 55.0, 30.0, math.pi, 2, 40]],
        )
        h, _ = run(obs)
        dest_planet = h.planets[1]
        self.assertIn(dest_planet, h.destination_list)
        arrivals = h.destination_list[dest_planet]
        self.assertEqual(len(arrivals), 1)
        owner, ships, t, sx, sy, px, py = arrivals[0]
        self.assertEqual(owner, 1)
        self.assertAlmostEqual(ships, 40)

    def test_enemy_fleet_in_initial_state(self):
        """The in-flight enemy fleet from build_destination_list is loaded into
        WarchestState.enemy_fleets with the correct fleet_size and arrival_turn."""
        obs = make_obs(
            [[0, 0, 2.0, 30.0, 2.0, 50, 3],
             [1, 0, 30.0, 30.0, 2.0, 15, 2],
             [2, 1, 60.0, 30.0, 2.0, 60, 2]],
            fleets=[[0, 1, 55.0, 30.0, math.pi, 2, 40]],
        )
        h, _ = run(obs)
        initial = h.warchest_initial_state()
        self.assertEqual(len(initial.enemy_fleets), 1)
        self.assertEqual(initial.enemy_fleets[0].fleet_size, 40)
        self.assertEqual(initial.enemy_fleets[0].arrival_turn, 8)


# ---------------------------------------------------------------------------
# 19. warchest_copy
# ---------------------------------------------------------------------------

class TestWarchestCopy(unittest.TestCase):
    def test_copy_is_independent_garrison(self):
        """Mutating the copy's garrison dict does not affect the original, confirming
        garrison is deep-copied."""
        state = make_warchest_state(
            garrison={0: 50.0}, production={0: 3.0}, ownership={0: 0},
        )
        copy = warchest_copy(state)
        copy.garrison[0] = 999.0
        self.assertAlmostEqual(state.garrison[0], 50.0)

    def test_copy_shares_production_reference(self):
        """production is read-only during the DFS so the copy shares the same dict object
        to avoid unnecessary allocation."""
        state = make_warchest_state(
            garrison={0: 50.0}, production={0: 3.0}, ownership={0: 0},
        )
        copy = warchest_copy(state)
        self.assertIs(copy.production, state.production)

    def test_copy_shares_enemy_fleets_reference(self):
        """enemy_fleets is never mutated in-place (warchest_advance rebinds the list),
        so the copy safely shares the same list object."""
        ef = WarchestFleet(owner=1, destination_id=0, fleet_size=10.0,
                           garrison_on_arrival=0.0, arrival_turn=5,
                           is_capture=True, production_id=0)
        state = make_warchest_state(
            garrison={0: 50.0}, production={0: 3.0}, ownership={0: 0},
            enemy=[ef],
        )
        copy = warchest_copy(state)
        self.assertIs(copy.enemy_fleets, state.enemy_fleets)

    def test_copy_independent_committed_ids(self):
        """committed_ids is deep-copied; adding an id to the copy does not pollute the
        original, keeping sibling DFS branches independent."""
        state = make_warchest_state(
            garrison={0: 50.0}, production={0: 3.0}, ownership={0: 0},
        )
        state.committed_ids.add(0)
        copy = warchest_copy(state)
        copy.committed_ids.add(1)
        self.assertNotIn(1, state.committed_ids)


# ---------------------------------------------------------------------------
# 20. Full agent() integration
# ---------------------------------------------------------------------------

class TestAgentIntegration(unittest.TestCase):
    def test_returns_list(self):
        """agent() always returns a list, never raises."""
        from hellburner import agent
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        result = agent(obs)
        self.assertIsInstance(result, list)

    def test_move_format_valid(self):
        """Every move in the returned list is a 3-element list of [int, float, int]
        with a positive ship count."""
        from hellburner import agent
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]])
        moves = agent(obs)
        for m in moves:
            self.assertEqual(len(m), 3)
            planet_id, angle, ships = m
            self.assertIsInstance(planet_id, int)
            self.assertIsInstance(angle, float)
            self.assertIsInstance(ships, int)
            self.assertGreater(ships, 0)

    def test_exception_returns_empty_list(self):
        """A malformed obs that raises an exception inside main() is caught by agent();
        the function returns an empty list rather than propagating the exception."""
        from hellburner import agent
        result = agent({})
        self.assertIsInstance(result, list)

    def test_later_step_still_produces_moves(self):
        """agent() functions correctly mid-game (step=100); scene_step is derived from
        obs['step']-1 and the horizon calculation stays within [0, 500]."""
        from hellburner import agent
        obs = make_obs([[0, 0, 15.0, 30.0, 2.0, 100, 3],
                        [1, 1, 15.0, 60.0, 2.0, 20, 2]],
                       step=100)
        result = agent(obs)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
