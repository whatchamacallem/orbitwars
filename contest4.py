"""
contest4.py — 4-player contest: 1 challenger vs 3 baselines.

Each game has the challenger at seat (seed % 4) and a fresh baseline at each
of the other three seats.  Place is computed as:
    1 + number of baselines that strictly beat the challenger
so all three tied losers get place 2, not place 4.  Null win-rate is 25%
(random baseline in a fair 4-player game).

Usage:
    python contest4.py [--games 200] [--jobs 24] [--const REINFORCEMENT_SIZE=12]
"""
import argparse
import concurrent.futures
import random
import sys

sys.path.insert(0, '/home/t/orbitwars')


def parse_overrides(specs: list[str]) -> dict:
    overrides = {}
    for spec in specs:
        k, _, v = spec.partition('=')
        k = k.strip()
        v = v.strip()
        try:
            overrides[k] = int(v)
        except ValueError:
            try:
                overrides[k] = float(v)
            except ValueError:
                overrides[k] = v
    return overrides


def run_game(seed: int, overrides: dict) -> tuple[float, list[float], int]:
    """Run one 4-player game. Returns (challenger_reward, baseline_rewards, place_1_to_4)."""
    import random as _random
    _random.seed(seed)

    from hellburner import make_agent
    #challenger_agent = make_agent(**overrides)
    challenger_agent = make_agent()
    # Three distinct baseline function objects — avoids any agent-identity aliasing.
    b0, b1, b2 = make_agent(), make_agent(), make_agent()

    from kaggle_environments import make
    env = make('orbit_wars', debug=False)

    env.run([challenger_agent, b0, b1, b2])
    final = env.steps[-1]
    rewards = [final[i].reward or 0.0 for i in range(4)]

    c_reward = rewards[0]
    b_rewards = [rewards[i] for i in [1,2,3]]

    # Place: 1 = won outright; 2 = one baseline beat us; etc.
    # Uses only the three baseline rewards so tied losers are place 2, not 4.
    place = 1 + sum(1 for r in b_rewards if r > c_reward)

    return place


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--jobs',  type=int, default=24)
    parser.add_argument('--const', action='append', default=[],
                        metavar='NAME=VALUE',
                        help='Override a Hellburner constant for the challenger (repeatable)')
    args = parser.parse_args()

    overrides = parse_overrides(args.const)
    if not overrides:
        print('No overrides specified. Use --const NAME=VALUE. Exiting.')
        sys.exit(1)

    print(f'4-way: Challenger ({overrides}) vs 3 Baselines')
    print(f'Running {args.games} games across {args.jobs} workers...\n')

    seeds = [random.randint(0, 99999) for _ in range(args.games)]

    wins = 0
    total_place = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_game, s, overrides): s for s in seeds}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            seed = futures[fut]
            try:
                place = fut.result()
                if place == 1:
                    wins += 1
                total_place += place
                print(f'  [{i}/{args.games}] seed={seed} place={place}  '
                      f'wins={wins}  avg_place={total_place/i:.2f}', end='\r')
            except Exception as e:
                print(f'\n  [{i}/{args.games}] seed={seed} ERROR: {e}')

    print()
    print(f'\nResults: {wins}/{args.games} wins ({100*wins/args.games:.1f}%)  '
          f'avg place={total_place/args.games:.2f}')




if __name__ == '__main__':
    main()
