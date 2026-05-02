"""
contest.py — run N parallel games of baseline vs challenger Hellburner.

Usage:
    python contest.py [--games 24] [--jobs 24] [--const REINFORCEMENT_SIZE=12]

Each game uses a random seed so results are independent.
At the end prints win/loss/draw counts and a p-value (binomial test).
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


def run_game(seed: int, overrides: dict) -> tuple[float, float]:
    """Run one game. Returns (baseline_reward, challenger_reward)."""
    import random as _random
    _random.seed(seed)

    from hellburner import agent as baseline_agent, make_agent
    challenger_agent = make_agent(**overrides)

    from kaggle_environments import make
    env = make('orbit_wars', debug=False)

    # Alternate who plays player 0 each game based on seed parity to remove position bias.
    if seed % 2 == 0:
        players = [baseline_agent, challenger_agent]
        baseline_idx, challenger_idx = 0, 1
    else:
        players = [challenger_agent, baseline_agent]
        baseline_idx, challenger_idx = 1, 0

    env.run(players)
    final = env.steps[-1]
    b_reward = final[baseline_idx].reward or 0.0
    c_reward = final[challenger_idx].reward or 0.0
    return b_reward, c_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=24)
    parser.add_argument('--jobs',  type=int, default=24)
    parser.add_argument('--const', action='append', default=[],
                        metavar='NAME=VALUE',
                        help='Override a Hellburner constant for the challenger (repeatable)')
    args = parser.parse_args()

    overrides = parse_overrides(args.const)
    if not overrides:
        print('No overrides specified. Use --const NAME=VALUE. Exiting.')
        sys.exit(1)

    print(f'Baseline vs Challenger ({overrides})')
    print(f'Running {args.games} games across {args.jobs} workers...\n')

    seeds = [random.randint(0, 99999) for _ in range(args.games)]

    baseline_wins = challenger_wins = draws = 0
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_game, s, overrides): s for s in seeds}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            seed = futures[fut]
            try:
                b, c = fut.result()
                results.append((b, c))
                if b > c:
                    baseline_wins += 1
                    outcome = 'Baseline wins'
                elif c > b:
                    challenger_wins += 1
                    outcome = 'Challenger wins'
                else:
                    draws += 1
                    outcome = 'Draw'
                print(f'  Game {i:3d}/{args.games} seed={seed}: baseline={b:.1f} challenger={c:.1f}  [{outcome}]')
            except Exception as e:
                print(f'  Game {i:3d}/{args.games} seed={seed}: ERROR {e}')
                draws += 1

    total = args.games
    decisive = baseline_wins + challenger_wins
    print(f'\n--- Results ({total} games) ---')
    print(f'  Baseline wins:    {baseline_wins:3d}')
    print(f'  Challenger wins:  {challenger_wins:3d}')
    print(f'  Draws:            {draws:3d}')
    if decisive > 0:
        win_rate = challenger_wins / decisive
        print(f'  Challenger win rate (decisive): {win_rate:.1%}')
        try:
            from scipy.stats import binomtest
            result = binomtest(challenger_wins, decisive, 0.5, alternative='greater')
            print(f'  p-value (challenger > baseline): {result.pvalue:.4f}')
        except ImportError:
            # Rough normal approximation
            import math
            z = (challenger_wins - decisive / 2) / math.sqrt(decisive / 4)
            print(f'  z-score (challenger > baseline): {z:.2f}')
    print()


if __name__ == '__main__':
    main()
