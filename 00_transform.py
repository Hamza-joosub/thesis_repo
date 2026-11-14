"""
00_transform.py — build a transformed panel from a raw Bloomberg export.

For every series in the raw file it writes three columns: the raw level, the weekly log
difference, and the weekly simple difference. Where a transform is not defined for a series
(non-positive values make the log undefined, a missing observation, the first row) the cell
is left blank so that downstream code sees NaN rather than a fabricated number.

Nothing is dropped and no row is deleted: the panel keeps every date in the raw file so that
a run can start or end wherever you like. Use --start and --end to cut a window, which is
what makes it possible to look at a particular episode in isolation.

Usage
  python 00_transform.py                                  raw_data.csv -> transformed_data.csv
  python 00_transform.py --in raw.csv --out panel.csv
  python 00_transform.py --start 2015-09-01 --end 2016-03-31 --out nenegate.csv
  python 00_transform.py --require-complete                keep only dates where every series is present
  python 00_transform.py --report                          print a per-series coverage summary and exit
"""
import argparse, sys
import numpy as np
import pandas as pd

# Series whose level is a rate, a yield or a spread rather than a price. For these the simple
# difference is the meaningful transform and the log difference is reported but rarely used.
RATE_LIKE = ('USGG', 'GT10', 'VIX', 'ALBIAY', 'SARPRT', 'CDS', 'Corp', 'GTZAR')


def is_rate_like(col: str) -> bool:
    return any(k.lower() in col.lower() for k in RATE_LIKE)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = 'Date'
    # Bloomberg writes '#N/A N/A' and similar; coerce everything to numbers
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in df.columns:
        s = df[col]
        out[col] = s
        # log difference: undefined where the level is missing or not strictly positive,
        # and undefined on the first observation of the series
        pos = s.where(s > 0)
        out[f'{col}_Log_Diff'] = np.log(pos) - np.log(pos.shift(1))
        # simple difference: defined wherever this observation and the previous one exist
        out[f'{col}_Diff'] = s - s.shift(1)
    panel = pd.DataFrame(out, index=df.index)
    # keep the raw / log-diff / diff triplets adjacent and in the raw file's order
    order = [c for col in df.columns for c in (col, f'{col}_Log_Diff', f'{col}_Diff')]
    return panel[order]


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in df.columns if not c.endswith(('_Log_Diff', '_Diff'))]:
        s = df[col]
        obs = s.notna()
        rows.append({
            'series': col,
            'first': s.first_valid_index(),
            'last': s.last_valid_index(),
            'obs': int(obs.sum()),
            'missing_inside': int((~obs.loc[s.first_valid_index():s.last_valid_index()]).sum()) if obs.any() else 0,
            'non_positive': int((s <= 0).sum()),
            'log_diff_ok': int(df[f'{col}_Log_Diff'].notna().sum()),
            'rate_like': is_rate_like(col),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='raw_data.csv')
    ap.add_argument('--out', default='transformed_data.csv')
    ap.add_argument('--start', default=None, help='first date to keep, e.g. 2015-09-01')
    ap.add_argument('--end', default=None, help='last date to keep')
    ap.add_argument('--require-complete', action='store_true',
                    help='keep only dates on which every series has a level')
    ap.add_argument('--min-obs', type=int, default=0,
                    help='drop series with fewer than this many observations in the window')
    ap.add_argument('--report', action='store_true', help='print coverage and exit')
    a = ap.parse_args()

    raw = load_raw(a.inp)
    print(f'read {a.inp}: {raw.shape[0]} dates, {raw.shape[1]} series, '
          f'{raw.index.min().date()} to {raw.index.max().date()}')

    panel = transform(raw)

    # transforms are computed on the full history first, then the window is cut, so that the
    # first row of a window keeps its difference from the week before it
    if a.start: panel = panel.loc[a.start:]
    if a.end:   panel = panel.loc[:a.end]

    if a.min_obs:
        keep = [c for c in raw.columns if panel[c].notna().sum() >= a.min_obs]
        dropped = [c for c in raw.columns if c not in keep]
        if dropped: print(f'dropped {len(dropped)} series below {a.min_obs} observations: {", ".join(dropped[:6])}'
                          + (' ...' if len(dropped) > 6 else ''))
        panel = panel[[c for col in keep for c in (col, f'{col}_Log_Diff', f'{col}_Diff')]]

    if a.require_complete:
        levels = [c for c in panel.columns if not c.endswith(('_Log_Diff', '_Diff'))]
        before = len(panel)
        panel = panel[panel[levels].notna().all(axis=1)]
        print(f'require-complete: {before} dates -> {len(panel)}')

    if a.report:
        cov = coverage(panel)
        pd.set_option('display.width', 200, 'display.max_rows', 300)
        print(cov.to_string(index=False))
        sys.exit(0)

    panel.to_csv(a.out, float_format='%.10g')
    lv = [c for c in panel.columns if not c.endswith(('_Log_Diff', '_Diff'))]
    print(f'wrote {a.out}: {panel.shape[0]} dates, {len(lv)} series, {panel.shape[1]} columns '
          f'({panel.index.min().date()} to {panel.index.max().date()})')
    print(f'cells present: levels {panel[lv].notna().mean().mean():.1%}, '
          f'log differences {panel[[c for c in panel.columns if c.endswith("_Log_Diff")]].notna().mean().mean():.1%}, '
          f'differences {panel[[c for c in panel.columns if c.endswith("_Diff")]].notna().mean().mean():.1%}')
    bad = [c for c in lv if (panel[c] <= 0).any()]
    if bad: print(f'log difference blank where the level is not positive: {", ".join(bad)}')
