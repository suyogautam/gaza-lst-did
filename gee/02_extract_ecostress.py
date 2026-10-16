"""
Google Earth Engine extraction for "Land surface warming after tree-cover and building loss in the Gaza Strip"
(S. Gautam). Step 2 of 2: ECOSTRESS overpasses (NASA/ECOSTRESS/L2T_LSTE/V2, 70 m) on the same 100 m grid.
Run 01_extract_landsat.py first; this script reads its data/grid.json and data/cells.pkl.

What it writes (under data/ecostress/):
  inventory.csv                every ECOSTRESS tile over the region (Oct 2018 - Sep 2026): id, time, share of Gaza
                               with clear, valid LST
  selected_overpasses.csv      dry-season overpasses in the mid-morning (09:00-11:30) and late-evening
                               (21:00-05:00) local-solar-time windows with >= 30% of Gaza valid
  scenes/pass_NNNNN.npz        per selected overpass and candidate cell: LST (deg C x 100), wide-band emissivity
                               (x 10000) and the share of the cell's 70 m pixels that were valid (x 1000)

Masking: cloud and water masks, LST uncertainty <= 2.5 K, 250-345 K. Tiles less than 3 minutes apart are one
overpass and are mosaicked. Needs an Earth Engine account; set CFG['PROJECT'] below. Run from the repository
root:   python gee/02_extract_ecostress.py
"""
import os, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

CFG = dict(
    PROJECT = 'your-ee-project',                      # your Earth Engine cloud project
    COLLECTION = 'NASA/ECOSTRESS/L2T_LSTE/V2',
    T_START = '2018-10-01', T_END = '2026-10-01',     # same period as the Landsat extraction
    WAR_DATE = '2023-10-07',
    # time-of-day windows in local solar hours (UTC + longitude/15); Landsat 8/9 pass at about 10:00
    WINDOWS = {'morning': (9.0, 11.5), 'night': (21.0, 5.0)},
    DRY = (5, 6, 7, 8, 9, 10),
    SEASON = 'dry',                                    # 'dry' or 'all'
    MIN_CLEAR_FRAC = 0.30,     # an overpass is kept only if >= 30% of Gaza has clear, valid LST
    LST_ERR_MAX = 2.5,         # K; drop pixels whose LST uncertainty is above this
    LST_RANGE_K = (250.0, 345.0),
    USE_QC_BITS = False,
    QC_BAD_MASK = 0,
    N_WORKERS = 4,
    DATA_DIR = 'data',
)
ECO_DIR = os.path.join(CFG['DATA_DIR'], 'ecostress')
SCENE_DIR = os.path.join(ECO_DIR, 'scenes')
os.makedirs(SCENE_DIR, exist_ok=True)
GAZA_LON = 34.4                                         # for local solar time
WAR = pd.Timestamp(CFG['WAR_DATE'])

GRID_FILE = os.path.join(CFG['DATA_DIR'], 'grid.json')
CELLS_FILE = os.path.join(CFG['DATA_DIR'], 'cells.pkl')
for f in [GRID_FILE, CELLS_FILE]:
    if not os.path.isfile(f):
        raise FileNotFoundError(f'{f} not found. Run gee/01_extract_landsat.py first.')
GRID = json.load(open(GRID_FILE))
CELLS = pd.read_pickle(CELLS_FILE)
print(f'grid {GRID}; {len(CELLS):,} candidate cells')

# ---------------------------------------------------------------- Earth Engine (started only when needed)
R = GRID['RES']
INV_FILE = os.path.join(ECO_DIR, 'inventory.csv')
EE_READY = False

def ee_start():
    global ee, GAZA, GRID_PROJ, REGION, ECO, P70, EE_READY
    if EE_READY:
        return
    import ee
    try:
        ee.Initialize(project=CFG['PROJECT'])
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=CFG['PROJECT'])
    gaul = ee.FeatureCollection('FAO/GAUL/2015/level0')
    GAZA = gaul.filter(ee.Filter.eq('ADM0_NAME', 'Gaza Strip')).geometry()
    GRID_PROJ = ee.Projection(GRID['CRS'], [R, 0, GRID['X0'], 0, -R, GRID['Y0']])
    REGION = ee.Geometry.Rectangle([GRID['X0'], GRID['Y0'] - GRID['H'] * R, GRID['X0'] + GRID['W'] * R,
                                    GRID['Y0']], ee.Projection(GRID['CRS']), False)
    ECO = ee.ImageCollection(CFG['COLLECTION']).filterBounds(REGION).filterDate(CFG['T_START'], CFG['T_END'])
    P70 = ee.Projection(GRID['CRS']).atScale(70)
    EE_READY = True

def ee_retry(fn, tries=6):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1:
                raise
            wait = 5 * 2 ** k
            print(f'   EE error ({str(e)[:90]}...), retry in {wait}s')
            time.sleep(wait)

def fetch(img, bands, max_bytes=12e6):
    """ee.Image -> {band: (H, W) float32} on the grid written by 01_extract_landsat.py, in row strips."""
    H, W = GRID['H'], GRID['W']
    img = img.select(bands).toFloat().unmask(-9999, False)
    rows = max(1, int(max_bytes // (4 * len(bands) * W)))
    out = {b: np.full((H, W), np.nan, np.float32) for b in bands}
    for r0 in range(0, H, rows):
        h = min(rows, H - r0)
        req = {'expression': img, 'fileFormat': 'NUMPY_NDARRAY',
               'grid': {'dimensions': {'width': W, 'height': h},
                        'affineTransform': {'scaleX': R, 'shearX': 0, 'translateX': GRID['X0'],
                                            'shearY': 0, 'scaleY': -R, 'translateY': GRID['Y0'] - r0 * R},
                        'crsCode': GRID['CRS']}}
        arr = ee_retry(lambda: ee.data.computePixels(req))
        for b in bands:
            a = np.asarray(arr[b], np.float32)
            a[a <= -9998] = np.nan
            out[b][r0:r0 + h] = a
    return out

if not os.path.isfile(INV_FILE):                  # units check, first run only
    ee_start()
    first = ee.Image(ECO.filterDate('2023-07-01', '2023-09-01').sort('system:time_start').first())
    print('bands:', first.bandNames().getInfo())
    stats = first.select(['LST', 'EmisWB', 'QC', 'cloud', 'water']).reduceRegion(
        ee.Reducer.percentile([5, 50, 95]), GAZA, 300, maxPixels=1e9).getInfo()
    print('LST / EmisWB / QC / cloud / water percentiles over Gaza on one summer scene:')
    for k in sorted(stats):
        print(f'   {k}: {stats[k]}')
    lst_med = stats.get('LST_p50')
    if lst_med is not None and not (240 < lst_med < 350):
        raise ValueError(f'LST median {lst_med} is not in kelvin. Check the catalog for a scale factor and set '
                         'it in prep() below before going on.')
else:
    print('inventory and scenes cached: Earth Engine not needed unless scenes are missing')

# ---------------------------------------------------------------- scene inventory
def prep(img):
    """Mask clouds, water, poor LST, implausible values. Returns LST (deg C) and EmisWB."""
    lst = img.select('LST')
    ok = (img.select('cloud').eq(0)
          .And(img.select('water').eq(0))
          .And(img.select('LST_err').lte(CFG['LST_ERR_MAX']))
          .And(lst.gt(CFG['LST_RANGE_K'][0])).And(lst.lt(CFG['LST_RANGE_K'][1])))
    if CFG['USE_QC_BITS'] and CFG['QC_BAD_MASK']:
        ok = ok.And(img.select('QC').toInt().bitwiseAnd(CFG['QC_BAD_MASK']).eq(0))
    out = lst.subtract(273.15).rename('LST').addBands(img.select('EmisWB').rename('EMIS')).updateMask(ok)
    return ee.Image(out.copyProperties(img, ['system:time_start', 'system:index']))

if not os.path.isfile(INV_FILE):
    ee_start()
    def clear_frac(img):
        p = prep(img)
        frac = p.select('LST').mask().unmask(0).reduceRegion(ee.Reducer.mean(), GAZA, 300, maxPixels=1e9).get('LST')
        return ee.Feature(None, {'id': img.get('system:index'), 't': img.get('system:time_start'),
                                 'clear': frac})
    parts = []
    for y in range(2018, 2027):
        sub = ECO.filterDate(f'{y}-01-01', f'{y + 1}-01-01')
        n = sub.size().getInfo()
        if n == 0:
            continue
        fc = ee_retry(lambda: ee.FeatureCollection(sub.map(clear_frac)).getInfo())
        parts.append(pd.DataFrame([f['properties'] for f in fc['features']]))
        print(f'  {y}: {n} tile images')
    inv = pd.concat(parts, ignore_index=True)
    inv.to_csv(INV_FILE, index=False)

INV = pd.read_csv(INV_FILE)
INV['clear'] = INV['clear'].fillna(0.0)
INV['utc'] = pd.to_datetime(INV.t, unit='ms', utc=True)
INV = INV.sort_values('utc').reset_index(drop=True)
# group tiles of one pass: images less than 3 minutes apart belong to the same overpass
INV['pass_id'] = (INV.utc.diff().dt.total_seconds().fillna(1e9) > 180).cumsum()
P = (INV.groupby('pass_id')
       .agg(utc=('utc', 'min'), clear=('clear', 'sum'), ids=('id', lambda s: '|'.join(s)))
       .reset_index())
P['clear'] = P['clear'].clip(upper=1.0)        # tiles of one pass cover different parts of Gaza
P['solar_hour'] = ((P.utc.dt.hour + P.utc.dt.minute / 60 + GAZA_LON / 15) % 24).round(2)
P['date'] = P.utc.dt.tz_convert(None)
P['month'] = P.date.dt.month
P['post'] = (P.date >= WAR).astype(int)
print(f'{len(INV):,} tile images -> {len(P):,} overpasses; '
      f"{(P.clear >= CFG['MIN_CLEAR_FRAC']).sum():,} with >= {CFG['MIN_CLEAR_FRAC']:.0%} of Gaza clear")

# ---------------------------------------------------------------- overpasses in each window
def in_window(h, w):
    h0, h1 = w
    return (h >= h0) & (h < h1) if h0 < h1 else (h >= h0) | (h < h1)

SEL = {}
for name, w in CFG['WINDOWS'].items():
    m = in_window(P.solar_hour, w) & (P.clear >= CFG['MIN_CLEAR_FRAC'])
    if CFG['SEASON'] == 'dry':
        m &= P.month.isin(CFG['DRY'])
    SEL[name] = P[m].copy()
    n_pre, n_post = int((SEL[name].post == 0).sum()), int((SEL[name].post == 1).sum())
    print(f'{name:8s}: {len(SEL[name]):4d} overpasses ({n_pre} pre-war, {n_post} post-war)')
    if len(SEL[name]):
        print('          by year:', SEL[name].date.dt.year.value_counts().sort_index().to_dict())
pd.concat([v.assign(window=k) for k, v in SEL.items()])[['window', 'pass_id', 'utc', 'solar_hour', 'clear', 'post', 'ids']] \
    .to_csv(os.path.join(ECO_DIR, 'selected_overpasses.csv'), index=False)

# ---------------------------------------------------------------- extract each overpass onto the 100 m grid
def overpass_image(ids):
    imgs = [prep(ee.Image(f"{CFG['COLLECTION']}/{i}")) for i in ids.split('|')]
    mos = ee.ImageCollection(imgs).mosaic().setDefaultProjection(P70)
    valid = mos.select('LST').mask().unmask(0).rename('COV').setDefaultProjection(P70)
    return (mos.addBands(valid)
            .reduceResolution(ee.Reducer.mean(), maxPixels=64)
            .reproject(GRID_PROJ))

def scene_file(pid):
    return os.path.join(SCENE_DIR, f'pass_{int(pid):05d}.npz')

FLAT = CELLS.flat.to_numpy()                 # candidate cells, same order as cells.pkl

def do_pass(row):
    if os.path.isfile(scene_file(row.pass_id)):
        return row.pass_id, 'cached'
    full = fetch(overpass_image(row.ids), ['LST', 'EMIS', 'COV'])
    lst = full['LST'].ravel()[FLAT]; em = full['EMIS'].ravel()[FLAT]; cov = full['COV'].ravel()[FLAT]
    np.savez_compressed(scene_file(row.pass_id),
                        LST=np.where(np.isfinite(lst), np.round(lst * 100), -32768).astype(np.int16),
                        EMIS=np.where(np.isfinite(em), np.round(em * 10000), -32768).astype(np.int16),
                        COV=np.where(np.isfinite(cov), np.round(cov * 1000), -32768).astype(np.int16))
    return row.pass_id, 'ok'

todo = pd.concat(SEL.values()).drop_duplicates('pass_id')
todo = todo[[not os.path.isfile(scene_file(p)) for p in todo.pass_id]]
if len(todo):
    ee_start()
print(f'{len(todo)} overpasses to fetch')
t_ = time.time()
with ThreadPoolExecutor(CFG['N_WORKERS']) as pool:
    futs = [pool.submit(do_pass, r) for r in todo.itertuples()]
    for k, f in enumerate(as_completed(futs), 1):
        pid, msg = f.result()
        if k % 10 == 0 or k == len(futs):
            print(f'  [{k}/{len(futs)}] ({time.time() - t_:.0f}s)')

print('done: data/ecostress/inventory.csv, selected_overpasses.csv, scenes/*.npz')
