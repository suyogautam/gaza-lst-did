"""
Google Earth Engine extraction for "Land surface warming after tree-cover and building loss in the Gaza Strip"
(S. Gautam). Step 1 of 2: the 100 m grid, static layers and 96 monthly Landsat 8/9 composites.

What it writes (under data/):
  grid.json            100 m UTM 36N grid over the Gaza Strip plus a 25 km buffer
  static_raw.npz       ESA WorldCover 2021 class fractions, SRTM elevation, GAUL country code, per grid cell
  cells.pkl            candidate cells: every Gaza land cell and the eligible Israeli/Egyptian land cells
  monthly/YYYY_MM.npz  per candidate cell and month (Oct 2018 - Sep 2026): mean LST (deg C), NDVI, broadband
                       albedo, the emissivity assumed by the Landsat ST product, mean clear-observation count
                       and the share of the cell's 30 m pixels with at least one clear observation

Pixels are pulled straight into Python with ee.data.computePixels (no Drive exports). Anything already on disk
is reused, so the script can be stopped and restarted. Needs an Earth Engine account and a cloud project:
set CFG['PROJECT'] below. Run from the repository root:   python gee/01_extract_landsat.py
"""
import os, json, math, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import ndimage

# ---------------------------------------------------------------- configuration
CFG = dict(
    PROJECT = 'your-ee-project',        # your Earth Engine cloud project

    # ---- time ------------------------------------------------------------------------------
    T0 = '2018-10-01',                  # first month (start of war-year -5)
    N_MONTHS = 96,                      # Oct 2018 .. Sep 2026 = 5 pre + 3 post war-years
    WAR_MONTH = 60,                     # month index of Oct 2023 (0 = Oct 2018)

    # ---- grid ------------------------------------------------------------------------------
    CRS = 'EPSG:32636',                 # UTM 36N; Gaza and Landsat's own tiles are in this zone
    RES = 100,                          # TIRS is 100 m; the 30 m ST band is resampled, so 100 m is the honest unit
    BUFFER_KM = 25,                     # how far into Israel/Egypt we look for control land

    # ---- Landsat ---------------------------------------------------------------------------
    MAX_SCENE_CLOUD = 80,               # drop near-overcast scenes before compositing (speed only)
    ST_QA_MAX = 3.0,                    # K; drop pixels whose ST uncertainty is above this

    # ---- external control land -------------------------------------------------------------
    EXCL_ISRAEL_KM = 7.0,               # Israeli communities within ~7 km were evacuated Oct 2023; military staging
    EXCL_EGYPT_KM = 5.0,                # Egypt's Rafah buffer zone (demolished 2014-2020) and the 2024 enclosure
    ELEV_MARGIN_M = 15,                 # external cells must sit inside Gaza's elevation range +- this
    DCOAST_MARGIN_KM = 2.0,             # ...and inside Gaza's distance-to-coast range + this
    BLOCK_CELLS = 20,                   # 20 x 100 m = 2 km blocks (stored with each cell)

    # ---- Earth Engine requests ---------------------------------------------------------------
    N_WORKERS = 4,                      # parallel Earth Engine requests
    REFRESH_DAYS = 45,                  # re-fetch a month once if it was downloaded < this many days after it ended

    DATA_DIR = 'data',
)

for d in [CFG['DATA_DIR'], os.path.join(CFG['DATA_DIR'], 'monthly')]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- helpers
T0 = pd.Timestamp(CFG['T0'])
MONTHS = pd.date_range(T0, periods=CFG['N_MONTHS'], freq='MS')
MOY = MONTHS.month.to_numpy()                                   # calendar month of each index
WY = (np.arange(CFG['N_MONTHS']) // 12) - 5                     # war-year: -5 .. 2
W0 = CFG['WAR_MONTH']
assert MONTHS[W0] == pd.Timestamp('2023-10-01')

def month_index(d):
    d = pd.to_datetime(d)
    return (d.year - T0.year) * 12 + (d.month - T0.month)

def ll_to_utm(lon, lat, zone=36):
    """WGS84 lon/lat -> UTM (northern hemisphere). Snyder's series; mm-level vs pyproj here."""
    a = 6378137.0; f = 1 / 298.257223563; k0 = 0.9996
    e2 = f * (2 - f); ep2 = e2 / (1 - e2); e4 = e2 * e2; e6 = e4 * e2
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    phi = np.radians(np.asarray(lat, float)); lam = np.radians(np.asarray(lon, float))
    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2); T = np.tan(phi) ** 2
    C = ep2 * np.cos(phi) ** 2; A = np.cos(phi) * (lam - lon0)
    M = a * ((1 - e2/4 - 3*e4/64 - 5*e6/256) * phi - (3*e2/8 + 3*e4/32 + 45*e6/1024) * np.sin(2*phi)
             + (15*e4/256 + 45*e6/1024) * np.sin(4*phi) - (35*e6/3072) * np.sin(6*phi))
    x = k0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18*T + T**2 + 72*C - 58*ep2) * A**5 / 120) + 500000.0
    y = k0 * (M + N * np.tan(phi) * (A**2 / 2 + (5 - T + 9*C + 4*C**2) * A**4 / 24
                                     + (61 - 58*T + T**2 + 600*C - 330*ep2) * A**6 / 720))
    return x, y

GRID_FILE = os.path.join(CFG['DATA_DIR'], 'grid.json')
STATIC_FILE = os.path.join(CFG['DATA_DIR'], 'static_raw.npz')
CELLS_FILE = os.path.join(CFG['DATA_DIR'], 'cells.pkl')
MONTH_DIR = os.path.join(CFG['DATA_DIR'], 'monthly')

# monthly bands and how they are packed to int16 on disk (value = int / scale)
MBANDS = {'LST': 100.0, 'NDVI': 10000.0, 'ALB': 10000.0, 'EMIS': 10000.0, 'N': 100.0, 'COV': 10000.0}
NA16 = -32768

def pack(a, scale):
    q = np.clip(np.round(np.asarray(a, np.float64) * scale), -32767, 32767)
    q[~np.isfinite(q)] = NA16
    return q.astype(np.int16)

def unpack(q, scale):
    a = q.astype(np.float32) / scale
    a[q == NA16] = np.nan
    return a

def month_file(i):
    return os.path.join(MONTH_DIR, f'{MONTHS[i]:%Y_%m}.npz')

def save_month(i, full, n_scenes, cells_flat):
    """full: dict band -> (H, W) float array on the grid. Only candidate cells are kept on disk."""
    out = {b: pack(full[b].ravel()[cells_flat], s) for b, s in MBANDS.items()}
    np.savez_compressed(month_file(i), n_scenes=n_scenes, saved=time.time(), **out)


# ---------------------------------------------------------------- candidate cells
WC_CLASSES = {'f_tree': 10, 'f_shrub': 20, 'f_grass': 30, 'f_crop': 40, 'f_built': 50, 'f_bare': 60, 'f_water': 80}

def build_cells(grid, st, cfg):
    H, W, R = grid['H'], grid['W'], grid['RES']
    country = np.nan_to_num(st['country']).astype(int)          # 1 Gaza, 2 Israel, 3 Egypt, 0 other/sea
    fw = st['f_water']
    wet = (country == 0) & (~np.isfinite(fw) | (fw >= 0.5))
    lab, _ = ndimage.label(wet)
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    # the sea = the largest wet component that touches the grid edge
    sizes = {k: (lab == k).sum() for k in edge}
    sea = lab == max(sizes, key=sizes.get) if sizes else np.zeros_like(wet)
    dcoast = ndimage.distance_transform_edt(~sea) * R / 1000.0
    gaza = country == 1
    dgaza = ndimage.distance_transform_edt(~gaza) * R / 1000.0
    land = ~sea & np.isfinite(st['elev']) & np.isfinite(st['f_built'])

    gz_elev = st['elev'][gaza & land]; gz_dc = dcoast[gaza & land]
    e_lo, e_hi = np.nanpercentile(gz_elev, 1) - cfg['ELEV_MARGIN_M'], np.nanpercentile(gz_elev, 99) + cfg['ELEV_MARGIN_M']
    dc_hi = np.nanpercentile(gz_dc, 99) + cfg['DCOAST_MARGIN_KM']
    ext = (land & ((country == 2) & (dgaza >= cfg['EXCL_ISRAEL_KM']) | (country == 3) & (dgaza >= cfg['EXCL_EGYPT_KM']))
           & (st['elev'] >= e_lo) & (st['elev'] <= e_hi) & (dcoast <= dc_hi))
    keep = (gaza & land) | ext
    rr, cc = np.nonzero(keep)
    cells = pd.DataFrame({'flat': rr * W + cc, 'row': rr, 'col': cc,
                          'x': grid['X0'] + (cc + 0.5) * R, 'y': grid['Y0'] - (rr + 0.5) * R,
                          'country': country[rr, cc], 'elev': st['elev'][rr, cc],
                          'dcoast_km': dcoast[rr, cc], 'dgaza_km': dgaza[rr, cc]})
    for k in WC_CLASSES:
        cells[k] = np.nan_to_num(st[k][rr, cc])
    cells['gz'] = (cells.country == 1).astype(int)
    b = cfg['BLOCK_CELLS']
    cells['block'] = (cells.row // b) * 10000 + cells.col // b
    cells['coastband'] = np.digitize(cells.dcoast_km, [1, 2.5, 5, 8])
    print(f'Gaza cells {cells.gz.sum():,} | external candidates {(cells.gz == 0).sum():,} '
          f'(Israel {(cells.country == 2).sum():,}, Egypt {(cells.country == 3).sum():,})')
    print(f'Gaza elevation 1-99%: {np.nanpercentile(gz_elev, 1):.0f}-{np.nanpercentile(gz_elev, 99):.0f} m; '
          f'distance to coast up to {np.nanpercentile(gz_dc, 99):.1f} km')
    return cells, dict(sea=sea, dcoast=dcoast, dgaza=dgaza, country=country)

def baseline_stratum(c):
    """Dominant baseline land cover from ESA WorldCover 2021 fractions (priority: built > tree > crop > bare)."""
    s = np.full(len(c), 'mixed', dtype=object)
    s[c.f_bare.to_numpy() >= 0.5] = 'bare'
    s[c.f_crop.to_numpy() >= 0.4] = 'crop'
    s[c.f_tree.to_numpy() >= 0.4] = 'tree'
    s[c.f_built.to_numpy() >= 0.5] = 'built'
    return s


# ---------------------------------------------------------------- what is already on disk
CACHE_KEYS = ['T0', 'N_MONTHS', 'CRS', 'RES', 'BUFFER_KM', 'MAX_SCENE_CLOUD', 'ST_QA_MAX',
              'EXCL_ISRAEL_KM', 'EXCL_EGYPT_KM', 'ELEV_MARGIN_M', 'DCOAST_MARGIN_KM']
CACHE_CFG = os.path.join(CFG['DATA_DIR'], 'cache_settings.json')
now_cfg = {k: CFG[k] for k in CACHE_KEYS}
if os.path.isfile(CACHE_CFG):
    old_cfg = json.load(open(CACHE_CFG))
    changed = {k: (old_cfg.get(k), v) for k, v in now_cfg.items() if old_cfg.get(k) != v}
    if changed:
        raise RuntimeError(f'Cached data were built with different settings {changed}. Either set them back, '
                           f'or delete the "{CFG["DATA_DIR"]}" folder (or point DATA_DIR somewhere new) to rebuild.')
else:
    json.dump(now_cfg, open(CACHE_CFG, 'w'), indent=1)

def month_ok(i):
    """True if month i is on disk and was downloaded late enough that no more scenes are expected."""
    f = month_file(i)
    if not os.path.isfile(f):
        return False
    with np.load(f) as z:
        saved = float(z['saved']) if 'saved' in z.files else time.time()
    month_end = (MONTHS[i] + pd.offsets.MonthBegin(1)).timestamp()
    settled = month_end + CFG['REFRESH_DAYS'] * 86400
    return saved >= settled or time.time() < settled    # too recent to know yet -> keep it for now

MONTHS_TODO = [i for i in range(CFG['N_MONTHS']) if not month_ok(i)]
status = {'grid.json': os.path.isfile(GRID_FILE), 'static_raw.npz': os.path.isfile(STATIC_FILE),
          'cells.pkl': os.path.isfile(CELLS_FILE),
          f'monthly ({CFG["N_MONTHS"] - len(MONTHS_TODO)}/{CFG["N_MONTHS"]})': not MONTHS_TODO}
for k, v in status.items():
    print(f'  {"on disk " if v else "MISSING "} {k}')
NEED_EE = not (os.path.isfile(GRID_FILE) and os.path.isfile(STATIC_FILE) and os.path.isfile(CELLS_FILE)
               and not MONTHS_TODO)
print('Earth Engine needed this run:', NEED_EE)



# ---------------------------------------------------------------- Earth Engine: boundaries and grid
GAZA_FALLBACK = [(34.2181, 31.3236), (34.2670, 31.2165), (34.3743, 31.3276), (34.3925, 31.3735),
                 (34.4108, 31.4195), (34.4451, 31.4537), (34.4953, 31.4762), (34.5257, 31.5133),
                 (34.5465, 31.5548), (34.4906, 31.5960)]

if NEED_EE:
    import ee
    try:
        ee.Initialize(project=CFG['PROJECT'])
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=CFG['PROJECT'])
    print('Earth Engine initialised with project', CFG['PROJECT'])

    gaul = ee.FeatureCollection('FAO/GAUL/2015/level0')
    GAZA = gaul.filter(ee.Filter.eq('ADM0_NAME', 'Gaza Strip'))
    if GAZA.size().getInfo() == 0:
        print('GAUL has no "Gaza Strip" unit here -> using the fallback polygon')
        GAZA = ee.FeatureCollection([ee.Feature(ee.Geometry.Polygon([GAZA_FALLBACK]))])
    ISRAEL = gaul.filter(ee.Filter.eq('ADM0_NAME', 'Israel'))
    EGYPT = gaul.filter(ee.Filter.eq('ADM0_NAME', 'Egypt'))
    print(f"Gaza area: {GAZA.geometry().area(100).getInfo() / 1e6:.0f} km2 (expect ~365)")

    if not os.path.isfile(GRID_FILE):
        bb = GAZA.geometry().bounds(100).coordinates().getInfo()[0]
        lo = [p[0] for p in bb]; la = [p[1] for p in bb]
        # buffer the lon/lat box generously, then snap to the 100 m lattice in UTM
        dlat = CFG['BUFFER_KM'] / 111.0; dlon = CFG['BUFFER_KM'] / (111.0 * math.cos(math.radians(31.4)))
        cx, cy = ll_to_utm(np.array([min(lo) - dlon, max(lo) + dlon, min(lo) - dlon, max(lo) + dlon]),
                           np.array([min(la) - dlat, min(la) - dlat, max(la) + dlat, max(la) + dlat]))
        R = CFG['RES']
        X0 = math.floor(cx.min() / R) * R; Y0 = math.ceil(cy.max() / R) * R
        grid = dict(X0=float(X0), Y0=float(Y0), W=int(math.ceil((cx.max() - X0) / R)),
                    H=int(math.ceil((Y0 - cy.min()) / R)), RES=R, CRS=CFG['CRS'])
        json.dump(grid, open(GRID_FILE, 'w'))

GRID = json.load(open(GRID_FILE))
print('grid:', GRID)

if NEED_EE:
    GRID_PROJ = ee.Projection(GRID['CRS'], [GRID['RES'], 0, GRID['X0'], 0, -GRID['RES'], GRID['Y0']])
    REGION = ee.Geometry.Rectangle([GRID['X0'], GRID['Y0'] - GRID['H'] * GRID['RES'],
                                    GRID['X0'] + GRID['W'] * GRID['RES'], GRID['Y0']],
                                   ee.Projection(GRID['CRS']), False)

    def ee_retry(fn, tries=6):
        for k in range(tries):
            try:
                return fn()
            except Exception as e:                         # 429 / timeouts: back off and retry
                if k == tries - 1:
                    raise
                wait = 5 * 2 ** k
                print(f'   EE error ({str(e)[:80]}...), retry in {wait}s')
                time.sleep(wait)

    def fetch(img, bands, max_bytes=12e6):   # lower max_bytes if a request times out
        """Pull an ee.Image onto GRID as numpy, splitting into row strips under the request limit."""
        H, W, R = GRID['H'], GRID['W'], GRID['RES']
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


# ---------------------------------------------------------------- static layers
if not os.path.isfile(STATIC_FILE):
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    fr = ee.Image.cat([wc.eq(v).rename(k) for k, v in WC_CLASSES.items()])
    fr = fr.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(GRID_PROJ)
    elev = (ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elev')
            .reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(GRID_PROJ))
    country = (ee.Image(0).paint(ISRAEL, 2).paint(EGYPT, 3).paint(GAZA, 1)
               .rename('country').reproject(GRID_PROJ))
    t_ = time.time()
    st = fetch(fr, list(WC_CLASSES))
    st.update(fetch(elev.addBands(country), ['elev', 'country']))
    np.savez_compressed(STATIC_FILE, **st)
    print(f'static layers fetched in {time.time() - t_:.0f}s')

if not os.path.isfile(CELLS_FILE):
    st = dict(np.load(STATIC_FILE))
    CELLS, _ = build_cells(GRID, st, CFG)
    CELLS['stratum0'] = baseline_stratum(CELLS)
    CELLS.to_pickle(CELLS_FILE)
CELLS = pd.read_pickle(CELLS_FILE)
print(CELLS.groupby(['country', 'stratum0']).size().unstack(fill_value=0))


# ---------------------------------------------------------------- monthly Landsat 8/9 composites
if MONTHS_TODO:
    L89 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
           .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')))

    def prep(img):
        qa = img.select('QA_PIXEL')
        bad = 0
        for bit in (0, 1, 2, 3, 4, 5, 7):
            bad = qa.bitwiseAnd(1 << bit).neq(0) if isinstance(bad, int) else bad.Or(qa.bitwiseAnd(1 << bit).neq(0))
        ok = (bad.Not()
              .And(img.select('QA_RADSAT').eq(0))
              .And(img.select('ST_QA').multiply(0.01).lte(CFG['ST_QA_MAX'])))
        sr = img.select(['SR_B2', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(0.0000275).add(-0.2)
        lst = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
        ndvi = sr.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        alb = sr.expression('0.356*b2 + 0.130*b4 + 0.373*b5 + 0.085*b6 + 0.072*b7 - 0.0018',
                            {'b2': sr.select('SR_B2'), 'b4': sr.select('SR_B4'), 'b5': sr.select('SR_B5'),
                             'b6': sr.select('SR_B6'), 'b7': sr.select('SR_B7')}).rename('ALB')
        emis = img.select('ST_EMIS').multiply(0.0001).rename('EMIS')
        return ee.Image.cat([lst, ndvi, alb, emis]).updateMask(ok)

    L30 = ee.Projection(GRID['CRS']).atScale(30)

    def monthly_image(i):
        s = MONTHS[i]; e = s + pd.offsets.MonthBegin(1)
        col = (L89.filterBounds(REGION).filterDate(str(s.date()), str(e.date()))
               .filter(ee.Filter.lt('CLOUD_COVER', CFG['MAX_SCENE_CLOUD'])).map(prep))
        n = col.size().getInfo()
        if n == 0:
            return None, 0
        mean = col.mean().setDefaultProjection(L30)
        cnt = col.select('LST').count().unmask(0).setDefaultProjection(L30)
        stack = mean.addBands(cnt.rename('N')).addBands(cnt.gt(0).rename('COV'))
        img = stack.reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(GRID_PROJ)
        return img, n

    def do_month(i):
        img, n = ee_retry(lambda: monthly_image(i))
        if img is None:
            full = {b: np.full((GRID['H'], GRID['W']), np.nan) for b in MBANDS}
        else:
            full = fetch(img, list(MBANDS))
        save_month(i, full, n, CELLS.flat.to_numpy())
        return i, f'{n} scenes'

    todo = MONTHS_TODO
    print(f'{CFG["N_MONTHS"] - len(todo)} months cached, {len(todo)} to fetch')
    t_ = time.time()
    with ThreadPoolExecutor(CFG['N_WORKERS']) as pool:
        futs = [pool.submit(do_month, i) for i in todo]
        for k, f in enumerate(as_completed(futs), 1):
            i, msg = f.result()
            print(f'  [{k}/{len(todo)}] {MONTHS[i]:%Y-%m}: {msg}  ({time.time() - t_:.0f}s)')
else:
    print('all 96 months already on disk -> nothing sent to Earth Engine')

print('done: data/grid.json, data/static_raw.npz, data/cells.pkl, data/monthly/*.npz')

