"""
Optimización tiempo-óptima de trayectoria para un circuito y año dados.

Uso:
    python scripts/04_optimizacion.py --year 2024 --circuit "Barcelona"
    python scripts/04_optimizacion.py --year 2024 --circuit "Bahrain" --session Q

Requiere:
    data/circuito_{slug}_{year}.csv  (generado por 03_reconstruccion.py)

Salida:
    data/trayectoria_tiempo_optima_{slug}_{year}.csv
"""

import argparse
import time
import warnings
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import interp1d as sp_interp1d
from scipy.interpolate import splprep, splev
from scipy.optimize import differential_evolution, minimize
from scipy.signal import savgol_filter

try:
    from numba import njit
    NUMBA = True
except ImportError:
    NUMBA = False

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
CACHE  = ROOT / "cache"
CONFIG = ROOT / "config" / "circuitos.yaml"

# ── Parámetros físicos fijos ──────────────────────────────────────────────────
G            = 9.81
k_downforce  = 0.0028
DRS_FACTOR   = 0.85
V_MIN        = 8.0
ANCHO_PISTA  = 12.0
MARGEN       = 0.5
d_max        = ANCHO_PISTA / 2.0 - MARGEN
N_CTRL       = 80
POP_SIZE     = 8
_PAD         = 25
MU_TARGET_OFFSET = 0.5   # t_ref ≈ POLE_TIME + offset (segundos)

# Defaults si el circuito no está en el YAML
DEFAULT_R_MIN_M       = 30.0
DEFAULT_V_MAX_DRS_KMH = None   # None → usar speed trap FastF1 o máximo telemetría


def _load_circuit_config(slug: str) -> dict:
    if not CONFIG.exists():
        return {}
    with open(CONFIG) as f:
        data = yaml.safe_load(f)
    circuits = data.get("circuits", {})
    if slug in circuits:
        return circuits[slug]
    slug_norm = slug.replace("_", "")
    for key, val in circuits.items():
        if key.replace("_", "") == slug_norm:
            return val
    return {}


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


# ── Numba forward-backward ────────────────────────────────────────────────────
if NUMBA:
    @njit(cache=True)
    def _fb_numba(v_cap, ds, kappa, a_lon_max, a_fren_max, k_d, k_df, mu_g,
                  vmax, vmin, drs_active, k_d_drs, vmax_drs):
        N = len(v_cap)
        v = v_cap.copy()
        for i in range(1, N):
            vp = v[i-1]
            kd_i = k_d_drs if drs_active[i-1] else k_d
            vm_i = vmax_drs if drs_active[i-1] else vmax
            a_lim = mu_g + k_df * vp * vp
            a_lat = vp * vp * kappa[i-1]
            if a_lat > a_lim: a_lat = a_lim
            a_lon_avail = (max(0.0, a_lim*a_lim - a_lat*a_lat)) ** 0.5
            ae = min(a_lon_max, a_lon_avail) - kd_i * vp * vp
            if ae < 0.0: ae = 0.0
            vn = (vp*vp + 2.0*ae*ds[i-1]) ** 0.5
            if vn > vm_i:     vn = vm_i
            if vn > v_cap[i]: vn = v_cap[i]
            v[i] = vn
        vb = v.copy()
        for i in range(N-2, -1, -1):
            vn = vb[i+1]
            kd_i = k_d_drs if drs_active[i+1] else k_d
            a_lim = mu_g + k_df * vn * vn
            a_lat = vn * vn * kappa[i+1]
            if a_lat > a_lim: a_lat = a_lim
            a_fren_avail = (max(0.0, a_lim*a_lim - a_lat*a_lat)) ** 0.5
            ae = min(a_fren_max, a_fren_avail) + kd_i * vn * vn
            vp = (vn*vn + 2.0*ae*ds[i]) ** 0.5
            if vp > v[i]: vp = v[i]
            vb[i] = vp
        return vb


def optimizar(year: int, circuit: str, session_type: str = "Q"):
    slug              = slugify(circuit)
    cfg               = _load_circuit_config(slug)
    r_min_m           = float(cfg.get("r_min_m",             DEFAULT_R_MIN_M))
    kappa_max         = 1.0 / r_min_m
    v_max_cfg         = cfg.get("v_max_drs_kmh",              DEFAULT_V_MAX_DRS_KMH)
    track_width_m     = float(cfg.get("track_width_m",        ANCHO_PISTA))
    d_max             = track_width_m / 2.0 - MARGEN
    mu_target_offset  = float(cfg.get("mu_target_offset_s",   MU_TARGET_OFFSET))
    k_drag_override   = cfg.get("k_drag",                     None)

    circuito_path = DATA / str(year) / f"circuito_{slug}_{year}.csv"
    if not circuito_path.exists():
        raise FileNotFoundError(
            f"No encontrado: {circuito_path}\n"
            f"Ejecuta primero: python scripts/03_reconstruccion.py --year {year} --circuit '{circuit}'"
        )

    circuito = pd.read_csv(circuito_path)
    x_ref    = circuito["x"].values
    y_ref    = circuito["y"].values
    dist_ref = circuito["dist"].values
    N        = len(x_ref)
    N_CTRL   = int(np.clip(dist_ref[-1] / 55.0, 80, 150))
    print(f"Circuito: {circuit} {year}  ({N} puntos, {dist_ref[-1]:.1f} m, N_CTRL={N_CTRL})")

    # ── Normales ──────────────────────────────────────────────────────────────
    dx = np.gradient(x_ref); dy = np.gradient(y_ref)
    L  = np.where(np.sqrt(dx**2 + dy**2) < 1e-9, 1e-9, np.sqrt(dx**2 + dy**2))
    nx, ny = -dy / L, dx / L

    # ── Telemetría y calibración ──────────────────────────────────────────────
    fastf1.Cache.enable_cache(str(CACHE))
    session = fastf1.get_session(year, circuit, session_type)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
    fastest_lap = session.laps.pick_fastest()
    tel      = fastest_lap.get_telemetry()
    v_tel    = tel["Speed"].values / 3.6
    dist_tel = tel["Distance"].values
    thr_tel  = tel["Throttle"].values
    brk_tel  = tel["Brake"].values.astype(float)
    drs_tel  = tel["DRS"].values.astype(float)
    ds_tel   = np.maximum(np.diff(dist_tel), 0.1)

    POLE_TIME = float(fastest_lap["LapTime"].total_seconds())
    print(f"Pole time: {POLE_TIME:.3f}s")

    # k_drag desde frenadas (DRS siempre cerrado al frenar)
    if k_drag_override is not None:
        k_drag = float(k_drag_override)
        print(f"k_drag override (YAML): {k_drag:.6f}")
    else:
        A_FREN_init = 2.5 * G * 1.30  # estimación inicial con MU=2.5
        frenada = (thr_tel[:-1] < 5) & (brk_tel[:-1] > 0.8) & (v_tel[:-1] > 50)
        k_drag = 0.0012
        if frenada.sum() > 5:
            a_obs    = np.diff(v_tel**2) / (2 * ds_tel)
            v_mid    = (v_tel[:-1] + v_tel[1:]) / 2
            muestras = -(a_obs[frenada] + A_FREN_init) / (v_mid[frenada]**2)
            muestras = muestras[muestras > 0]
            if len(muestras) > 3:
                k_drag = float(np.median(muestras))
        # Clamp: valores fuera de rango físico son ruido
        k_drag = float(np.clip(k_drag, 0.0008, 0.0020))
        print(f"k_drag calibrado: {k_drag:.6f}")

    # V_MAX_DRS: config YAML > speed trap FastF1 > máximo telemetría
    if v_max_cfg is not None:
        V_MAX_DRS = float(v_max_cfg) / 3.6
        src = "config YAML"
    else:
        V_MAX_DRS = v_tel.max()
        src = "telemetría"
        try:
            results = session.results
            st = results.loc[results["Abbreviation"] == fastest_lap["Driver"], "SpeedST"]
            if not st.empty and float(st.values[0]) > 0:
                V_MAX_DRS = float(st.values[0]) / 3.6
                src = "speed trap FastF1"
        except Exception:
            pass
    V_MAX = V_MAX_DRS * float(np.sqrt(DRS_FACTOR))
    print(f"V_MAX_DRS: {V_MAX_DRS*3.6:.1f} km/h ({src})  |  V_MAX: {V_MAX*3.6:.1f} km/h")
    print(f"R_min: {r_min_m:.0f} m  (KAPPA_MAX={kappa_max:.5f})")

    # Máscara DRS
    drs_mask = np.zeros(N, dtype=np.bool_)
    dist_clip  = np.clip(dist_ref, dist_tel[0], dist_tel[-1])
    drs_interp = sp_interp1d(dist_tel, drs_tel, kind="nearest",
                              bounds_error=False, fill_value=0.0)(dist_clip)
    drs_mask[:] = drs_interp >= 10
    print(f"DRS: {drs_mask.sum()} puntos ({drs_mask.sum()/N*100:.1f}%)")

    # ── Funciones del modelo ──────────────────────────────────────────────────
    N_INDIVIDUOS = POP_SIZE * N_CTRL
    idx_ctrl = np.linspace(0, N - 1, N_CTRL, dtype=int)
    t_ctrl   = idx_ctrl / (N - 1)
    t_all    = np.arange(N) / (N - 1)
    opt_bounds = [(-d_max, d_max)] * N_CTRL

    def expandir(n_ctrl):
        return np.interp(t_all, t_ctrl, n_ctrl)

    def calcular_curvatura(x, y):
        dx  = np.gradient(x);  dy  = np.gradient(y)
        ddx = np.gradient(dx); ddy = np.gradient(dy)
        den = (dx**2 + dy**2) ** 1.5
        den = np.where(den < 1e-9, 1e-9, den)
        return (dx * ddy - dy * ddx) / den

    def v_lateral_vec(kappa_abs, mu):
        kappa_safe = np.where(kappa_abs < 1e-6, 1e-6, kappa_abs)
        R          = 1.0 / kappa_safe
        vmax_local = np.where(drs_mask, V_MAX_DRS, V_MAX)
        v = np.full_like(R, 50.0)
        for _ in range(10):
            a_lim = mu * G + k_downforce * v**2
            v_new = np.sqrt(a_lim * R)
            v_new = np.clip(v_new, V_MIN, vmax_local)
            if np.max(np.abs(v_new - v)) < 0.01:
                break
            v = v_new
        return v_new

    if NUMBA:
        _d   = np.ones(10, dtype=np.float64)
        _ds  = np.full(9, 2.0, dtype=np.float64)
        _k   = np.full(10, 0.01, dtype=np.float64)
        _drs = np.zeros(10, dtype=np.bool_)
        _fb_numba(_d, _ds, _k, 22.0, 32.0, 0.0012, k_downforce,
                  24.5, 94.4, V_MIN, _drs, 0.0012 * DRS_FACTOR, 94.4)

    def forward_backward(v_cap, ds, kappa, mu):
        a_lon = mu * G * 0.90
        a_fren = mu * G * 1.30
        if NUMBA:
            return _fb_numba(
                v_cap.astype(np.float64), ds.astype(np.float64), kappa.astype(np.float64),
                a_lon, a_fren, k_drag, k_downforce, mu*G, V_MAX, V_MIN,
                drs_mask, k_drag * DRS_FACTOR, V_MAX_DRS,
            )
        Nl = len(v_cap)
        v  = v_cap.copy()
        for i in range(1, Nl):
            vp    = v[i-1]
            kd_i  = k_drag * DRS_FACTOR if drs_mask[i-1] else k_drag
            vm_i  = V_MAX_DRS if drs_mask[i-1] else V_MAX
            a_lim = mu*G + k_downforce * vp**2
            a_lat = min(vp**2 * kappa[i-1], a_lim)
            ae    = max(min(a_lon, (max(0.0, a_lim**2 - a_lat**2))**0.5) - kd_i*vp**2, 0.0)
            v[i]  = min((vp**2 + 2*ae*ds[i-1])**0.5, vm_i, v_cap[i])
        vb = v.copy()
        for i in range(Nl-2, -1, -1):
            vn    = vb[i+1]
            kd_i  = k_drag * DRS_FACTOR if drs_mask[i+1] else k_drag
            a_lim = mu*G + k_downforce * vn**2
            a_lat = min(vn**2 * kappa[i+1], a_lim)
            ae    = min(a_fren, (max(0.0, a_lim**2 - a_lat**2))**0.5) + kd_i*vn**2
            vb[i] = min((vn**2 + 2*ae*ds[i])**0.5, v[i])
        return vb

    def _prep(n_ctrl):
        n_full = expandir(n_ctrl)
        x_new  = x_ref + nx * n_full
        y_new  = y_ref + ny * n_full
        ds     = np.maximum(np.sqrt(np.diff(x_new)**2 + np.diff(y_new)**2), 1e-6)
        x_per  = np.concatenate([x_new[-_PAD:], x_new, x_new[:_PAD]])
        y_per  = np.concatenate([y_new[-_PAD:], y_new, y_new[:_PAD]])
        kappa  = savgol_filter(np.abs(calcular_curvatura(x_per, y_per)), 21, 3)
        kappa  = np.clip(kappa[_PAD:-_PAD], 0.0, kappa_max)
        return ds, kappa

    def tiempo_referencia(mu):
        ds, kappa = _prep(np.zeros(N_CTRL))
        v_lat = v_lateral_vec(kappa, mu)
        v     = forward_backward(v_lat, ds, kappa, mu)
        v_seg = np.maximum((v[:-1] + v[1:]) / 2.0, V_MIN)
        return float(np.sum(ds / v_seg))

    # ── Bisección de MU: t_ref ≈ POLE_TIME + mu_target_offset ───────────────
    target = POLE_TIME + mu_target_offset
    mu_lo, mu_hi = 1.2, 3.5
    for _ in range(20):
        mu_mid = (mu_lo + mu_hi) / 2
        t_mid  = tiempo_referencia(mu_mid)
        if t_mid > target:
            mu_lo = mu_mid
        else:
            mu_hi = mu_mid
        if abs(t_mid - target) < 0.02:
            break
    MU = mu_mid
    t_ref = t_mid
    print(f"MU calibrado: {MU:.4f}  →  t_ref={t_ref:.3f}s  ({t_ref - POLE_TIME:+.3f}s vs pole)")

    # ── Validación: velocidad de referencia vs telemetría ─────────────────────
    _ds_ref, _kappa_ref = _prep(np.zeros(N_CTRL))
    _v_ref = forward_backward(v_lateral_vec(_kappa_ref, MU), _ds_ref, _kappa_ref, MU)
    _v_tel_grid = np.interp(dist_ref,
                             np.clip(dist_tel, dist_ref[0], dist_ref[-1]), v_tel)
    _ratio = _v_ref.mean() / _v_tel_grid.mean()
    print(f"Validación ref vs tel: ratio medio {_ratio:.3f}  "
          f"|  v_min modelo: {_v_ref.min()*3.6:.1f} km/h  "
          f"|  v_min telemetría: {_v_tel_grid.min()*3.6:.1f} km/h  "
          f"(ratio esperado 0.97-1.00)")

    def objetivo(n_ctrl):
        ds, kappa = _prep(n_ctrl)
        v_lat = v_lateral_vec(kappa, MU)
        v     = forward_backward(v_lat, ds, kappa, MU)
        v_seg = np.maximum((v[:-1] + v[1:]) / 2.0, V_MIN)
        return float(np.sum(ds / v_seg))

    # ── Optimización ──────────────────────────────────────────────────────────
    rng      = np.random.default_rng(42)
    init_pop = rng.uniform(-d_max, d_max, (N_INDIVIDUOS, N_CTRL))
    init_pop[0] = np.zeros(N_CTRL)

    print("[Fase 1] Evolución Diferencial...")
    t0 = time.time()
    res_global = differential_evolution(
        objetivo, bounds=opt_bounds, maxiter=400, popsize=POP_SIZE, tol=1e-7,
        seed=42, init=init_pop, disp=False, workers=1,
        mutation=(0.5, 1.2), recombination=0.8,
    )
    t_de = time.time() - t0
    print(f"  {t_de:.1f}s  |  {res_global.fun:.3f}s  ({res_global.fun - POLE_TIME:+.3f}s vs pole)")

    print("[Fase 2] L-BFGS-B...")
    t1 = time.time()
    res_local = minimize(
        objetivo, res_global.x, method="L-BFGS-B", bounds=opt_bounds,
        options={"maxfun": 50000, "ftol": 1e-14, "gtol": 1e-9},
    )
    t_bfgs = time.time() - t1
    print(f"  {t_bfgs:.1f}s  |  {res_local.fun:.3f}s  ({res_local.fun - POLE_TIME:+.3f}s vs pole)")
    print(f"Total: {t_de+t_bfgs:.1f}s  |  Mejora vs referencia: {t_ref - res_local.fun:+.3f}s")

    n_opt = res_local.x

    # ── Post-procesado con spline analítica ───────────────────────────────────
    n_opt_full   = expandir(n_opt)
    n_opt_smooth = np.clip(savgol_filter(n_opt_full, 51, 3), -d_max, d_max)
    x_to = x_ref + nx * n_opt_smooth
    y_to = y_ref + ny * n_opt_smooth

    coords = np.column_stack([x_to, y_to])
    mask   = np.append(np.any(np.diff(coords, axis=0) != 0, axis=1), True)
    xc, yc = x_to[mask], y_to[mask]
    if not (np.isclose(xc[0], xc[-1]) and np.isclose(yc[0], yc[-1])):
        xc = np.append(xc, xc[0]); yc = np.append(yc, yc[0])
    tck_to, _ = splprep([xc, yc], s=len(xc) * 0.01, per=True, k=3)

    u_to = np.linspace(0, 1, N, endpoint=False)
    x_to, y_to = splev(u_to, tck_to)
    _dx, _dy   = splev(u_to, tck_to, der=1)
    _ddx, _ddy = splev(u_to, tck_to, der=2)
    _den       = np.where((_dx**2 + _dy**2)**1.5 < 1e-9, 1e-9, (_dx**2 + _dy**2)**1.5)
    kappa_to   = (_dx * _ddy - _dy * _ddx) / _den

    ds_to   = np.maximum(np.sqrt(np.diff(x_to)**2 + np.diff(y_to)**2), 1e-6)
    dist_to = np.concatenate([[0], np.cumsum(ds_to)])

    kappa_to_s = np.clip(savgol_filter(np.abs(kappa_to), 21, 3), 0.0, kappa_max)
    v_lat_to   = v_lateral_vec(kappa_to_s, MU)
    v_to       = forward_backward(v_lat_to, ds_to, kappa_to_s, MU)
    t_final    = float(np.sum(ds_to / np.maximum((v_to[:-1] + v_to[1:]) / 2.0, V_MIN)))

    print(f"\n=== RESULTADO FINAL ===")
    print(f"  Circuito:   {circuit} {year}")
    print(f"  MU:         {MU:.4f}  |  k_drag: {k_drag:.6f}  |  k_df: {k_downforce}")
    print(f"  R_min:      {r_min_m:.0f}m  |  ancho pista: {track_width_m:.0f}m  |  d_max: {d_max:.1f}m")
    print(f"  Pole:       {POLE_TIME:.3f}s")
    print(f"  Referencia: {t_ref:.3f}s  ({t_ref - POLE_TIME:+.3f}s)")
    print(f"  Óptima:     {t_final:.3f}s  ({t_final - POLE_TIME:+.3f}s)")
    print(f"  Mejora:     {t_ref - t_final:+.3f}s")
    print(f"  R_min efectivo: {1/kappa_to_s.max():.1f}m  |  v_min: {v_to.min()*3.6:.1f} km/h")
    print(f"  Tiempo cómputo: {t_de+t_bfgs:.0f}s  (DE: {t_de:.0f}s, L-BFGS-B: {t_bfgs:.0f}s)")

    # ── Guardar ───────────────────────────────────────────────────────────────
    year_dir = DATA / str(year)
    year_dir.mkdir(exist_ok=True)
    out = year_dir / f"trayectoria_tiempo_optima_{slug}_{year}.csv"
    pd.DataFrame({
        "x":                x_to,
        "y":                y_to,
        "kappa":            kappa_to,
        "dist":             dist_to,
        "n_desplazamiento": n_opt_smooth,
        "velocidad":        v_to,
    }).to_csv(out, index=False)
    print(f"✅ Guardado: {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--circuit", type=str, required=True)
    parser.add_argument("--session", type=str, default="Q")
    args = parser.parse_args()
    optimizar(args.year, args.circuit, args.session)


if __name__ == "__main__":
    main()
