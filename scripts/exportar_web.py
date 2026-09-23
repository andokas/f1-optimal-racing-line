"""
Exporta todos los datos de una temporada a un JSON estático para la web.

Uso:
    python scripts/exportar_web.py --year 2025
    python scripts/exportar_web.py --year 2025 --out web/data.json

Salida:
    web/data.json
"""

import argparse
import json
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
import yaml

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
WEB    = ROOT / "web"
CACHE  = ROOT / "cache"
CONFIG = ROOT / "config" / "circuitos.yaml"

G = 9.81
K_DF_LO, K_DF_HI = 0.0018, 0.0035
V_DF_LO, V_DF_HI = 170.0, 260.0
DRS_FACTOR = 0.85
MARGEN = 0.5
DEFAULT_ANCHO = 12.0
DEFAULT_R_MIN = 30.0


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

# Nombre FastF1 para cada slug (usado para cargar telemetría de la pole)
FASTF1_NAME = {
    "australia":      "Australia",
    "china":          "China",
    "japan":          "Japan",
    "bahrain":        "Bahrain",
    "saudi_arabia":   "Saudi Arabia",
    "miami":          "Miami",
    "emilia_romagna": "Emilia Romagna",
    "monaco":         "Monaco",
    "spain":          "Spain",
    "canada":         "Canada",
    "austria":        "Austria",
    "silverstone":    "Great Britain",
    "belgium":        "Belgium",
    "hungary":        "Hungary",
    "netherlands":    "Netherlands",
    "italy":          "Italy",
    "azerbaijan":     "Azerbaijan",
    "singapore":      "Singapore",
    "united_states":  "United States",
    "mexico":         "Mexico",
    "brazil":         "Brazil",
    "las_vegas":      "Las Vegas",
    "qatar":          "Qatar",
    "abu_dhabi":      "Abu Dhabi",
}

# Nombre display (el orden se obtiene del calendario FastF1)
CIRCUIT_NAMES = {
    "australia":      "Australia",
    "china":          "China",
    "japan":          "Japan",
    "bahrain":        "Bahrain",
    "saudi_arabia":   "Saudi Arabia",
    "miami":          "Miami",
    "emilia_romagna": "Emilia Romagna",
    "monaco":         "Monaco",
    "spain":          "Spain",
    "canada":         "Canada",
    "austria":        "Austria",
    "silverstone":    "Silverstone",
    "belgium":        "Belgium",
    "hungary":        "Hungary",
    "netherlands":    "Netherlands",
    "italy":          "Italy",
    "azerbaijan":     "Azerbaijan",
    "singapore":      "Singapore",
    "united_states":  "United States",
    "mexico":         "Mexico",
    "brazil":         "Brazil",
    "las_vegas":      "Las Vegas",
    "qatar":          "Qatar",
    "abu_dhabi":      "Abu Dhabi",
}


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _get_calendar_order(year: int) -> dict[str, int]:
    """Devuelve {slug: round_number} desde el calendario FastF1."""
    try:
        fastf1.Cache.enable_cache(str(CACHE))
        schedule = fastf1.get_event_schedule(year)
        order = {}
        for _, row in schedule.iterrows():
            rn = int(row["RoundNumber"])
            if rn == 0:
                continue
            event_name = str(row["EventName"])
            slug = _slugify(row["Location"]) if "Location" in row else _slugify(event_name)
            for our_slug, ff1_name in FASTF1_NAME.items():
                if ff1_name.lower() in event_name.lower() or ff1_name.lower() in slug:
                    order[our_slug] = rn
                    break
            else:
                order[slug] = rn
        return order
    except Exception as e:
        print(f"  ⚠  No se pudo obtener calendario {year}: {e}")
        return {}

# Circuitos que usan datos de un año distinto al de la temporada
DATA_YEAR_OVERRIDE = {
    "las_vegas": 2024,
}


def _round(arr, n):
    return [round(float(v), n) for v in arr]


def _extract_model_params(slug: str, data_year: int, opt_dist, opt_v_ms, opt_kappa) -> dict | None:
    """Extract model params and DRS mask from telemetry (lightweight calibration)."""
    fastf1_name = FASTF1_NAME.get(slug)
    if not fastf1_name:
        return None
    try:
        cfg = _load_circuit_config(slug)
        r_min_m = float(cfg.get("r_min_m", DEFAULT_R_MIN))
        track_w = float(cfg.get("track_width_m", DEFAULT_ANCHO))
        d_max = track_w / 2.0 - MARGEN

        s = fastf1.get_session(data_year, fastf1_name, "Q")
        s.load(telemetry=True, laps=True, weather=False, messages=False)
        lap = s.laps.pick_fastest()
        tel = lap.get_telemetry()
        v_tel = tel["Speed"].values / 3.6
        dist_tel = tel["Distance"].values
        drs_tel = tel["DRS"].values.astype(float)

        pole_time = float(lap["LapTime"].total_seconds())
        v_media_kmh = float(np.mean(v_tel) * 3.6)
        t_df = np.clip((v_media_kmh - V_DF_LO) / (V_DF_HI - V_DF_LO), 0.0, 1.0)
        k_downforce = float(K_DF_LO + t_df * (K_DF_HI - K_DF_LO))

        N = len(opt_dist)
        dist_clip = np.clip(opt_dist, dist_tel[0], dist_tel[-1])
        drs_interp = np.interp(dist_clip, dist_tel, drs_tel)
        drs_mask = (drs_interp >= 12).astype(float)
        drs_pct = round(float(drs_mask.sum() / N * 100), 1)

        # Subsample DRS mask to ~200 points for web (boolean-ish)
        step = max(1, N // 200)
        drs_sub = [int(drs_mask[i]) for i in range(0, N, step)]
        dist_sub = [round(float(opt_dist[i]), 1) for i in range(0, N, step)]

        return {
            "r_min_m": r_min_m,
            "d_max": round(d_max, 1),
            "track_width_m": track_w,
            "k_downforce": round(k_downforce, 4),
            "v_media_kmh": round(v_media_kmh, 1),
            "drs_pct": drs_pct,
            "drs_dist": dist_sub,
            "drs_mask": drs_sub,
            "pole_time": round(pole_time, 3),
        }
    except Exception as e:
        print(f"    ⚠  params no disponibles: {e}")
        return None


def _cargar_telemetria_pole(slug: str, data_year: int) -> dict | None:
    fastf1_name = FASTF1_NAME.get(slug)
    if not fastf1_name:
        return None
    try:
        fastf1.Cache.enable_cache(str(CACHE))
        s = fastf1.get_session(data_year, fastf1_name, "Q")
        s.load(telemetry=True, laps=True, weather=False, messages=False)
        lap = s.laps.pick_fastest()
        tel = lap.get_telemetry()[["Speed", "Distance", "X", "Y"]]
        driver = lap["Driver"]
        lap_time = float(lap["LapTime"].total_seconds())

        # Submuestrear posición a ~400 puntos (elimina duplicados GPS primero)
        xy = tel[["X", "Y"]].values / 10.0
        dist_arr = tel["Distance"].values
        mask = np.concatenate([[True], np.any(np.diff(xy, axis=0) != 0, axis=1)])
        xy_clean = xy[mask]
        dist_clean = dist_arr[mask]
        step = max(1, len(xy_clean) // 400)
        xy_sub = xy_clean[::step]
        dist_sub = dist_clean[::step]

        # ── Sectores: índices en el array subsampled ──────────────────────────
        sector_indices = []
        try:
            t1 = float(lap["Sector1Time"].total_seconds())
            t2 = float(lap["Sector2Time"].total_seconds())
            t_rel = (tel.index - tel.index[0]).total_seconds().values
            idx1 = int(np.searchsorted(t_rel, t1))
            idx2 = int(np.searchsorted(t_rel, t1 + t2))
            d1 = float(dist_arr[min(idx1, len(dist_arr)-1)])
            d2 = float(dist_arr[min(idx2, len(dist_arr)-1)])
            sector_indices = [
                int(np.searchsorted(dist_sub, d1)),
                int(np.searchsorted(dist_sub, d2)),
            ]
        except Exception:
            pass

        # ── Números de curva con posición XY ──────────────────────────────────
        corners = []
        try:
            ci = s.get_circuit_info()
            for _, row in ci.corners.iterrows():
                label = str(int(row["Number"])) + (row["Letter"] if pd.notna(row["Letter"]) and row["Letter"] else "")
                corners.append({
                    "n":    label,
                    "x":    round(float(row["X"]) / 10.0, 1),
                    "y":    round(float(row["Y"]) / 10.0, 1),
                    "dist": round(float(row["Distance"]), 1),
                })
        except Exception:
            pass

        return {
            "dist":         _round(dist_arr, 1),
            "v_kmh":        _round(tel["Speed"].values, 1),
            "driver":       driver,
            "lap_time":     round(lap_time, 3),
            "raw_x":           _round(xy_sub[:, 0], 1),
            "raw_y":           _round(xy_sub[:, 1], 1),
            "sector_indices":  sector_indices,
            "corners":         corners,
        }
    except Exception as e:
        print(f"    ⚠  telemetría pole no disponible: {e}")
        return None


def exportar(year: int, out_path: Path):
    year_dir = DATA / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"No hay datos para {year} en {year_dir}")

    calendar = _get_calendar_order(year)

    slugs = sorted(set(
        [p.stem.replace("circuito_", "").replace(f"_{year}", "")
         for p in year_dir.glob(f"circuito_*_{year}.csv")]
        + list(DATA_YEAR_OVERRIDE.keys())
    ))

    circuits = []
    for slug in slugs:
        data_year = DATA_YEAR_OVERRIDE.get(slug, year)
        data_dir  = DATA / str(data_year)
        ref_path  = data_dir / f"circuito_{slug}_{data_year}.csv"
        opt_path  = data_dir / f"trayectoria_tiempo_optima_{slug}_{data_year}.csv"

        if not ref_path.exists():
            print(f"  ✗  {slug}: falta {ref_path}")
            continue

        ref  = pd.read_csv(ref_path)
        display_name = CIRCUIT_NAMES.get(slug, slug.replace("_", " ").title())
        race_order = calendar.get(slug, 99)

        note = f" (datos {data_year})" if data_year != year else ""
        entry = {
            "slug":      slug,
            "name":      display_name,
            "order":     race_order,
            "year":      year,
            "data_year": data_year,
            "ref": {
                "x":    _round(ref["x"], 2),
                "y":    _round(ref["y"], 2),
                "dist": _round(ref["dist"], 1),
            },
        }

        if opt_path.exists():
            opt   = pd.read_csv(opt_path)
            v_kmh = opt["velocidad"].values * 3.6
            kappa_abs = np.abs(opt["kappa"].values)
            radius = np.where(kappa_abs > 1e-6, 1.0 / kappa_abs, 9999.0)
            step_k = max(1, len(kappa_abs) // 400)
            entry["opt"] = {
                "x":        _round(opt["x"], 2),
                "y":        _round(opt["y"], 2),
                "dist":     _round(opt["dist"], 1),
                "v_kmh":    _round(v_kmh, 1),
                "v_min":    round(float(v_kmh.min()), 1),
                "v_max":    round(float(v_kmh.max()), 1),
                "v_mean":   round(float(v_kmh.mean()), 1),
                "length_m": round(float(opt["dist"].iloc[-1]), 1),
                "n_desp":   _round(opt["n_desplazamiento"], 3),
                "radius":   _round(np.clip(radius[::step_k], 0, 500), 1),
                "radius_dist": _round(opt["dist"].values[::step_k], 1),
            }
            ds       = np.diff(opt["dist"].values)
            v_ms     = opt["velocidad"].values
            v_seg    = np.maximum((v_ms[:-1] + v_ms[1:]) / 2.0, 8.0 / 3.6)
            opt_time = round(float(np.sum(ds / v_seg)), 3)

            pole_tel = _cargar_telemetria_pole(slug, data_year)
            entry["pole_tel"]     = pole_tel
            entry["optimal_time"] = opt_time
            if pole_tel:
                entry["pole_time"]    = pole_tel["lap_time"]
                entry["diff_vs_pole"] = round(opt_time - pole_tel["lap_time"], 3)
            else:
                entry["pole_time"]    = None
                entry["diff_vs_pole"] = None

            params = _extract_model_params(slug, data_year, opt["dist"].values, v_ms, kappa_abs)
            entry["model"] = params

            pole_str = f"  pole={pole_tel['driver']} {pole_tel['lap_time']:.3f}s  diff={entry['diff_vs_pole']:+.3f}s" if pole_tel else ""
            print(f"  ✓  {display_name:20s}{note}  v=[{entry['opt']['v_min']:.0f}–{entry['opt']['v_max']:.0f}] km/h{pole_str}")
        else:
            entry["opt"] = None
            entry["pole_tel"] = None
            print(f"  ⚠  {display_name:20s}{note}  sin trayectoria óptima")

        circuits.append(entry)

    circuits.sort(key=lambda c: c["order"])
    return circuits


def exportar_multi(years: list[int], out_path: Path):
    all_circuits = []
    for year in years:
        print(f"\n── Temporada {year} ──")
        all_circuits.extend(exportar(year, out_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    js_path = out_path.with_suffix(".js")
    payload = json.dumps({"years": sorted(years), "circuits": all_circuits}, separators=(",", ":"))
    with open(js_path, "w") as f:
        f.write(f"window.F1_DATA={payload};")

    size_mb = js_path.stat().st_size / 1e6
    print(f"\n✅ {len(all_circuits)} circuitos ({len(years)} temporadas) → {js_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, nargs="+", required=True)
    parser.add_argument("--out",  type=str, default=None)
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else WEB / "data.json"
    print(f"Exportando temporadas {args.year} → {out_path}")
    exportar_multi(args.year, out_path)


if __name__ == "__main__":
    main()
