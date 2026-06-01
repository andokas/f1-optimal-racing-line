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

ROOT  = Path(__file__).parent.parent
DATA  = ROOT / "data"
WEB   = ROOT / "web"
CACHE = ROOT / "cache"

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

# Nombre display, orden cronológico y año de datos efectivo
CIRCUIT_META = {
    "australia":      {"name": "Australia",     "order":  1},
    "china":          {"name": "China",          "order":  2},
    "japan":          {"name": "Japan",          "order":  3},
    "bahrain":        {"name": "Bahrain",        "order":  4},
    "saudi_arabia":   {"name": "Saudi Arabia",   "order":  5},
    "miami":          {"name": "Miami",          "order":  6},
    "emilia_romagna": {"name": "Emilia Romagna", "order":  7},
    "monaco":         {"name": "Monaco",         "order":  8},
    "spain":          {"name": "Spain",          "order":  9},
    "canada":         {"name": "Canada",         "order": 10},
    "austria":        {"name": "Austria",        "order": 11},
    "silverstone":    {"name": "Silverstone",    "order": 12},
    "belgium":        {"name": "Belgium",        "order": 13},
    "hungary":        {"name": "Hungary",        "order": 14},
    "netherlands":    {"name": "Netherlands",    "order": 15},
    "italy":          {"name": "Italy",          "order": 16},
    "azerbaijan":     {"name": "Azerbaijan",     "order": 17},
    "singapore":      {"name": "Singapore",      "order": 18},
    "united_states":  {"name": "United States",  "order": 19},
    "mexico":         {"name": "Mexico",         "order": 20},
    "brazil":         {"name": "Brazil",         "order": 21},
    "las_vegas":      {"name": "Las Vegas",      "order": 22},
    "qatar":          {"name": "Qatar",          "order": 23},
    "abu_dhabi":      {"name": "Abu Dhabi",      "order": 24},
}

# Circuitos que usan datos de un año distinto al de la temporada
DATA_YEAR_OVERRIDE = {
    "las_vegas": 2024,
}


def _round(arr, n):
    return [round(float(v), n) for v in arr]


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
        meta = CIRCUIT_META.get(slug, {"name": slug.replace("_", " ").title(), "order": 99})

        note = f" (datos {data_year})" if data_year != year else ""
        entry = {
            "slug":      slug,
            "name":      meta["name"],
            "order":     meta["order"],
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
            pole_str = f"  pole={pole_tel['driver']} {pole_tel['lap_time']:.3f}s  diff={entry['diff_vs_pole']:+.3f}s" if pole_tel else ""
            print(f"  ✓  {meta['name']:20s}{note}  v=[{entry['opt']['v_min']:.0f}–{entry['opt']['v_max']:.0f}] km/h{pole_str}")
        else:
            entry["opt"] = None
            entry["pole_tel"] = None
            print(f"  ⚠  {meta['name']:20s}{note}  sin trayectoria óptima")

        circuits.append(entry)

    circuits.sort(key=lambda c: c["order"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Guardamos como JS para que funcione con doble clic sin servidor
    js_path = out_path.with_suffix(".js")
    payload = json.dumps({"year": year, "circuits": circuits}, separators=(",", ":"))
    with open(js_path, "w") as f:
        f.write(f"window.F1_DATA={payload};")

    size_mb = js_path.stat().st_size / 1e6
    print(f"\n✅ {len(circuits)} circuitos → {js_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out",  type=str, default=None)
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else WEB / "data.json"
    print(f"Exportando temporada {args.year} → {out_path}")
    exportar(args.year, out_path)


if __name__ == "__main__":
    main()
