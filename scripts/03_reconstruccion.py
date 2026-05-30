"""
Reconstrucción del circuito a partir de telemetría FastF1.

Uso:
    python scripts/03_reconstruccion.py --year 2024 --circuit "Barcelona"
    python scripts/03_reconstruccion.py --year 2024 --circuit "Bahrain" --session Q

Salida:
    data/circuito_{slug}_{year}.csv   (2000 puntos: x, y, kappa, dist)
"""

import argparse
import sys
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
DATA.mkdir(exist_ok=True)


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def calcular_curvatura_analitica(tck, u):
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    den = (dx**2 + dy**2) ** 1.5
    den = np.where(den < 1e-9, 1e-9, den)
    return (dx * ddy - dy * ddx) / den


def suavizar_trazada(x, y, n_puntos=2000, s_factor=0.25):
    coords = np.column_stack([x, y])
    mask = np.append(np.any(np.diff(coords, axis=0) != 0, axis=1), True)
    coords = coords[mask]
    xc, yc = coords[:, 0], coords[:, 1]
    if not (np.isclose(xc[0], xc[-1]) and np.isclose(yc[0], yc[-1])):
        xc = np.append(xc, xc[0])
        yc = np.append(yc, yc[0])
    n = len(xc)
    tck, _ = splprep([xc, yc], s=n * s_factor, per=True, k=3)
    u_new = np.linspace(0, 1, n_puntos, endpoint=False)
    xs, ys = splev(u_new, tck)
    return xs, ys, tck, u_new


def _cargar_sesion(year: int, circuit: str, session_type: str) -> fastf1.core.Session | None:
    try:
        s = fastf1.get_session(year, circuit, session_type)
        s.load()
        return s
    except Exception:
        return None


def _check_lluvia(session: fastf1.core.Session) -> None:
    weather = session.weather_data
    if weather is not None and not weather.empty and "Rainfall" in weather.columns:
        llovio = weather["Rainfall"].any()
        print(f"  Lluvia: {'SÍ ⚠️' if llovio else 'No'}")
    else:
        print("  Lluvia: datos no disponibles")


def reconstruir(year: int, circuit: str, session_type: str = "Q", n_top: int = 5):
    fastf1.Cache.enable_cache(str(CACHE))

    sessions_to_try = [session_type]
    if session_type == "Q":
        sessions_to_try.append("SQ")

    loaded_sessions = []
    for stype in sessions_to_try:
        print(f"Cargando sesión: {year} {circuit} {stype}")
        s = _cargar_sesion(year, circuit, stype)
        if s is None:
            print(f"  (no disponible)")
            continue
        _check_lluvia(s)
        laps = s.laps.pick_quicklaps()
        print(f"  Vueltas rápidas: {len(laps)}")
        loaded_sessions.append(s)

    if not loaded_sessions:
        print("ERROR: ninguna sesión disponible")
        sys.exit(1)

    # Mejor vuelta por piloto a través de todas las sesiones (conserva tipo Lap para get_telemetry)
    best_by_driver: dict[str, object] = {}
    for s in loaded_sessions:
        for driver in s.laps.pick_quicklaps()["Driver"].unique():
            lap = s.laps.pick_quicklaps().pick_drivers(driver).pick_fastest()
            if lap is None or pd.isna(lap["LapTime"]):
                continue
            if driver not in best_by_driver or lap["LapTime"] < best_by_driver[driver]["LapTime"]:
                best_by_driver[driver] = lap

    best_per_driver = list(best_by_driver.values())

    best_per_driver = [lap for lap in best_per_driver if pd.notna(lap["LapTime"])]
    best_per_driver.sort(key=lambda x: x["LapTime"])
    top_laps = best_per_driver[:n_top]

    print(f"Top {n_top} vueltas:")
    telemetries = []
    for lap in top_laps:
        try:
            tel = lap.get_telemetry()[["X", "Y", "Speed", "Distance"]].copy()
            tel["Driver"] = lap["Driver"]
            telemetries.append(tel)
            print(f"  {lap['Driver']}  {lap['LapTime']}  ({len(tel)} pts)")
        except Exception as e:
            print(f"  {lap['Driver']}  error: {e}")

    if not telemetries:
        print("ERROR: sin telemetría disponible")
        sys.exit(1)

    # Suavizar y remuestrear cada trazada a 2000 puntos
    trazadas = []  # (laptime_s, xs, ys)
    for lap, tel in zip(top_laps, telemetries):
        try:
            xs, ys, _, _ = suavizar_trazada(
                tel["X"].values / 10.0,
                tel["Y"].values / 10.0,
            )
            trazadas.append((lap["LapTime"].total_seconds(), xs, ys))
        except Exception as e:
            print(f"  {tel['Driver'].iloc[0]}  suavizado descartado: {e}")

    if not trazadas:
        print("ERROR: ninguna trazada válida")
        sys.exit(1)

    # Promedio ponderado por tiempo: pole=peso máximo, los demás decaen linealmente
    # Suaviza ruido GPS sin distorsionar chicanes como el promedio uniforme
    t_pole = trazadas[0][0]
    n = len(trazadas)
    pesos = np.array([1.0 - 0.6 * (t - t_pole) / t_pole * n for t, _, _ in trazadas])
    pesos = np.clip(pesos, 0.1, 1.0)
    pesos /= pesos.sum()

    xs_pole, ys_pole = trazadas[0][1], trazadas[0][2]
    alineadas = [(xs_pole, ys_pole)]
    for _, xs, ys in trazadas[1:]:
        dist = np.sqrt((xs - xs_pole[0]) ** 2 + (ys - ys_pole[0]) ** 2)
        offset = np.argmin(dist)
        alineadas.append((np.roll(xs, -offset), np.roll(ys, -offset)))

    x_centro = sum(p * xs for p, (xs, ys) in zip(pesos, alineadas))
    y_centro = sum(p * ys for p, (xs, ys) in zip(pesos, alineadas))
    print(f"Centerline: promedio ponderado ({n} trazadas, pesos {np.round(pesos, 2)})")

    # Re-splinear el centro para curvatura analítica
    _, _, tck_centro, u_centro = suavizar_trazada(x_centro, y_centro, s_factor=0.05)
    x_fin, y_fin = splev(u_centro, tck_centro)
    kappa = calcular_curvatura_analitica(tck_centro, u_centro)

    ds = np.sqrt(np.diff(x_fin) ** 2 + np.diff(y_fin) ** 2)
    dist = np.concatenate([[0], np.cumsum(ds)])

    print(f"Longitud circuito: {dist[-1]:.1f} m")
    print(f"Curvatura máx: {np.abs(kappa).max():.5f}  (R_min: {1/np.abs(kappa).max():.1f} m)")

    slug = slugify(circuit)
    year_dir = DATA / str(year)
    year_dir.mkdir(exist_ok=True)
    out = year_dir / f"circuito_{slug}_{year}.csv"
    pd.DataFrame({"x": x_fin, "y": y_fin, "kappa": kappa, "dist": dist}).to_csv(
        out, index=False
    )
    print(f"✅ Guardado: {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--circuit", type=str, required=True)
    parser.add_argument("--session", type=str, default="Q")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    reconstruir(args.year, args.circuit, args.session, args.top)


if __name__ == "__main__":
    main()
