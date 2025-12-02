from fastapi import APIRouter, Request, Query
from fastapi.templating import Jinja2Templates
from datetime import date
from urllib.parse import urlencode
import pandas as pd
import numpy as np

from db import query_df, table_exists
from routers.filters import MOMENT_ORDER

router = APIRouter(tags=["sessions"])
templates = Jinja2Templates(directory="templates")

# Cache pour la stratégie de récupération des données Vehicle
_vehicle_strategy_cache = None


def _get_vehicle_strategy():
    """
    Détermine la stratégie optimale pour récupérer les données Vehicle.
    Cette fonction est appelée une seule fois au premier accès et met en cache le résultat.

    Returns:
        tuple: (vehicle_select, join_clause) pour la requête SQL
    """
    global _vehicle_strategy_cache

    if _vehicle_strategy_cache is not None:
        return _vehicle_strategy_cache

    if table_exists("kpi_charges_mac"):
        # Vehicle existe dans kpi_charges_mac, faire un LEFT JOIN sur ID
        # JOIN sur ID est plus fiable que MAC Address + Datetime
        _vehicle_strategy_cache = (
            "c.Vehicle",
            """
            LEFT JOIN kpi_charges_mac c
                ON k.ID = c.ID
        """
        )
    else:
        # Aucune source de données Vehicle disponible
        _vehicle_strategy_cache = ("NULL AS Vehicle", "")

    return _vehicle_strategy_cache


def _build_conditions(sites: str, date_debut: date | None, date_fin: date | None, table_alias: str | None = None):
    conditions = ["1=1"]
    params = {}

    # Add table qualifier if provided
    datetime_col = f"{table_alias}.`Datetime start`" if table_alias else "`Datetime start`"
    site_col = f"{table_alias}.Site" if table_alias else "Site"

    if date_debut:
        conditions.append(f"{datetime_col} >= :date_debut")
        params["date_debut"] = str(date_debut)
    if date_fin:
        conditions.append(f"{datetime_col} < DATE_ADD(:date_fin, INTERVAL 1 DAY)")
        params["date_fin"] = str(date_fin)
    if sites:
        site_list = [s.strip() for s in sites.split(",") if s.strip()]
        if site_list:
            placeholders = ",".join([f":site_{i}" for i in range(len(site_list))])
            conditions.append(f"{site_col} IN ({placeholders})")
            for i, s in enumerate(site_list):
                params[f"site_{i}"] = s

    return " AND ".join(conditions), params


def _apply_status_filters(df: pd.DataFrame, error_type_list: list[str], moment_list: list[str]) -> pd.DataFrame:
    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)
    mask_nok = ~df["is_ok"]
    mask_type = (
        df["type_erreur"].isin(error_type_list)
        if error_type_list and "type_erreur" in df.columns
        else pd.Series(True, index=df.index)
    )
    mask_moment = (
        df["moment"].isin(moment_list)
        if moment_list and "moment" in df.columns
        else pd.Series(True, index=df.index)
    )
    df["is_ok_filt"] = np.where(mask_nok & mask_type & mask_moment, False, True)
    return df


def _comparaison_base_context(
    request: Request,
    filters: dict,
    site_focus: str = "",
    month_focus: str = "",
    error_message: str | None = None,
):
    return {
        "request": request,
        "site_rows": [],
        "count_bars": [],
        "percent_bars": [],
        "max_total": 0,
        "peak_rows": [],
        "heatmap_rows": [],
        "heatmap_hours": [],
        "heatmap_max": 0,
        "site_options": [],
        "site_focus": site_focus,
        "month_options": [],
        "month_focus": month_focus,
        "monthly_rows": [],
        "daily_rows": [],
        "filters": filters,
        "error_message": error_message,
    }


@router.get("/sessions/stats")
async def get_sessions_stats(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
):
    """
    Retourne les statistiques complètes des sessions (énergie, puissance, SOC, durées, etc.)
    """
    error_type_list = [e.strip() for e in error_types.split(",") if e.strip()] if error_types else []
    moment_list = [m.strip() for m in moments.split(",") if m.strip()] if moments else []

    where_clause, params = _build_conditions(sites, date_debut, date_fin, table_alias="k")

    # Récupération complète des données avec toutes les colonnes nécessaires
    vehicle_select, join_clause = _get_vehicle_strategy()

    sql = f"""
        SELECT
            k.ID,
            k.Site,
            k.PDC,
            k.`Datetime start`,
            k.`Datetime end`,
            k.`Energy (Kwh)`,
            k.`Mean Power (Kw)`,
            k.`Max Power (Kw)`,
            k.`SOC Start`,
            k.`SOC End`,
            k.`MAC Address`,
            k.`State of charge(0:good, 1:error)` as state,
            k.type_erreur,
            k.moment,
            {vehicle_select}
        FROM kpi_sessions k
        {join_clause}
        WHERE {where_clause}
    """

    df = query_df(sql, params)

    if df.empty:
        return templates.TemplateResponse(
            "partials/sessions_stats.html",
            {
                "request": request,
                "no_data": True,
            }
        )

    # Conversion des types
    for col in ["Datetime start", "Datetime end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["Energy (Kwh)", "Mean Power (Kw)", "Max Power (Kw)", "SOC Start", "SOC End"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Déterminer is_ok
    df["is_ok_raw"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)

    # Filtrage par type d'erreur et moment
    df = _apply_status_filters(df, error_type_list, moment_list)

    # Séparer OK et NOK
    ok_mask = df["is_ok_filt"]
    nok_mask = ~df["is_ok_filt"]

    ok_df = df[ok_mask].copy()
    nok_df = df[nok_mask].copy()

    # === STATISTIQUES D'ÉNERGIE ===
    energy_all = pd.to_numeric(df.get("Energy (Kwh)", pd.Series(dtype=float)), errors="coerce")
    e_total_all = round(float(energy_all.sum(skipna=True)), 3) if energy_all.notna().any() else 0

    # Pour moyenne et max : utiliser uniquement les charges OK
    energy_ok = pd.to_numeric(ok_df.get("Energy (Kwh)", pd.Series(dtype=float)), errors="coerce")
    e_mean = round(float(energy_ok.mean(skipna=True)), 3) if energy_ok.notna().any() else 0
    e_max = round(float(energy_ok.max(skipna=True)), 3) if energy_ok.notna().any() else 0

    # === STATISTIQUES DE PUISSANCE MOYENNE ===
    pmean_ok = pd.to_numeric(ok_df.get("Mean Power (Kw)", pd.Series(dtype=float)), errors="coerce")
    pm_mean = round(float(pmean_ok.mean(skipna=True)), 3) if pmean_ok.notna().any() else 0
    pm_max = round(float(pmean_ok.max(skipna=True)), 3) if pmean_ok.notna().any() else 0

    # === STATISTIQUES DE PUISSANCE MAXIMALE ===
    pmax_ok = pd.to_numeric(ok_df.get("Max Power (Kw)", pd.Series(dtype=float)), errors="coerce")
    px_mean = round(float(pmax_ok.mean(skipna=True)), 3) if pmax_ok.notna().any() else 0
    px_max = round(float(pmax_ok.max(skipna=True)), 3) if pmax_ok.notna().any() else 0

    # === STATISTIQUES SOC ===
    soc_start = pd.to_numeric(ok_df.get("SOC Start", pd.Series(dtype=float)), errors="coerce")
    soc_end = pd.to_numeric(ok_df.get("SOC End", pd.Series(dtype=float)), errors="coerce")
    soc_start_mean = round(float(soc_start.mean(skipna=True)), 2) if soc_start.notna().any() else 0
    soc_end_mean = round(float(soc_end.mean(skipna=True)), 2) if soc_end.notna().any() else 0

    if soc_start.notna().any() and soc_end.notna().any():
        soc_gain_mean = round(float((soc_end - soc_start).mean(skipna=True)), 2)
    else:
        soc_gain_mean = 0

    # === DURÉES DE CHARGE ===
    dt_start = pd.to_datetime(ok_df.get("Datetime start"), errors="coerce")
    dt_end = pd.to_datetime(ok_df.get("Datetime end"), errors="coerce")
    durations = (dt_end - dt_start).dt.total_seconds() / 60  # minutes
    dur_mean = round(float(durations.mean(skipna=True)), 1) if durations.notna().any() else 0

    # === CHARGES PAR SITE ===
    if not ok_df.empty:
        ok_df["day"] = pd.to_datetime(ok_df["Datetime start"]).dt.date
        charges_by_site_day = (
            ok_df.groupby(["Site", "day"])
            .size()
            .reset_index(name="Nb")
        )

        # Statistiques par jour
        daily_stats = charges_by_site_day.groupby("day")["Nb"].sum().reset_index()
        nb_days = len(daily_stats)
        mean_day = round(float(daily_stats["Nb"].mean()), 2) if nb_days else 0
        med_day = round(float(daily_stats["Nb"].median()), 2) if nb_days else 0

        # Max par jour
        if not charges_by_site_day.empty:
            max_row = charges_by_site_day.loc[charges_by_site_day["Nb"].idxmax()]
            max_day_site = str(max_row["Site"])
            max_day_date = str(max_row["day"])
            max_day_nb = int(max_row["Nb"])
        else:
            max_day_site = "—"
            max_day_date = "—"
            max_day_nb = 0
    else:
        nb_days = 0
        mean_day = 0
        med_day = 0
        max_day_site = "—"
        max_day_date = "—"
        max_day_nb = 0

    # === DURÉES DE FONCTIONNEMENT PAR SITE ===
    if not ok_df.empty and "Datetime start" in ok_df.columns and "Datetime end" in ok_df.columns:
        dur_df = ok_df[["Site", "PDC", "Datetime start", "Datetime end"]].copy()
        dur_df = dur_df.dropna(subset=["Datetime start", "Datetime end"])
        dur_df["dur_min"] = (
            pd.to_datetime(dur_df["Datetime end"]) - pd.to_datetime(dur_df["Datetime start"])
        ).dt.total_seconds() / 60

        by_site_dur = (
            dur_df.groupby("Site")["dur_min"]
            .sum()
            .reset_index()
            .assign(Heures=lambda d: (d["dur_min"] / 60).round(1))
            .sort_values("Heures", ascending=False)
        )
        durations_by_site = by_site_dur[["Site", "Heures"]].to_dict("records")

        by_pdc_dur = (
            dur_df.groupby(["Site", "PDC"])["dur_min"]
            .sum()
            .reset_index()
            .assign(Heures=lambda d: (d["dur_min"] / 60).round(1))
        )
        durations_by_pdc_raw = by_pdc_dur.to_dict("records")
    else:
        durations_by_site = []
        durations_by_pdc_raw = []

    # Grouper par site pour le sélecteur
    durations_by_site_dict = {}
    for row in durations_by_pdc_raw:
        site = row["Site"]
        if site not in durations_by_site_dict:
            durations_by_site_dict[site] = []
        durations_by_site_dict[site].append({
            "PDC": row["PDC"],
            "Heures": row["Heures"]
        })

    # Trier les PDC par durée décroissante dans chaque site
    for site in durations_by_site_dict:
        durations_by_site_dict[site] = sorted(
            durations_by_site_dict[site],
            key=lambda x: x["Heures"],
            reverse=True
        )

    # Conserver l'ordre des sites affichés dans le tableau principal
    site_options_order = [row["Site"] for row in durations_by_site]

    # === STATISTIQUES PAR TYPE DE VÉHICULE ===
    vehicle_stats = []
    vehicle_debug_info = {
        "has_column": "Vehicle" in df.columns,
        "total_rows": len(df),
        "non_null_count": 0,
        "valid_count": 0,
        "unknown_count": 0,
    }

    if "Vehicle" in df.columns and not df.empty:
        # Nettoyer les données Vehicle
        df_vehicle = df.copy()
        df_vehicle["Vehicle"] = df_vehicle["Vehicle"].astype(str).str.strip()

        # Compter les non-NULL avant nettoyage
        vehicle_debug_info["non_null_count"] = int(df_vehicle["Vehicle"].notna().sum())

        df_vehicle["Vehicle"] = df_vehicle["Vehicle"].replace(
            {"": "Unknown", "nan": "Unknown", "none": "Unknown", "NULL": "Unknown", "None": "Unknown"},
            regex=False
        )
        df_vehicle["Vehicle"] = df_vehicle["Vehicle"].fillna("Unknown")

        # Compter les Unknown
        vehicle_debug_info["unknown_count"] = int((df_vehicle["Vehicle"] == "Unknown").sum())

        # Exclure les véhicules inconnus
        df_vehicle = df_vehicle[df_vehicle["Vehicle"] != "Unknown"]
        vehicle_debug_info["valid_count"] = len(df_vehicle)

        if not df_vehicle.empty:
            # Grouper par véhicule et calculer les statistiques
            vehicle_grouped = (
                df_vehicle.groupby("Vehicle", dropna=False)["is_ok_filt"]
                .agg(total="size", ok="sum")
                .reset_index()
            )
            vehicle_grouped["nok"] = vehicle_grouped["total"] - vehicle_grouped["ok"]
            vehicle_grouped["percent_ok"] = np.where(
                vehicle_grouped["total"] > 0,
                (vehicle_grouped["ok"] / vehicle_grouped["total"] * 100).round(2),
                0.0
            )
            vehicle_grouped["percent_nok"] = 100 - vehicle_grouped["percent_ok"]

            # Trier par taux de réussite décroissant puis par volume décroissant
            vehicle_grouped = vehicle_grouped.sort_values(
                ["percent_ok", "total"],
                ascending=[False, False]
            ).reset_index(drop=True)

            vehicle_stats = vehicle_grouped.to_dict("records")

    return templates.TemplateResponse(
        "partials/sessions_stats.html",
        {
            "request": request,
            "no_data": False,
            "total_charges": len(df),
            "total_ok": len(ok_df),
            "total_nok": len(nok_df),
            # Énergie
            "e_total_all": e_total_all,
            "e_mean": e_mean,
            "e_max": e_max,
            # Puissance moyenne
            "pm_mean": pm_mean,
            "pm_max": pm_max,
            # Puissance maximale
            "px_mean": px_mean,
            "px_max": px_max,
            # SOC
            "soc_start_mean": soc_start_mean,
            "soc_end_mean": soc_end_mean,
            "soc_gain_mean": soc_gain_mean,
            # Durées
            "dur_mean": dur_mean,
            # Charges par jour
            "nb_days": nb_days,
            "mean_day": mean_day,
            "med_day": med_day,
            "max_day_site": max_day_site,
            "max_day_date": max_day_date,
            "max_day_nb": max_day_nb,
            # Durées de fonctionnement
            "durations_by_site": durations_by_site,
            "durations_by_site_dict": durations_by_site_dict,
            "site_options_dur": site_options_order if site_options_order else list(durations_by_site_dict.keys()),
            # Statistiques par véhicule
            "vehicle_stats": vehicle_stats,
            "vehicle_debug_info": vehicle_debug_info,
        }
    )


@router.get("/sessions/general")
async def get_sessions_general(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
):
    error_type_list = [e.strip() for e in error_types.split(",") if e.strip()] if error_types else []
    moment_list = [m.strip() for m in moments.split(",") if m.strip()] if moments else []

    where_clause, params = _build_conditions(sites, date_debut, date_fin)

    sql = f"""
        SELECT
            Site,
            `State of charge(0:good, 1:error)` as state,
            type_erreur,
            moment
        FROM kpi_sessions
        WHERE {where_clause}
    """

    df = query_df(sql, params)

    if df.empty:
        return templates.TemplateResponse(
            "partials/sessions_general.html",
            {
                "request": request,
                "total": 0,
                "ok": 0,
                "nok": 0,
                "taux_reussite": 0,
                "taux_echec": 0,
                "recap_columns": [],
                "recap_rows": [],
                "moment_distribution": [],
                "moment_total_errors": 0,
            },
        )

    df = _apply_status_filters(df, error_type_list, moment_list)

    total = len(df)
    ok = int(df["is_ok_filt"].sum())
    nok = total - ok
    taux_reussite = round(ok / total * 100, 1) if total else 0
    taux_echec = round(nok / total * 100, 1) if total else 0

    stats_site = (
        df.groupby("Site")
        .agg(
            total=("is_ok_filt", "count"),
            ok=("is_ok_filt", "sum"),
        )
        .reset_index()
    )
    stats_site["nok"] = stats_site["total"] - stats_site["ok"]
    stats_site["taux_ok"] = np.where(
        stats_site["total"] > 0,
        (stats_site["ok"] / stats_site["total"] * 100).round(1),
        0,
    )

    stat_global = stats_site.rename(columns={"Site": "Site", "total": "Total", "ok": "Total_OK"})
    stat_global["Total_NOK"] = stat_global["Total"] - stat_global["Total_OK"]
    stat_global["% OK"] = np.where(
        stat_global["Total"] > 0,
        (stat_global["Total_OK"] / stat_global["Total"] * 100).round(2),
        0,
    )
    stat_global["% NOK"] = np.where(
        stat_global["Total"] > 0,
        (stat_global["Total_NOK"] / stat_global["Total"] * 100).round(2),
        0,
    )

    err = df[~df["is_ok_filt"]].copy()

    recap_columns = []
    recap_rows = []
    moment_distribution = []
    moment_total_errors = 0

    if not err.empty:
        err_grouped = (
            err.groupby(["Site", "moment"])
            .size()
            .reset_index(name="Nb")
            .pivot(index="Site", columns="moment", values="Nb")
            .fillna(0)
            .astype(int)
            .reset_index()
        )

        moment_cols = [m for m in MOMENT_ORDER if m in err_grouped.columns]
        moment_cols += [c for c in err_grouped.columns if c not in moment_cols + ["Site"]]

        recap = (
            stat_global
            .merge(err_grouped, on="Site", how="left")
            .fillna(0)
            .sort_values("Total_NOK", ascending=False)
            .reset_index(drop=True)
        )

        recap_columns = [
            "Site",
            "Total",
            "Total_OK",
            "Total_NOK",
        ] + moment_cols + ["% OK", "% NOK"]

        recap = recap[recap_columns]

        numeric_moment_cols = [c for c in moment_cols if c in recap.columns]
        if numeric_moment_cols:
            recap[numeric_moment_cols] = recap[numeric_moment_cols].astype(int)

        recap_rows = recap.to_dict("records")

        counts_moment = (
            err.groupby("moment")
            .size()
            .reindex(MOMENT_ORDER, fill_value=0)
            .reset_index(name="count")
        )
        counts_moment = counts_moment[counts_moment["count"] > 0]

        total_err = len(err)
        moment_total_errors = int(total_err)
        moment_distribution = [
            {
                "moment": row["moment"],
                "count": int(row["count"]),
                "percent": round(row["count"] / total_err * 100, 1) if total_err else 0,
            }
            for _, row in counts_moment.iterrows()
        ]

    return templates.TemplateResponse(
        "partials/sessions_general.html",
        {
            "request": request,
            "total": total,
            "ok": ok,
            "nok": nok,
            "taux_reussite": taux_reussite,
            "taux_echec": taux_echec,
            "recap_columns": recap_columns,
            "recap_rows": recap_rows,
            "moment_distribution": moment_distribution,
            "moment_total_errors": moment_total_errors,
        },
    )


@router.get("/sessions/comparaison")
async def get_sessions_comparaison(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
    site_focus: str = Query(default=""),
    month_focus: str = Query(default=""),
):
    error_type_list = [e.strip() for e in error_types.split(",") if e.strip()] if error_types else []
    moment_list = [m.strip() for m in moments.split(",") if m.strip()] if moments else []

    filters = {
        "sites": sites,
        "date_debut": str(date_debut) if date_debut else "",
        "date_fin": str(date_fin) if date_fin else "",
        "error_types": error_types,
        "moments": moments,
    }

    where_clause, params = _build_conditions(sites, date_debut, date_fin)

    sql = f"""
        SELECT
            Site,
            `Datetime start`,
            `State of charge(0:good, 1:error)` as state,
            type_erreur,
            moment
        FROM kpi_sessions
        WHERE {where_clause}
    """

    try:
        df = query_df(sql, params)
    except Exception as exc:  # pragma: no cover - defensive fallback for UI visibility
        return templates.TemplateResponse(
            "partials/sessions_comparaison.html",
            _comparaison_base_context(
                request,
                filters,
                site_focus="",
                month_focus="",
                error_message=str(exc),
            ),
        )

    if df.empty:
        return templates.TemplateResponse(
            "partials/sessions_comparaison.html",
            _comparaison_base_context(request, filters),
        )

    df = _apply_status_filters(df, error_type_list, moment_list)

    if "Datetime start" in df.columns:
        df["Datetime start"] = pd.to_datetime(df["Datetime start"], errors="coerce")

    site_col = "Site"

    by_site = (
        df.groupby(site_col, as_index=False)
        .agg(
            Total_Charges=("is_ok_filt", "count"),
            Charges_OK=("is_ok_filt", "sum"),
        )
    )
    by_site["Charges_NOK"] = by_site["Total_Charges"] - by_site["Charges_OK"]
    by_site["% Réussite"] = np.where(
        by_site["Total_Charges"].gt(0),
        (by_site["Charges_OK"] / by_site["Total_Charges"] * 100).round(2),
        0.0,
    )
    by_site["% Échec"] = np.where(
        by_site["Total_Charges"].gt(0),
        (by_site["Charges_NOK"] / by_site["Total_Charges"] * 100).round(2),
        0.0,
    )
    by_site = by_site.reset_index(drop=True)

    site_rows = by_site.to_dict("records")
    by_site_sorted = by_site.sort_values("Total_Charges", ascending=False)
    max_total = int(by_site_sorted["Total_Charges"].max()) if not by_site_sorted.empty else 0

    count_bars = [
        {
            "site": row[site_col],
            "ok": int(row["Charges_OK"]),
            "nok": int(row["Charges_NOK"]),
            "total": int(row["Total_Charges"]),
        }
        for _, row in by_site_sorted.iterrows()
    ]

    percent_bars = [
        {
            "site": row[site_col],
            "ok_pct": float(row["% Réussite"]),
            "nok_pct": float(row["% Échec"]),
        }
        for _, row in by_site_sorted.iterrows()
    ]

    base = df.copy()
    base["hour"] = pd.to_datetime(base["Datetime start"], errors="coerce").dt.hour

    g = (
        base.dropna(subset=["hour"])
        .groupby([site_col, "hour"])
        .size()
        .reset_index(name="Nb")
    )

    peak_rows = []
    heatmap_rows = []
    heatmap_hours: list[int] = []
    heatmap_max = 0

    if not g.empty:
        peak = g.loc[g.groupby(site_col)["Nb"].idxmax()][[site_col, "hour", "Nb"]].rename(
            columns={"hour": "Heure de pic", "Nb": "Nb au pic"}
        )

        def _w_median_hours(dfh: pd.DataFrame) -> int:
            s = dfh.sort_values("hour")
            c = s["Nb"].cumsum()
            half = s["Nb"].sum() / 2.0
            return int(s.loc[c >= half, "hour"].iloc[0])

        med = (
            g.groupby(site_col)[["hour", "Nb"]]
            .apply(_w_median_hours)
            .reset_index(name="Heure médiane")
        )
        summ = peak.merge(med, on=site_col, how="left")

        for _, row in summ.sort_values(site_col).iterrows():
            peak_rows.append(
                {
                    "site": row[site_col],
                    "peak_hour": f"{int(row['Heure de pic']):02d}:00",
                    "peak_nb": int(row["Nb au pic"]),
                    "median_hour": f"{int(row['Heure médiane']):02d}:00",
                }
            )

        heatmap = g.pivot(index=site_col, columns="hour", values="Nb").fillna(0)
        heatmap_hours = sorted(heatmap.columns.tolist())
        heatmap_max = int(heatmap.values.max()) if heatmap.size else 0
        for idx in heatmap.index:
            heatmap_rows.append(
                {
                    "site": idx,
                    "values": [int(heatmap.at[idx, h]) if h in heatmap.columns else 0 for h in heatmap_hours],
                }
            )

    site_options = by_site_sorted[site_col].tolist()
    site_focus_value = site_focus if site_focus and site_focus in site_options else (site_options[0] if site_options else "")

    monthly_rows = []
    daily_rows = []
    month_options: list[str] = []
    month_focus_value = ""

    if site_focus_value:
        base_site = base[base[site_col] == site_focus_value].copy()
        ok_focus = base_site[base_site["is_ok_filt"]].copy()
        nok_focus = base_site[~base_site["is_ok_filt"]].copy()

        ok_focus["month"] = pd.to_datetime(ok_focus["Datetime start"], errors="coerce").dt.to_period("M").astype(str)
        nok_focus["month"] = pd.to_datetime(nok_focus["Datetime start"], errors="coerce").dt.to_period("M").astype(str)

        g_ok_m = ok_focus.groupby("month").size().reset_index(name="Nb").assign(Status="OK")
        g_nok_m = nok_focus.groupby("month").size().reset_index(name="Nb").assign(Status="NOK")

        g_both_m = pd.concat([g_ok_m, g_nok_m], ignore_index=True)
        g_both_m["month"] = pd.to_datetime(g_both_m["month"], errors="coerce")
        g_both_m = g_both_m.dropna(subset=["month"]).sort_values("month")
        g_both_m["month"] = g_both_m["month"].dt.strftime("%Y-%m")

        if not g_both_m.empty:
            piv_m = g_both_m.pivot(index="month", columns="Status", values="Nb").fillna(0).sort_index()
            month_options = piv_m.index.tolist()
            month_focus_value = month_focus if month_focus in month_options else (month_options[-1] if month_options else "")
            for month in month_options:
                ok_val = int(piv_m.at[month, "OK"]) if "OK" in piv_m.columns else 0
                nok_val = int(piv_m.at[month, "NOK"]) if "NOK" in piv_m.columns else 0
                total_val = ok_val + nok_val
                ok_pct = round(ok_val / total_val * 100, 1) if total_val else 0
                nok_pct = round(nok_val / total_val * 100, 1) if total_val else 0
                monthly_rows.append(
                    {
                        "month": month,
                        "ok": ok_val,
                        "nok": nok_val,
                        "ok_pct": ok_pct,
                        "nok_pct": nok_pct,
                    }
                )

            if month_focus_value:
                ok_month = ok_focus[ok_focus["month"] == month_focus_value].copy()
                nok_month = nok_focus[nok_focus["month"] == month_focus_value].copy()

                ok_month["day"] = pd.to_datetime(ok_month["Datetime start"], errors="coerce").dt.strftime("%Y-%m-%d")
                nok_month["day"] = pd.to_datetime(nok_month["Datetime start"], errors="coerce").dt.strftime("%Y-%m-%d")

                per = pd.Period(month_focus_value, freq="M")
                days = pd.date_range(per.to_timestamp(how="start"), per.to_timestamp(how="end"), freq="D").strftime("%Y-%m-%d")

                g_ok_d = ok_month.groupby("day").size().reindex(days, fill_value=0).reset_index()
                g_ok_d.columns = ["day", "Nb"]
                g_ok_d["Status"] = "OK"
                g_nok_d = nok_month.groupby("day").size().reindex(days, fill_value=0).reset_index()
                g_nok_d.columns = ["day", "Nb"]
                g_nok_d["Status"] = "NOK"

                g_both_d = pd.concat([g_ok_d, g_nok_d], ignore_index=True)
                piv_d = g_both_d.pivot(index="day", columns="Status", values="Nb").fillna(0)
                for day in piv_d.index.tolist():
                    ok_val = int(piv_d.at[day, "OK"]) if "OK" in piv_d.columns else 0
                    nok_val = int(piv_d.at[day, "NOK"]) if "NOK" in piv_d.columns else 0
                    total_val = ok_val + nok_val
                    ok_pct = round(ok_val / total_val * 100, 1) if total_val else 0
                    nok_pct = round(nok_val / total_val * 100, 1) if total_val else 0
                    daily_rows.append(
                        {
                            "day": day,
                            "ok": ok_val,
                            "nok": nok_val,
                            "ok_pct": ok_pct,
                            "nok_pct": nok_pct,
                        }
                    )

    context = _comparaison_base_context(
        request,
        filters,
        site_focus=site_focus_value,
        month_focus=month_focus_value,
    )
    context.update(
        {
            "site_rows": site_rows,
            "count_bars": count_bars,
            "percent_bars": percent_bars,
            "max_total": max_total,
            "peak_rows": peak_rows,
            "heatmap_rows": heatmap_rows,
            "heatmap_hours": heatmap_hours,
            "heatmap_max": heatmap_max,
            "site_options": site_options,
            "month_options": month_options,
            "monthly_rows": monthly_rows,
            "daily_rows": daily_rows,
        }
    )

    return templates.TemplateResponse("partials/sessions_comparaison.html", context)


def _format_soc(s0, s1):
    if pd.notna(s0) and pd.notna(s1):
        try:
            return f"{int(round(s0))}% → {int(round(s1))}%"
        except Exception:
            return ""
    return ""


def _prepare_query_params(request: Request) -> str:
    allowed = {"sites", "date_debut", "date_fin", "error_types", "moments"}
    data = {k: v for k, v in request.query_params.items() if k in allowed and v}
    return urlencode(data)


@router.get("/sessions/site-details")
async def get_sessions_site_details(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
    site_focus: str = Query(default=""),
    pdc: str = Query(default=""),
):
    error_type_list = [e.strip() for e in error_types.split(",") if e.strip()] if error_types else []
    moment_list = [m.strip() for m in moments.split(",") if m.strip()] if moments else []

    where_clause, params = _build_conditions(sites, date_debut, date_fin)

    sql = f"""
        SELECT
            Site,
            PDC,
            ID,
            `Datetime start`,
            `Datetime end`,
            `Energy (Kwh)`,
            `MAC Address`,
            type_erreur,
            moment,
            `SOC Start`,
            `SOC End`,
            `Downstream Code PC`,
            `EVI Error Code`,
            `State of charge(0:good, 1:error)` as state
        FROM kpi_sessions
        WHERE {where_clause}
    """

    df = query_df(sql, params)

    if df.empty:
        return templates.TemplateResponse(
            "partials/sessions_site_details.html",
            {"request": request, "site_options": [], "base_query": _prepare_query_params(request)},
        )

    df["PDC"] = df["PDC"].astype(str)
    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)

    mask_type = df["type_erreur"].isin(error_type_list) if error_type_list and "type_erreur" in df.columns else True
    mask_moment = df["moment"].isin(moment_list) if moment_list and "moment" in df.columns else True
    mask_nok = ~df["is_ok"]
    mask_filtered_error = mask_nok & mask_type & mask_moment
    df["is_ok_filt"] = np.where(mask_filtered_error, False, True)

    site_options = sorted(df["Site"].dropna().unique().tolist())
    site_value = site_focus if site_focus in site_options else (site_options[0] if site_options else "")

    df_site = df[df["Site"] == site_value].copy()
    if df_site.empty:
        return templates.TemplateResponse(
            "partials/sessions_site_details.html",
            {
                "request": request,
                "site_options": site_options,
                "site_focus": site_value,
                "pdc_options": [],
                "selected_pdc": [],
                "base_query": _prepare_query_params(request),
            },
        )

    pdc_options = sorted(df_site["PDC"].dropna().unique().tolist())
    selected_pdc = [p.strip() for p in pdc.split(",") if p.strip()] if pdc else pdc_options
    selected_pdc = [p for p in selected_pdc if p in pdc_options] or pdc_options

    df_site = df_site[df_site["PDC"].isin(selected_pdc)].copy()

    mask_type_site = (
        df_site["type_erreur"].isin(error_type_list)
        if error_type_list and "type_erreur" in df_site.columns
        else pd.Series(True, index=df_site.index)
    )
    mask_moment_site = (
        df_site["moment"].isin(moment_list)
        if moment_list and "moment" in df_site.columns
        else pd.Series(True, index=df_site.index)
    )
    df_filtered = df_site[mask_type_site & mask_moment_site].copy()

    for col in ["Datetime start", "Datetime end"]:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_datetime(df_filtered[col], errors="coerce")
    for col in ["Energy (Kwh)", "SOC Start", "SOC End"]:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

    err_rows = df_filtered[~df_filtered["is_ok"]].copy()
    err_rows["evolution_soc"] = err_rows.apply(lambda r: _format_soc(r.get("SOC Start"), r.get("SOC End")), axis=1)
    err_rows["elto"] = err_rows["ID"].apply(lambda x: f"https://elto.nidec-asi-online.com/Charge/detail?id={str(x).strip()}" if pd.notna(x) else "") if "ID" in err_rows.columns else ""
    err_display_cols = [
        "ID",
        "Datetime start",
        "Datetime end",
        "PDC",
        "Energy (Kwh)",
        "MAC Address",
        "type_erreur",
        "moment",
        "evolution_soc",
        "elto",
    ]
    err_table = err_rows[err_display_cols].copy() if not err_rows.empty else pd.DataFrame(columns=err_display_cols)
    if "Datetime start" in err_table.columns:
        err_table = err_table.sort_values("Datetime start", ascending=False)

    ok_rows = df_filtered[df_filtered["is_ok"]].copy()
    ok_rows["evolution_soc"] = ok_rows.apply(lambda r: _format_soc(r.get("SOC Start"), r.get("SOC End")), axis=1)
    ok_rows["elto"] = ok_rows["ID"].apply(lambda x: f"https://elto.nidec-asi-online.com/Charge/detail?id={str(x).strip()}" if pd.notna(x) else "") if "ID" in ok_rows.columns else ""
    ok_display_cols = [
        "ID",
        "Datetime start",
        "Datetime end",
        "PDC",
        "Energy (Kwh)",
        "MAC Address",
        "evolution_soc",
        "elto",
    ]
    ok_table = ok_rows[ok_display_cols].copy() if not ok_rows.empty else pd.DataFrame(columns=ok_display_cols)
    if "Datetime start" in ok_table.columns:
        ok_table = ok_table.sort_values("Datetime start", ascending=False)

    by_pdc = (
        df_site.groupby("PDC", as_index=False)
        .agg(Total_Charges=("is_ok_filt", "count"), Charges_OK=("is_ok_filt", "sum"))
        .assign(Charges_NOK=lambda d: d["Total_Charges"] - d["Charges_OK"])
    )
    by_pdc["% Réussite"] = np.where(
        by_pdc["Total_Charges"].gt(0),
        (by_pdc["Charges_OK"] / by_pdc["Total_Charges"] * 100).round(2),
        0.0,
    )
    by_pdc = by_pdc.sort_values(["% Réussite", "PDC"], ascending=[True, True])

    err_evi = err_rows[err_rows["type_erreur"] == "Erreur_EVI"].copy() if not err_rows.empty else pd.DataFrame()
    evi_moment: list[dict] = []
    evi_moment_grouped: list[dict] = []
    if not err_evi.empty and "moment" in err_evi.columns:
        counts = err_evi.groupby("moment").size().reset_index(name="Nb")
        total = counts["Nb"].sum()
        if total:
            evi_moment = (
                counts.assign(percent=lambda d: (d["Nb"] / total * 100).round(2))
                .sort_values("percent", ascending=False)
                .to_dict("records")
            )

        mapping = {
            "Init": "Avant charge",
            "Lock Connector": "Avant charge",
            "CableCheck": "Avant charge",
            "Charge": "Charge",
            "Fin de charge": "Fin de charge",
            "Unknown": "Unknown",
        }

        counts_grouped = (
            counts.assign(Moment_grp=counts["moment"].map(mapping))
            .groupby("Moment_grp", as_index=False)["Nb"].sum()
            .sort_values("Nb", ascending=False)
        )

        total_grouped = counts_grouped["Nb"].sum()
        if total_grouped:
            evi_moment_grouped = (
                counts_grouped.assign(percent=lambda d: (d["Nb"] / total_grouped * 100).round(2))
                .to_dict("records")
            )

    downstream_occ: list[dict] = []
    downstream_moments: list[str] = []
    if not err_rows.empty:
        need_cols_ds = {"Downstream Code PC", "moment"}
        if need_cols_ds.issubset(err_rows.columns):
            ds_num = pd.to_numeric(err_rows["Downstream Code PC"], errors="coerce").fillna(0).astype(int)
            mask_downstream = (ds_num != 0) & (ds_num != 8192)
            sub = err_rows.loc[mask_downstream, ["Downstream Code PC", "moment"]].copy()

            if not sub.empty:
                sub["Code_PC"] = pd.to_numeric(sub["Downstream Code PC"], errors="coerce").fillna(0).astype(int)
                tmp = sub.groupby(["Code_PC", "moment"]).size().reset_index(name="Occurrences")
                downstream_moments = [m for m in MOMENT_ORDER if m in tmp["moment"].unique()]
                downstream_moments += [m for m in sorted(tmp["moment"].unique()) if m not in downstream_moments]

                table = (
                    tmp.pivot(index="Code_PC", columns="moment", values="Occurrences")
                    .reindex(columns=downstream_moments, fill_value=0)
                    .reset_index()
                )

                # Fill potential missing occurrences before casting to int to avoid
                # pandas IntCastingNaNError when moments are absent for a Code_PC.
                table[downstream_moments] = table[downstream_moments].fillna(0).astype(int)
                table["Total"] = table[downstream_moments].sum(axis=1).astype(int)
                table = table.sort_values("Total", ascending=False).reset_index(drop=True)

                total_all = int(table["Total"].sum())
                table["Percent"] = np.where(
                    total_all > 0,
                    (table["Total"] / total_all * 100).round(2),
                    0.0,
                )

                table.insert(0, "Rank", range(1, len(table) + 1))

                total_row = {
                    "Rank": "",
                    "Code_PC": "Total",
                    **{m: int(table[m].sum()) for m in downstream_moments},
                }
                total_row["Total"] = int(table["Total"].sum())
                total_row["Percent"] = 100.0 if total_all else 0.0

                downstream_occ = table.to_dict("records") + [total_row]

    evi_occ: list[dict] = []
    evi_occ_moments: list[str] = []
    if not err_rows.empty:
        need_cols_evi = {"EVI Error Code", "moment"}
        if need_cols_evi.issubset(err_rows.columns):
            ds_num = pd.to_numeric(err_rows.get("Downstream Code PC", 0), errors="coerce").fillna(0).astype(int)
            evi_code = pd.to_numeric(err_rows["EVI Error Code"], errors="coerce").fillna(0).astype(int)

            mask_evi = (ds_num == 8192) | ((ds_num == 0) & (evi_code != 0))
            sub = err_rows.loc[mask_evi, ["EVI Error Code", "moment"]].copy()

            if not sub.empty:
                sub["EVI_Code"] = pd.to_numeric(sub["EVI Error Code"], errors="coerce").astype(int)
                tmp = sub.groupby(["EVI_Code", "moment"]).size().reset_index(name="Occurrences")
                evi_occ_moments = [m for m in MOMENT_ORDER if m in tmp["moment"].unique()]
                evi_occ_moments += [m for m in sorted(tmp["moment"].unique()) if m not in evi_occ_moments]

                table = (
                    tmp.pivot(index="EVI_Code", columns="moment", values="Occurrences")
                    .reindex(columns=evi_occ_moments, fill_value=0)
                    .reset_index()
                )

                # Fill potential missing occurrences before casting to int to avoid
                # pandas IntCastingNaNError when moments are absent for an EVI code.
                table[evi_occ_moments] = table[evi_occ_moments].fillna(0).astype(int)
                table["Total"] = table[evi_occ_moments].sum(axis=1).astype(int)
                table = table.sort_values("Total", ascending=False).reset_index(drop=True)

                total_all = int(table["Total"].sum())
                table["Percent"] = np.where(
                    total_all > 0,
                    (table["Total"] / total_all * 100).round(2),
                    0.0,
                )

                table.insert(0, "Rank", range(1, len(table) + 1))

                total_row = {
                    "Rank": "",
                    "EVI_Code": "Total",
                    **{m: int(table[m].sum()) for m in evi_occ_moments},
                }
                total_row["Total"] = int(table["Total"].sum())
                total_row["Percent"] = 100.0 if total_all else 0.0

                evi_occ = table.to_dict("records") + [total_row]

    return templates.TemplateResponse(
        "partials/sessions_site_details.html",
        {
            "request": request,
            "site_options": site_options,
            "site_focus": site_value,
            "pdc_options": pdc_options,
            "selected_pdc": selected_pdc,
            "err_rows": err_table.to_dict("records"),
            "ok_rows": ok_table.to_dict("records"),
            "by_pdc": by_pdc.to_dict("records"),
            "evi_moment": evi_moment,
            "evi_moment_grouped": evi_moment_grouped,
            "downstream_occ": downstream_occ,
            "downstream_moments": downstream_moments,
            "evi_occ": evi_occ,
            "evi_occ_moments": evi_occ_moments,
            "base_query": _prepare_query_params(request),
        },
    )
