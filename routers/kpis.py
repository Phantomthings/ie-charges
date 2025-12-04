from fastapi import APIRouter, Request, Query
from fastapi.templating import Jinja2Templates
from datetime import date
import pandas as pd
import numpy as np
import re

from db import query_df

router = APIRouter(tags=["kpis"])
templates = Jinja2Templates(directory="templates")
BASE_CHARGE_URL = "https://elto.nidec-asi-online.com/Charge/detail?id="


@router.get("/kpi/suspicious")
async def get_suspicious(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
):
    """Transactions suspectes (<1 kWh)"""
    sql = "SELECT * FROM kpi_suspicious_under_1kwh"
    df = query_df(sql)

    if not df.empty:
        if "Datetime start" in df.columns:
            df["Datetime start"] = pd.to_datetime(df["Datetime start"], errors="coerce")

            if date_debut:
                df = df[df["Datetime start"] >= pd.Timestamp(date_debut)]
            if date_fin:
                df = df[df["Datetime start"] < pd.Timestamp(date_fin) + pd.Timedelta(days=1)]

        if sites and "Site" in df.columns:
            site_list = [s.strip() for s in sites.split(",") if s.strip()]
            if site_list:
                df = df[df["Site"].isin(site_list)]

        if "Datetime start" in df.columns:
            df = df.sort_values("Datetime start")

    def format_ts(value):
        if pd.isna(value):
            return ""
        try:
            ts = pd.to_datetime(value, errors="coerce")
            return ts.strftime("%Y-%m-%d %H:%M") if not pd.isna(ts) else ""
        except Exception:
            return str(value)

    def to_str(value):
        if pd.isna(value):
            return ""
        return str(value)

    def to_float(value):
        if pd.isna(value):
            return ""
        try:
            return round(float(value), 3)
        except Exception:
            return value

    rows = []
    if not df.empty:
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            charge_id = to_str(row.get("ID", ""))
            rows.append(
                {
                    "rank": idx,
                    "id": charge_id,
                    "url": f"{BASE_CHARGE_URL}{charge_id}" if charge_id else "",
                    "site": row.get("Site", ""),
                    "pdc": to_str(row.get("PDC", "")),
                    "mac": row.get("MAC Address", ""),
                    "vehicle": row.get("Vehicle", ""),
                    "start": format_ts(row.get("Datetime start")),
                    "end": format_ts(row.get("Datetime end")),
                    "energy": to_float(row.get("Energy (Kwh)")),
                    "soc_start": to_float(row.get("SOC Start")),
                    "soc_end": to_float(row.get("SOC End")),
                }
            )

    return templates.TemplateResponse(
        "partials/suspicious.html",
        {
            "request": request,
            "rows": rows,
        },
    )


@router.get("/kpi/multi-attempts")
async def get_multi_attempts(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
):
    """Tentatives multiples par heure"""
    sql = "SELECT * FROM kpi_multi_attempts_hour"
    df = query_df(sql)

    if not df.empty and "Date_heure" in df.columns:
        df["Date_heure"] = pd.to_datetime(df["Date_heure"], errors="coerce")

        if date_debut:
            df = df[df["Date_heure"] >= pd.Timestamp(date_debut)]
        if date_fin:
            df = df[df["Date_heure"] < pd.Timestamp(date_fin) + pd.Timedelta(days=1)]

        if sites and "Site" in df.columns:
            site_list = [s.strip() for s in sites.split(",") if s.strip()]
            if site_list:
                df = df[df["Site"].isin(site_list)]

    if not df.empty:
        sort_cols = []
        sort_order = []
        if "Date_heure" in df.columns:
            sort_cols.append("Date_heure")
            sort_order.append(True)
        if "Site" in df.columns:
            sort_cols.append("Site")
            sort_order.append(True)
        if "tentatives" in df.columns:
            sort_cols.append("tentatives")
            sort_order.append(False)
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=sort_order)

    soc_columns = [
        col
        for col in ["SOC start min", "SOC start max", "SOC end min", "SOC end max"]
        if col in df.columns
    ]

    def format_ts(value):
        if pd.isna(value):
            return ""
        try:
            ts = pd.to_datetime(value, errors="coerce")
            return ts.strftime("%Y-%m-%d %H:%M") if not pd.isna(ts) else ""
        except Exception:
            return str(value)

    def parse_ids(value):
        if not isinstance(value, str):
            value = "" if pd.isna(value) else str(value)
        ids = [v.strip() for v in value.split(",") if v.strip()]
        return [{"id": iid, "url": f"{BASE_CHARGE_URL}{iid}"} for iid in ids]

    table_rows = []
    if not df.empty:
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            date_value = row.get("Date_heure")
            hour_value = row.get("Heure")
            if (pd.isna(hour_value) or hour_value == "") and not pd.isna(date_value):
                ts_hour = pd.to_datetime(date_value, errors="coerce")
                hour_value = ts_hour.strftime("%Y-%m-%d %H:%M") if not pd.isna(ts_hour) else ""

            tentatives_val = pd.to_numeric(row.get("tentatives", 0), errors="coerce")
            tentatives = int(tentatives_val) if pd.notna(tentatives_val) else 0

            table_rows.append(
                {
                    "rank": idx,
                    "site": row.get("Site", ""),
                    "hour": hour_value or "",
                    "mac": row.get("MAC", ""),
                    "vehicle": row.get("Vehicle", ""),
                    "tentatives": tentatives,
                    "pdc": row.get("PDC(s)", ""),
                    "first_attempt": format_ts(row.get("1ère tentative")),
                    "last_attempt": format_ts(row.get("Dernière tentative")),
                    "ids": parse_ids(row.get("ID(s)")),
                    "soc_values": {col: row.get(col, "") for col in soc_columns},
                }
            )

    return templates.TemplateResponse(
        "partials/multi_attempts.html",
        {
            "request": request,
            "rows": table_rows,
            "soc_columns": soc_columns,
        },
    )


@router.get("/kpi/mac-id")
async def get_mac_identifier(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    mac_prefix: str = Query(default=""),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
):
    """Top 10 des MAC non identifiées et détails des charges associées."""

    def normalize_mac(value: str) -> str:
        if value is None:
            return ""
        return re.sub(r"[^0-9a-f]", "", str(value).lower().replace("0x", "", 1))

    def format_mac(value: str) -> str:
        norm = normalize_mac(value)
        return ":".join(norm[i : i + 2].upper() for i in range(0, len(norm), 2)) if norm else ""

    def format_dt(value):
        if pd.isna(value):
            return ""
        try:
            ts = pd.to_datetime(value, errors="coerce")
            return ts.strftime("%Y-%m-%d %H:%M") if not pd.isna(ts) else ""
        except Exception:
            return str(value)

    def format_float(value, ndigits=3):
        if pd.isna(value):
            return ""
        try:
            return round(float(value), ndigits)
        except Exception:
            return value

    filters = {
        "sites": sites or "",
        "date_debut": date_debut.isoformat() if date_debut else "",
        "date_fin": date_fin.isoformat() if date_fin else "",
        "error_types": error_types or "",
        "moments": moments or "",
    }

    # Top 10 MAC non identifiées
    mac_counts = query_df("SELECT * FROM kpi_mac_id")
    top_rows = []

    if not mac_counts.empty:
        mac_col = "Mac" if "Mac" in mac_counts.columns else None
        count_col = None
        for col in ("nombre_de_charges", "Nombre_de_charges", "nb_charges", "Nombre de charges"):
            if col in mac_counts.columns:
                count_col = col
                break

        if mac_col is None:
            mac_counts["Mac"] = ""
            mac_col = "Mac"

        if count_col is None:
            mac_counts["Nombre de charges"] = np.nan
            count_col = "Nombre de charges"

        mac_counts = mac_counts.rename(columns={count_col: "Nombre de charges"})
        mac_counts["Nombre de charges"] = pd.to_numeric(
            mac_counts["Nombre de charges"], errors="coerce"
        ).fillna(0)
        mac_counts = mac_counts.sort_values("Nombre de charges", ascending=False).head(10)

        for idx, (_, row) in enumerate(mac_counts.iterrows(), start=1):
            top_rows.append(
                {
                    "rank": idx,
                    "mac": format_mac(row.get(mac_col, "")),
                    "mac_raw": normalize_mac(row.get(mac_col, "")),
                    "charges": int(row.get("Nombre de charges", 0)),
                }
            )

    # Détails des charges pour un préfixe MAC
    charges_df = query_df("SELECT * FROM kpi_charges_mac")
    sessions_df = query_df("SELECT ID, `Datetime end` FROM kpi_sessions")

    mac_query = normalize_mac(mac_prefix)
    charges_rows = []
    summary = None
    selected_mac_display = format_mac(mac_query) if mac_query else ""

    if not charges_df.empty and mac_query:
        df = charges_df.copy()

        mac_col = None
        for col in ("mac", "MAC", "MAC Address", "Mac"):
            if col in df.columns:
                mac_col = col
                break

        if mac_col:
            df["_mac_norm"] = df[mac_col].astype(str).map(normalize_mac)
            df = df[df["_mac_norm"].str.startswith(mac_query)]
        else:
            df = df.iloc[0:0]

        if "Site" in df.columns and sites:
            site_list = [s.strip() for s in sites.split(",") if s.strip()]
            if site_list:
                df = df[df["Site"].isin(site_list)]

        if "Datetime start" in df.columns:
            df["Datetime start"] = pd.to_datetime(df["Datetime start"], errors="coerce")
            if date_debut:
                df = df[df["Datetime start"] >= pd.Timestamp(date_debut)]
            if date_fin:
                df = df[df["Datetime start"] < pd.Timestamp(date_fin) + pd.Timedelta(days=1)]

        if not sessions_df.empty and {"ID", "Datetime end"}.issubset(df.columns):
            sess_lookup = sessions_df[["ID", "Datetime end"]].copy()
            sess_lookup["ID"] = sess_lookup["ID"].astype(str).str.strip()
            df["ID"] = df["ID"].astype(str).str.strip()
            df = df.merge(sess_lookup, on="ID", how="left", suffixes=("", "_sess"))
            if "Datetime end_sess" in df.columns:
                df["Datetime end"] = df["Datetime end"].fillna(df.pop("Datetime end_sess"))

        if "is_ok" in df.columns:
            ok_series = pd.to_numeric(df["is_ok"], errors="coerce").fillna(0).astype(int).astype(bool)
            total = len(ok_series)
            ok_count = int(ok_series.sum())
            summary = {
                "total": total,
                "ok": ok_count,
                "rate": round((ok_count / total * 100), 1) if total else 0.0,
            }
            df["_is_ok"] = ok_series

        display_cols = [
            "Site",
            "PDC",
            "Datetime start",
            "Datetime end",
            "MAC Address",
            "Vehicle",
            "Energy (Kwh)",
            "ID",
        ]
        df = df[[c for c in display_cols if c in df.columns]].copy()

        if "Datetime start" in df.columns:
            df = df.sort_values("Datetime start", ascending=False)

        for col in ("Datetime start", "Datetime end"):
            if col in df.columns:
                df[col] = df[col].apply(format_dt)

        if "Energy (Kwh)" in df.columns:
            df["Energy (Kwh)"] = df["Energy (Kwh)"].apply(lambda v: format_float(v, 3))

        for _, row in df.iterrows():
            charge_id = row.get("ID", "")
            charges_rows.append(
                {
                    "site": row.get("Site", ""),
                    "pdc": row.get("PDC", ""),
                    "start": row.get("Datetime start", ""),
                    "end": row.get("Datetime end", ""),
                    "mac": format_mac(row.get("MAC Address", mac_query)),
                    "vehicle": row.get("Vehicle", ""),
                    "energy": row.get("Energy (Kwh)", ""),
                    "url": f"{BASE_CHARGE_URL}{charge_id}" if charge_id else "",
                    "id": charge_id,
                    "is_ok": bool(row.get("_is_ok", False)) if "_is_ok" in df.columns else None,
                }
            )

    return templates.TemplateResponse(
        "partials/mac_id.html",
        {
            "request": request,
            "filters": filters,
            "top_rows": top_rows,
            "mac_prefix": mac_prefix,
            "mac_query": mac_query,
            "selected_mac": selected_mac_display,
            "charges_rows": charges_rows,
            "summary": summary,
        },
    )
