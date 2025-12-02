"""
Router pour les KPIs génériques
Endpoints pour transactions suspectes, tentatives multiples, etc.
"""

from fastapi import APIRouter, Request, Query
from fastapi.templating import Jinja2Templates
from datetime import date
import pandas as pd

from db import query_df

router = APIRouter(tags=["kpis"])
templates = Jinja2Templates(directory="templates")


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
    
    if not df.empty and "Datetime start" in df.columns:
        df["Datetime start"] = pd.to_datetime(df["Datetime start"], errors="coerce")
        
        if date_debut:
            df = df[df["Datetime start"] >= pd.Timestamp(date_debut)]
        if date_fin:
            df = df[df["Datetime start"] < pd.Timestamp(date_fin) + pd.Timedelta(days=1)]
        
        if sites and "Site" in df.columns:
            site_list = [s.strip() for s in sites.split(",") if s.strip()]
            if site_list:
                df = df[df["Site"].isin(site_list)]
    
    nb = len(df)
    status = "danger" if nb > 5 else ("warning" if nb > 0 else "success")
    
    return templates.TemplateResponse(
        "partials/kpi_card.html",
        {
            "request": request,
            "value": nb,
            "label": "Transactions <1 kWh",
            "status": status,
        }
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
    
    nb = len(df)
    status = "danger" if nb > 5 else ("warning" if nb > 0 else "success")
    
    return templates.TemplateResponse(
        "partials/kpi_card.html",
        {
            "request": request,
            "value": nb,
            "label": "Utilisateurs multi-tentatives",
            "status": status,
        }
    )
