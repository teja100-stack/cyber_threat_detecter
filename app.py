"""
=============================================================================
AI Cyber Threat Detection and Intelligent Response
=============================================================================
Educational Prototype for Small Organizations & Academic Labs
Framework: Streamlit
Backend: Scikit-learn (Isolation Forest), Pandas, NumPy, Joblib
=============================================================================
"""

import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Import training utility directly for in-app model training capability
try:
    from train_model import train_isolation_forest, MODEL_FILENAME, FEATURE_COLUMNS
except ImportError:
    MODEL_FILENAME = "cyber_threat_model.joblib"
    FEATURE_COLUMNS = ["login_attempts", "is_known_device", "data_transferred_mb", "is_off_hours"]
    train_isolation_forest = None

# =============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & CUSTOM STYLES
# =============================================================================
st.set_page_config(
    page_title="AI Cyber Threat Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS: SECURITY OPERATIONS CENTER (SOC) DASHBOARD (CSS ONLY)
# =============================================================================
st.markdown("""
<style>
    /* =========================================================================
       AI CYBER THREAT DETECTION — SOC DASHBOARD CSS
       Streamlit-compatible. Zero external dependencies.
       ========================================================================= */

    /* -------------------------------------------------------------------------
       1. CORE TYPOGRAPHY & THEME RESETS
       ------------------------------------------------------------------------- */
    html, body, [class*="css"], .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #f1f5f9;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    code, pre, .mono-text, [data-testid="stCodeBlock"] {
        font-family: ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em !important;
    }

    p, span, label, li { color: #e2e8f0; }

    [data-testid="stCaptionContainer"] p, .stCaption { color: #94a3b8 !important; }

    hr { border-color: rgba(51, 65, 85, 0.45) !important; }

    /* -------------------------------------------------------------------------
       2. APP BACKGROUND & ATMOSPHERE
       ------------------------------------------------------------------------- */
    @keyframes gridDrift {
        0%   { background-position: 0 0, 0 0, 0 0, 0 0, 0 0; }
        100% { background-position: 0 0, 0 0, 0 0, 0 0, 48px 48px; }
    }

    [data-testid="stAppViewContainer"], .stApp {
        background-color: #040812 !important;
        background-image:
            radial-gradient(ellipse 90% 55% at 50% -10%, rgba(6,182,212,0.12), transparent 60%),
            radial-gradient(ellipse 70% 45% at 100% 35%, rgba(20,184,166,0.08), transparent 55%),
            radial-gradient(ellipse 60% 40% at 0% 75%, rgba(14,165,233,0.08), transparent 60%),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 160 160'%3E%3Cpath d='M0 30 H60 L75 45 H160 M40 0 V50 L55 65 V160 M90 160 V110 L105 95 H160 M0 120 H40 L55 105 H100 L115 120 H160' fill='none' stroke='%2306b6d4' stroke-width='0.8' stroke-opacity='0.10'/%3E%3C/svg%3E"),
            linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 100% 100%, 160px 160px, 48px 48px, 48px 48px !important;
        background-attachment: fixed !important;
        animation: gridDrift 60s linear infinite !important;
        color: #f1f5f9 !important;
    }

    @media (prefers-reduced-motion: reduce) {
        *, [data-testid="stAppViewContainer"], .stApp {
            animation: none !important;
            transition: none !important;
        }
    }

    /* Main Content Container */
    .main .block-container, [data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1260px;
    }

    /* Top Header Bar */
    [data-testid="stHeader"] {
        background: rgba(6, 11, 20, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.10) !important;
    }

    /* -------------------------------------------------------------------------
       3. COMPACT TOP HEADER
       ------------------------------------------------------------------------- */
    .dashboard-header {
        background: linear-gradient(135deg, rgba(10,17,33,0.95) 0%, rgba(12,23,44,0.95) 60%, rgba(14,29,54,0.95) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.20) !important;
        border-radius: 10px !important;
        padding: 18px 24px !important;
        margin-bottom: 20px !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35) !important;
        position: relative;
        overflow: hidden;
    }

    .dashboard-header::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #0284c7 0%, #06b6d4 50%, #14b8a6 100%);
    }

    .dashboard-header-title {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em !important;
        margin: 0 !important;
        line-height: 1.25 !important;
    }

    .dashboard-header-subtitle {
        font-size: 0.92rem !important;
        color: #94a3b8 !important;
        margin-top: 4px !important;
        margin-bottom: 0 !important;
    }

    .status-badge-live {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.30);
        color: #38bdf8;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .live-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        flex-shrink: 0;
    }

    /* -------------------------------------------------------------------------
       4. SIDEBAR
       ------------------------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #040810 !important;
        background-image: linear-gradient(180deg, rgba(14,165,233,0.05) 0%, rgba(20,184,166,0.02) 50%, transparent 100%) !important;
        border-right: 1px solid rgba(51, 65, 85, 0.6) !important;
    }

    [data-testid="stSidebar"] * { color: #e2e8f0; }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(51, 65, 85, 0.45) !important;
        margin: 1.25rem 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #94a3b8 !important;
    }

    /* -------------------------------------------------------------------------
       5. METRIC CARDS
       ------------------------------------------------------------------------- */
    [data-testid="stMetric"] {
        background: rgba(12, 19, 34, 0.85) !important;
        border: 1px solid rgba(56, 189, 248, 0.18) !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.28) !important;
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(45, 212, 191, 0.4) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.35), 0 0 10px rgba(20,184,166,0.1) !important;
    }

    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
    }

    /* -------------------------------------------------------------------------
       6. BUTTONS — Clear Visual Hierarchy
       ------------------------------------------------------------------------- */

    /* ── Secondary / Preset Buttons ────────────────────────────────────── */
    button[data-testid="baseButton-secondary"],
    .stButton > button {
        background: rgba(16, 26, 46, 0.90) !important;
        color: #cbd5e1 !important;
        border: 1.5px solid rgba(71, 85, 105, 0.72) !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        padding: 0.5rem 0.9rem !important;
        min-height: 40px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.22) !important;
    }

    button[data-testid="baseButton-secondary"]:hover,
    .stButton > button:hover {
        background: rgba(22, 38, 68, 0.98) !important;
        color: #ffffff !important;
        border-color: rgba(56, 189, 248, 0.60) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3), 0 0 10px rgba(56,189,248,0.15) !important;
        transform: translateY(-1px);
    }

    button[data-testid="baseButton-secondary"]:focus-visible,
    .stButton > button:focus-visible {
        outline: 2px solid #06b6d4 !important;
        outline-offset: 2px !important;
    }

    button[data-testid="baseButton-secondary"]:active,
    .stButton > button:active { transform: translateY(0); }

    /* ── PRIMARY SUBMIT BUTTON — most prominent element in the form ─────── */
    button[data-testid="baseButton-primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #0d9488 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        border: 1.5px solid rgba(45, 212, 191, 0.55) !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1.0rem !important;
        letter-spacing: 0.01em !important;
        padding: 0.65rem 1.5rem !important;
        min-height: 46px !important;
        box-shadow: 0 4px 18px rgba(13, 148, 136, 0.40) !important;
        transition: all 0.2s ease !important;
    }

    button[data-testid="baseButton-primary"]:hover,
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #14b8a6 0%, #0ea5e9 100%) !important;
        border-color: rgba(94, 234, 212, 0.82) !important;
        box-shadow: 0 6px 24px rgba(13, 148, 136, 0.55) !important;
        transform: translateY(-1px);
    }

    button[data-testid="baseButton-primary"]:focus-visible,
    button[kind="primary"]:focus-visible {
        outline: 2px solid #5eead4 !important;
        outline-offset: 3px !important;
    }

    button[data-testid="baseButton-primary"]:active,
    button[kind="primary"]:active { transform: translateY(0); }

    /* ── CSV Download Button ─────────────────────────────────────────────── */
    [data-testid="stDownloadButton"] > button {
        background: rgba(16, 26, 46, 0.88) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(56, 189, 248, 0.42) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        min-height: 40px !important;
    }

    [data-testid="stDownloadButton"] > button:hover {
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 0 12px rgba(56,189,248,0.25) !important;
    }

    /* -------------------------------------------------------------------------
       7. FORMS, INPUT CONTROLS — High-Contrast & Accessible
       ------------------------------------------------------------------------- */

    /* Border Containers */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: rgba(12, 19, 34, 0.80) !important;
        border: 1px solid rgba(56, 189, 248, 0.18) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
        padding: 1.25rem !important;
    }

    /* Streamlit Forms */
    [data-testid="stForm"] {
        background: rgba(10, 16, 30, 0.65) !important;
        border: 1px solid rgba(56, 189, 248, 0.16) !important;
        border-radius: 10px !important;
        padding: 1.4rem !important;
    }

    /* ── ALL WIDGET LABELS — white, large, bold ─────────────────────────── */
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p {
        color: #f0f6ff !important;
        font-weight: 700 !important;
        font-size: 0.93rem !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 5px !important;
    }

    /* ── NUMBER INPUTS ───────────────────────────────────────────────────── */
    div[data-baseweb="input"] {
        background-color: #0a1020 !important;
        border: 1.5px solid rgba(71, 85, 105, 0.78) !important;
        border-radius: 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        min-height: 46px !important;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.28), 0 0 10px rgba(6,182,212,0.18) !important;
    }

    div[data-baseweb="input"] input {
        color: #f8fafc !important;
        background-color: transparent !important;
        font-size: 0.96rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 0.8rem !important;
    }

    div[data-baseweb="input"] input::placeholder { color: #475569 !important; }

    /* Number Input stepper ± buttons */
    [data-testid="stNumberInput"] button {
        background-color: #111827 !important;
        color: #cbd5e1 !important;
        border: none !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background-color: #1e2f4a !important;
        color: #ffffff !important;
    }

    [data-testid="stNumberInput"] button:focus-visible {
        outline: 2px solid #06b6d4 !important;
    }

    /* ── SELECTBOXES ─────────────────────────────────────────────────────── */
    div[data-baseweb="select"] > div {
        background-color: #0a1020 !important;
        border: 1.5px solid rgba(71, 85, 105, 0.78) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
        min-height: 46px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: rgba(56, 189, 248, 0.58) !important;
    }

    div[data-baseweb="select"] > div:focus-within {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.28) !important;
    }

    div[data-baseweb="select"] span {
        color: #f8fafc !important;
        font-size: 0.94rem !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="select"] svg { fill: #94a3b8 !important; }

    /* ── SELECTBOX DROPDOWN POPOVER ─────────────────────────────────────── */
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] {
        background-color: #0d1a2e !important;
        border: 1.5px solid rgba(71, 85, 105, 0.72) !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.48) !important;
    }

    div[data-baseweb="popover"] li {
        color: #e2e8f0 !important;
        font-size: 0.93rem !important;
        padding: 9px 14px !important;
    }

    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background-color: rgba(22, 38, 68, 0.92) !important;
        color: #2dd4bf !important;
    }

    /* ── RADIO BUTTONS — card-style rows ─────────────────────────────────── */
    /* Each option rendered as a selectable row card */
    [data-testid="stRadio"] > div {
        display: flex;
        flex-wrap: wrap;
        gap: 10px !important;
        margin-top: 4px !important;
    }

    [data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        padding: 10px 16px !important;
        border-radius: 8px !important;
        border: 1.5px solid rgba(71, 85, 105, 0.60) !important;
        background: rgba(10, 16, 30, 0.75) !important;
        cursor: pointer !important;
        transition: background 0.15s ease, border-color 0.15s ease !important;
        min-height: 46px !important;
    }

    [data-testid="stRadio"] label:hover {
        background: rgba(22, 38, 68, 0.88) !important;
        border-color: rgba(56, 189, 248, 0.58) !important;
    }

    /* Radio option text */
    [data-testid="stRadio"] label span,
    [data-testid="stRadio"] label p {
        color: #dde5f0 !important;
        font-size: 0.93rem !important;
        font-weight: 500 !important;
        line-height: 1.3 !important;
    }

    /* Selected radio row */
    [data-testid="stRadio"] label:has(input:checked) {
        background: rgba(6, 182, 212, 0.10) !important;
        border-color: rgba(6, 182, 212, 0.55) !important;
    }

    [data-testid="stRadio"] label:has(input:checked) span,
    [data-testid="stRadio"] label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Keyboard focus on radio container */
    [data-testid="stRadio"] label:focus-within {
        outline: 2px solid #06b6d4 !important;
        outline-offset: 2px !important;
    }

    /* ── TEXT INPUT ──────────────────────────────────────────────────────── */
    [data-testid="stTextInput"] div[data-baseweb="input"] {
        min-height: 46px !important;
    }

    /* -------------------------------------------------------------------------
       8. STREAMLIT TABS — Visible & Keyboard-Accessible
       ------------------------------------------------------------------------- */
    [data-testid="stTabs"] {
        border-bottom: 1px solid rgba(51, 65, 85, 0.5) !important;
        gap: 4px;
    }

    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        padding: 10px 16px !important;
        border-radius: 7px 7px 0 0 !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease !important;
        min-height: 42px !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #38bdf8 !important;
        background-color: rgba(22, 36, 62, 0.40) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #06b6d4 !important;
        background-color: rgba(6, 182, 212, 0.08) !important;
        border-bottom: 2px solid #06b6d4 !important;
    }

    button[data-baseweb="tab"]:focus-visible {
        outline: 2px solid #06b6d4 !important;
        outline-offset: 1px !important;
    }

    /* -------------------------------------------------------------------------
       9. RESULTS PANELS (Factors & Recommendations)
       ------------------------------------------------------------------------- */
    .score-indicator-container { margin-top: 10px; margin-bottom: 8px; }
    .score-indicator-header    { font-size: 0.95rem; margin-bottom: 6px; color: #f1f5f9; }

    .risk-high-badge   { color: #f87171; font-weight: 700; }
    .risk-medium-badge { color: #fbbf24; font-weight: 700; }
    .risk-low-badge    { color: #34d399; font-weight: 700; }

    .score-scale-legend {
        display: flex;
        justify-content: space-between;
        font-size: 0.73rem;
        color: #64748b;
        margin-top: 5px;
        padding: 0 2px;
    }

    .factor-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 9px;
        font-size: 0.88rem;
    }

    .indicator-elevated {
        background-color: rgba(239, 68, 68, 0.10) !important;
        border: 1px solid rgba(239, 68, 68, 0.32) !important;
    }

    .indicator-normal {
        background-color: rgba(16, 185, 129, 0.08) !important;
        border: 1px solid rgba(16, 185, 129, 0.24) !important;
    }

    .factor-info { display: flex; flex-direction: column; gap: 2px; }

    .factor-title { font-weight: 600; color: #f1f5f9; font-size: 0.89rem; }
    .factor-desc  { font-size: 0.78rem; color: #94a3b8; }

    .point-badge-elevated {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 0.80rem;
        background: rgba(239, 68, 68, 0.20);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.42);
        white-space: nowrap;
    }

    .point-badge-normal {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 0.80rem;
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.30);
        white-space: nowrap;
    }

    .response-action-card {
        background: rgba(14, 23, 42, 0.80) !important;
        border-left: 3px solid #0284c7 !important;
        border-top: 1px solid rgba(51, 65, 85, 0.40) !important;
        border-right: 1px solid rgba(51, 65, 85, 0.40) !important;
        border-bottom: 1px solid rgba(51, 65, 85, 0.40) !important;
        padding: 12px 16px !important;
        border-radius: 0 7px 7px 0 !important;
        margin-bottom: 9px !important;
        font-size: 0.92rem !important;
        color: #e2e8f0 !important;
        line-height: 1.50 !important;
    }

    .governance-disclaimer {
        font-size: 0.80rem !important;
        color: #94a3b8 !important;
        background: rgba(12, 19, 34, 0.65) !important;
        border: 1px dashed rgba(71, 85, 105, 0.50) !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        margin-top: 12px !important;
        line-height: 1.50 !important;
    }

    .safeguard-reminder {
        font-size: 0.80rem !important;
        color: #94a3b8 !important;
        background: rgba(12, 19, 34, 0.65) !important;
        border-left: 3px solid #f59e0b !important;
        border-top: 1px solid rgba(51, 65, 85, 0.30) !important;
        border-right: 1px solid rgba(51, 65, 85, 0.30) !important;
        border-bottom: 1px solid rgba(51, 65, 85, 0.30) !important;
        border-radius: 0 6px 6px 0 !important;
        padding: 10px 14px !important;
        margin-top: 12px !important;
        line-height: 1.50 !important;
    }

    /* -------------------------------------------------------------------------
       10. PROGRESS BAR, ALERTS & DATAFRAMES
       ------------------------------------------------------------------------- */
    [data-testid="stProgress"] > div {
        background-color: rgba(22, 34, 58, 0.85) !important;
        border-radius: 9999px !important;
        height: 10px !important;
    }

    [data-testid="stAlert"] {
        background-color: rgba(12, 19, 34, 0.88) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(51, 65, 85, 0.58) !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(51, 65, 85, 0.60) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        background-color: rgba(8, 13, 24, 0.60) !important;
    }

    /* -------------------------------------------------------------------------
       11. RESPONSIVE — Mobile & Desktop
       ------------------------------------------------------------------------- */
    @media (max-width: 768px) {
        .main .block-container, [data-testid="stMainBlockContainer"] {
            padding-top: 1rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }

        .dashboard-header {
            padding: 14px 16px !important;
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 10px !important;
        }

        .dashboard-header-title    { font-size: 1.3rem !important; }
        .dashboard-header-subtitle { font-size: 0.82rem !important; }

        [data-testid="stMetric"] { padding: 12px 14px !important; }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div { font-size: 1.25rem !important; }

        button[data-baseweb="tab"] {
            padding: 8px 10px !important;
            font-size: 0.82rem !important;
        }

        /* Stack radio options column on narrow screens */
        [data-testid="stRadio"] > div {
            flex-direction: column !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. SESSION STATE INITIALIZATION
# =============================================================================
if "history" not in st.session_state:
    st.session_state.history = []

# Input form field values in session state for preset autofill
if "form_login_attempts" not in st.session_state:
    st.session_state.form_login_attempts = 1
if "form_device_status" not in st.session_state:
    st.session_state.form_device_status = "Known Device (Registered)"
if "form_data_mb" not in st.session_state:
    st.session_state.form_data_mb = 25.0
if "form_time_cat" not in st.session_state:
    st.session_state.form_time_cat = "Standard Business Hours (08:00 - 18:00)"
if "form_activity_note" not in st.session_state:
    st.session_state.form_activity_note = "Routine staff portal session"


# =============================================================================
# 3. MODEL MANAGEMENT: LOAD OR PROMPT TRAINING
# =============================================================================
@st.cache_resource
def load_threat_model():
    """
    Loads serialized Isolation Forest artifact from disk.
    Returns (model_artifact, error_message).
    """
    if not os.path.exists(MODEL_FILENAME):
        return None, f"Model file '{MODEL_FILENAME}' was not found in the project root."
    try:
        artifact = joblib.load(MODEL_FILENAME)
        if not isinstance(artifact, dict) or "model" not in artifact:
            return None, "Model file is corrupt or does not contain expected keys."
        return artifact, None
    except Exception as e:
        return None, f"Error reading model file: {str(e)}"


# =============================================================================
# 4. THREAT SCORING & LOGIC EXPLANATION
# =============================================================================
def compute_cyber_threat_assessment(login_attempts, is_known_device, data_mb, is_off_hours, model_artifact):
    """
    Analyzes submitted activity using Isolation Forest + rule-based heuristics.
    
    Returns:
    - anomaly_predicted (bool): True if flagged by Isolation Forest
    - raw_decision_score (float): Sklearn decision_function output
    - threat_score (int): Normalized 0 to 100 risk score
    - risk_level (str): "Low", "Medium", or "High"
    - indicators (list of dicts): Detailed contributing factors
    - recommendations (list of str): Educational advisory triage steps
    """
    model = model_artifact["model"]
    
    feature_cols = model_artifact.get("feature_names", ["login_attempts", "is_known_device", "data_transferred_mb", "is_off_hours"])
    sample = pd.DataFrame([{
        "login_attempts": login_attempts,
        "is_known_device": is_known_device,
        "data_transferred_mb": data_mb,
        "is_off_hours": is_off_hours
    }])[feature_cols]
    
    # Isolation Forest:
    # predict() returns 1 (inlier/normal) or -1 (outlier/anomaly)
    raw_pred = model.predict(sample)[0]
    anomaly_predicted = (raw_pred == -1)
    
    # decision_function() returns anomaly score.
    # Negative scores represent outliers; positive scores represent normal samples.
    raw_decision_score = float(model.decision_function(sample)[0])
    
    # -------------------------------------------------------------
    # Rule-Based Threat Scoring (0 to 100 Scale)
    # Combining statistical isolation with transparent cybersecurity heuristics
    # -------------------------------------------------------------
    score_components = []
    
    # Component 1: Machine Learning Anomaly Score contribution (0 to 30 pts)
    # Decision scores typically range from -0.25 (severe anomaly) to +0.20 (very normal)
    if anomaly_predicted:
        # Scale negative decision score into 15 to 30 points
        ml_points = min(30, int(15 + max(0, -raw_decision_score) * 100))
        score_components.append(("Isolation Forest Statistical Outlier", ml_points, "Elevated"))
    else:
        ml_points = 0
        score_components.append(("Isolation Forest Pattern Match", 0, "Normal"))
        
    # Component 2: Failed Login Attempts (Brute Force / Password Guessing indicator)
    login_points = 0
    if login_attempts >= 15:
        login_points = 35
        login_status = "Critical (High brute-force volume)"
    elif login_attempts >= 6:
        login_points = 25
        login_status = "Elevated (Repeated authentication attempts)"
    elif login_attempts >= 3:
        login_points = 12
        login_status = "Moderate (Multiple attempts detected)"
    else:
        login_points = 0
        login_status = "Normal (Single or standard authentication)"
    score_components.append(("Authentication Attempts", login_points, login_status))
    
    # Component 3: Device Trust (Known vs Unregistered device)
    device_points = 0
    if is_known_device == 0:
        device_points = 18
        device_status = "Untrusted (Unregistered hardware/browser fingerprint)"
    else:
        device_points = 0
        device_status = "Trusted (Known organization device)"
    score_components.append(("Device Registration", device_points, device_status))
    
    # Component 4: Data Transfer Volume (Potential Exfiltration indicator)
    data_points = 0
    if data_mb > 5000:
        data_points = 28
        data_status = f"Critical Spike ({data_mb:,.1f} MB exceeds safe baseline)"
    elif data_mb > 1000:
        data_points = 18
        data_status = f"Elevated Volume ({data_mb:,.1f} MB exceeds typical session)"
    elif data_mb > 350:
        data_points = 8
        data_status = f"Moderate Volume ({data_mb:,.1f} MB)"
    else:
        data_points = 0
        data_status = f"Normal Volume ({data_mb:,.1f} MB standard browsing)"
    score_components.append(("Data Transfer Volume", data_points, data_status))
    
    # Component 5: Time of Day Category (Off-hours access)
    time_points = 0
    if is_off_hours == 1:
        time_points = 14
        time_status = "Off-Hours Activity (Late night or weekend access)"
    else:
        time_points = 0
        time_status = "Standard Business / Campus Hours"
    score_components.append(("Timing & Window", time_points, time_status))
    
    # Calculate Total Threat Score (clamped between 0 and 100)
    raw_total = ml_points + login_points + device_points + data_points + time_points
    threat_score = max(0, min(100, raw_total))
    
    # Categorize Risk Level
    if threat_score >= 68:
        risk_level = "High"
    elif threat_score >= 32:
        risk_level = "Medium"
    else:
        risk_level = "Low"
        
    # Generate Advisory Recommendations (Educational, strictly non-offensive)
    recommendations = []
    if risk_level == "High":
        recommendations = [
            "Promptly notify the responsible IT/security administrator.",
            "Review recent account and device activity.",
            "Follow the organization's incident-response procedure."
        ]
    elif risk_level == "Medium":
        recommendations = [
            "Verify the activity with the account owner using an approved channel.",
            "Review related sign-in records.",
            "Follow the organization's reporting process if it remains unusual."
        ]
    else:
        recommendations = [
            "Review the activity and compare it with expected behavior."
        ]
        
    return {
        "anomaly_predicted": anomaly_predicted,
        "raw_decision_score": raw_decision_score,
        "threat_score": threat_score,
        "risk_level": risk_level,
        "components": score_components,
        "recommendations": recommendations
    }


def render_threat_gauge_svg(score: int, risk_level: str) -> str:
    """
    Renders an accessible, zero-dependency SVG semi-circular gauge
    visualizing the threat score from 0 to 100 with color-coded risk bands.
    """
    clamped_score = max(0, min(100, int(score)))
    # Semicircular arc with radius 85: arc length = pi * 85 ~= 267.04
    arc_length = 267.04
    offset = arc_length * (1.0 - (clamped_score / 100.0))
    
    if risk_level == "High":
        meter_color = "#ef4444"
        risk_bg = "rgba(239, 68, 68, 0.15)"
        risk_border = "rgba(239, 68, 68, 0.45)"
        risk_text_color = "#f87171"
    elif risk_level == "Medium":
        meter_color = "#f59e0b"
        risk_bg = "rgba(245, 158, 11, 0.15)"
        risk_border = "rgba(245, 158, 11, 0.45)"
        risk_text_color = "#fbbf24"
    else:
        meter_color = "#06b6d4"
        risk_bg = "rgba(6, 182, 212, 0.15)"
        risk_border = "rgba(6, 182, 212, 0.45)"
        risk_text_color = "#22d3ee"
        
    return f'''
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:10px 4px; margin: 0 auto; width:100%; max-width:260px;">
        <svg viewBox="0 0 240 135" style="width:100%; max-width:240px; overflow:visible;" role="img" aria-label="Threat score meter: {clamped_score} out of 100, {risk_level} risk">
            <defs>
                <linearGradient id="gaugeTrackGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#14b8a6" stop-opacity="0.25" />
                    <stop offset="50%" stop-color="#f59e0b" stop-opacity="0.25" />
                    <stop offset="100%" stop-color="#ef4444" stop-opacity="0.25" />
                </linearGradient>
            </defs>
            <!-- Background Track -->
            <path d="M 35 112 A 85 85 0 0 1 205 112" 
                  fill="none" 
                  stroke="rgba(30, 41, 59, 0.9)" 
                  stroke-width="14" 
                  stroke-linecap="round" />
            <!-- Active Fill -->
            <path d="M 35 112 A 85 85 0 0 1 205 112" 
                  fill="none" 
                  stroke="{meter_color}" 
                  stroke-width="14" 
                  stroke-linecap="round" 
                  stroke-dasharray="{arc_length}" 
                  stroke-dashoffset="{offset}" />
            <!-- Centered Score Value -->
            <text x="120" y="92" text-anchor="middle" fill="#ffffff" font-size="34" font-weight="700" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">
                {clamped_score}
            </text>
            <text x="120" y="110" text-anchor="middle" fill="#94a3b8" font-size="12" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">
                Threat Score / 100
            </text>
            <!-- Scale Bounds -->
            <text x="32" y="128" text-anchor="middle" fill="#64748b" font-size="10" font-family="monospace">0</text>
            <text x="120" y="24" text-anchor="middle" fill="#64748b" font-size="10" font-family="monospace">50</text>
            <text x="208" y="128" text-anchor="middle" fill="#64748b" font-size="10" font-family="monospace">100</text>
        </svg>
        <div style="margin-top:4px; display:inline-block; padding:3px 12px; border-radius:9999px; background:{risk_bg}; border:1px solid {risk_border}; color:{risk_text_color}; font-size:0.80rem; font-weight:700; letter-spacing:0.04em;">
            {risk_level.upper()} RISK
        </div>
    </div>
    '''


# =============================================================================
# 5. PRESET SAMPLES & FORM HELPERS
# =============================================================================
def apply_preset(name):
    """Fills form state with curated safe demonstration scenarios."""
    if name == "normal":
        st.session_state.form_login_attempts = 1
        st.session_state.form_device_status = "Known Device (Registered)"
        st.session_state.form_data_mb = 32.5
        st.session_state.form_time_cat = "Standard Business Hours (08:00 - 18:00)"
        st.session_state.form_activity_note = "Regular staff member checking portal"
    elif name == "brute_force":
        st.session_state.form_login_attempts = 18
        st.session_state.form_device_status = "Unknown / Unregistered Device"
        st.session_state.form_data_mb = 12.0
        st.session_state.form_time_cat = "Standard Business Hours (08:00 - 18:00)"
        st.session_state.form_activity_note = "Spike in failed authentication attempts"
    elif name == "exfiltration":
        st.session_state.form_login_attempts = 1
        st.session_state.form_device_status = "Known Device (Registered)"
        st.session_state.form_data_mb = 6450.0
        st.session_state.form_time_cat = "Late Night / Off-Hours (23:00 - 08:00)"
        st.session_state.form_activity_note = "Unusually massive off-hours data transfer"
    elif name == "intrusion":
        st.session_state.form_login_attempts = 8
        st.session_state.form_device_status = "Unknown / Unregistered Device"
        st.session_state.form_data_mb = 1820.0
        st.session_state.form_time_cat = "Late Night / Off-Hours (23:00 - 08:00)"
        st.session_state.form_activity_note = "Off-hours access from unrecognized hardware"
    elif name == "reset":
        st.session_state.form_login_attempts = 1
        st.session_state.form_device_status = "Known Device (Registered)"
        st.session_state.form_data_mb = 25.0
        st.session_state.form_time_cat = "Standard Business Hours (08:00 - 18:00)"
        st.session_state.form_activity_note = ""


# =============================================================================
# 6. SIDEBAR: MODEL STATUS & QUICK ACTIONS
# =============================================================================
with st.sidebar:
    st.markdown("### System Controls")
    st.markdown("**Cyber Threat Detection**  \n*Activity Risk Assessment Engine*")
    
    st.divider()
    
    # Model Status Check
    model_artifact, load_err = load_threat_model()
    
    if model_artifact:
        st.success("Model Active: Isolation Forest")
        meta = model_artifact
        st.caption(f"**Version:** `{meta.get('version', '1.0.0')}`")
        st.caption(f"**Trained Samples:** `{meta.get('training_samples', 'N/A'):,}`")
        st.caption(f"**Contamination:** `{meta.get('contamination', 0.09):.0%}`")
        if "created_at" in meta:
            st.caption(f"**Trained On:** `{meta['created_at']}`")
    else:
        st.error("Model Not Loaded")
        st.info("The trained model artifact is missing or corrupted.")
        if st.button("Train Model Now (In-App)", use_container_width=True):
            if train_isolation_forest:
                with st.spinner("Generating synthetic data and training Isolation Forest..."):
                    train_isolation_forest()
                    st.cache_resource.clear()
                    st.success("Model trained successfully! Reloading...")
                    st.rerun()
            else:
                st.error("`train_model.py` module could not be imported.")
        st.caption("Or run in terminal: `python train_model.py`")

    st.divider()
    
    # Session History Management
    st.markdown("### Session Log")
    history_count = len(st.session_state.history)
    st.metric("Logged Session Events", history_count)
    
    if history_count > 0:
        if st.button("Clear Session History", use_container_width=True):
            st.session_state.history = []
            st.toast("Session history cleared.")
            st.rerun()
            
    st.divider()
    
    # Operational Reference
    st.markdown("### Operational Notes")
    st.markdown("""
    - Use **Synthetic Demonstration Profiles** to evaluate nominal and high-deviation traffic.
    - Review the transparent factor breakdown to inspect individual feature contributions.
    - All outputs are advisory guidance to support human verification.
    """)


# =============================================================================
# 7. MAIN INTERFACE
# =============================================================================

# Header
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-header-left">
        <div class="dashboard-header-title">Cyber Threat Detection</div>
        <div class="dashboard-header-subtitle">Review activity patterns. Understand potential risk. Decide what to check next.</div>
    </div>
    <div class="status-badge-live">
        <span class="live-dot"></span> System Ready
    </div>
</div>
""", unsafe_allow_html=True)

# Missing model alert if applicable
if not model_artifact:
    st.warning(f"Setup Required: {load_err}")
    st.markdown("""
    To train the synthetic Isolation Forest model:
    1. Click the **'Train Model Now (In-App)'** button in the left sidebar, or
    2. Open your terminal in this directory and execute:
    ```bash
    python train_model.py
    ```
    """)
    st.stop()


# Tabs for Main Experience
tab_analyze, tab_history, tab_about = st.tabs([
    "Activity Analysis & Triage",
    f"Session Activity History ({len(st.session_state.history)})",
    "Governance, Privacy & Limitations"
])


# =============================================================================
# TAB 1: ANALYZE ACTIVITY & TRIAGE
# =============================================================================
with tab_analyze:
    
    # Quick Activity Profile Preset Buttons
    st.markdown("##### Synthetic Demonstration Profiles (Simulated Data)")
    st.caption("Load simulated activity patterns to evaluate baseline and elevated risk profiles. Synthetic demo profiles only — not real user activity:")
    
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    
    with col_p1:
        if st.button("Standard User", use_container_width=True, help="1 login, known device, 32.5 MB, business hours"):
            apply_preset("normal")
            st.rerun()
    with col_p2:
        if st.button("Repeated Logins", use_container_width=True, help="18 logins, unknown device, business hours"):
            apply_preset("brute_force")
            st.rerun()
    with col_p3:
        if st.button("Data Transfer Spike", use_container_width=True, help="6,450 MB transfer spike, off-hours"):
            apply_preset("exfiltration")
            st.rerun()
    with col_p4:
        if st.button("Off-Hours Unknown Device", use_container_width=True, help="8 logins, unknown device, 1,820 MB, late night"):
            apply_preset("intrusion")
            st.rerun()
    with col_p5:
        if st.button("Reset Inputs", use_container_width=True, help="Clear and reset form fields to default"):
            apply_preset("reset")
            st.rerun()
            
    st.write("")
    
    # Input Form Container
    with st.container(border=True):
        st.markdown("##### Session Activity Parameters")
        st.caption("Provide observable network and authentication telemetry for assessment. No credentials or private data are collected.")
        
        with st.form("threat_analysis_form"):
            c1, c2 = st.columns(2)
            
            with c1:
                input_logins = st.number_input(
                    "Login Attempts in Window",
                    min_value=1,
                    max_value=50,
                    value=st.session_state.form_login_attempts,
                    step=1,
                    help="Count of login attempts recorded within the session window."
                )
                
                device_options = ["Known Device (Registered)", "Unknown / Unregistered Device"]
                curr_dev_idx = 0 if "Known" in st.session_state.form_device_status else 1
                input_device_label = st.radio(
                    "Device Recognition Status",
                    options=device_options,
                    index=curr_dev_idx,
                    horizontal=True,
                    help="Indicates whether the client hardware or certificate has been registered in organizational inventory."
                )
                
            with c2:
                input_data_mb = st.number_input(
                    "Total Data Transferred (MB)",
                    min_value=0.1,
                    max_value=50000.0,
                    value=float(st.session_state.form_data_mb),
                    step=25.0,
                    help="Megabytes downloaded or uploaded during the active session."
                )
                
                time_categories = [
                    "Standard Business Hours (08:00 - 18:00)",
                    "Evening / Extended Hours (18:00 - 23:00)",
                    "Late Night / Off-Hours (23:00 - 08:00)"
                ]
                curr_time_idx = 0
                for idx, t_cat in enumerate(time_categories):
                    if t_cat == st.session_state.form_time_cat:
                        curr_time_idx = idx
                        break
                        
                input_time_cat = st.selectbox(
                    "Session Time Window",
                    options=time_categories,
                    index=curr_time_idx,
                    help="Temporal context for when the session was initiated."
                )
                
            input_note = st.text_input(
                "Activity Context / Reference Note (Optional)",
                value=st.session_state.form_activity_note,
                placeholder="e.g., Workstation #12, remote research gateway, finance sync"
            )
            
            col_submit, col_hint = st.columns([1, 2])
            with col_submit:
                submitted = st.form_submit_button("Analyze Activity", type="primary", use_container_width=True)
            with col_hint:
                st.caption("Telemetry is evaluated locally against the Isolation Forest model and heuristic rules.")

    # Process and display analysis results
    if submitted:
        # Input Validation
        if input_logins < 1 or input_logins > 50:
            st.error("Validation Error: Login attempts must be between 1 and 50.")
            st.stop()
        if input_data_mb < 0.1 or input_data_mb > 50000.0:
            st.error("Validation Error: Total data transferred must be between 0.1 MB and 50,000 MB.")
            st.stop()
            
        # Map user inputs to model features
        is_known = 1 if "Known" in input_device_label else 0
        is_off_hours = 1 if "Late Night" in input_time_cat else 0
        
        # Compute assessment
        result = compute_cyber_threat_assessment(
            login_attempts=input_logins,
            is_known_device=is_known,
            data_mb=input_data_mb,
            is_off_hours=is_off_hours,
            model_artifact=model_artifact
        )
        
        # Save to session history (in-memory only)
        record = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "event_note": input_note if input_note else "Unlabeled Event",
            "logins": input_logins,
            "device": "Known" if is_known == 1 else "Unknown",
            "data_mb": round(input_data_mb, 2),
            "timing": input_time_cat.split(" (")[0],
            "model_pred": "Anomaly" if result["anomaly_predicted"] else "Normal",
            "threat_score": result["threat_score"],
            "risk_level": result["risk_level"]
        }
        st.session_state.history.insert(0, record)  # Newest first
        
        st.write("")
        st.markdown("### Assessment Overview")
        
        # Clear explanation that an anomaly is unusual activity, not proof of an attack
        if result["anomaly_predicted"]:
            st.warning(
                "**Anomaly Detected (Unusual Activity):** The activity deviates statistically from baseline patterns. "
                "An anomaly indicates unusual activity, not proof of an attack. Review the contributing indicators below before deciding what to check next."
            )
        else:
            st.success(
                "**Baseline Activity Profile:** The activity patterns conform to typical baseline distributions. "
                "No statistical anomaly was flagged by the Isolation Forest model."
            )
            
        # Score Visualization (Gauge & Metrics)
        with st.container(border=True):
            col_gauge, col_stats = st.columns([1, 1])
            
            with col_gauge:
                st.markdown(render_threat_gauge_svg(result["threat_score"], result["risk_level"]), unsafe_allow_html=True)
                
            with col_stats:
                st.markdown("##### Threat Score & Telemetry Metrics")
                st.markdown(f"""
                <div style="font-size:0.95rem; margin-bottom:8px;">
                    <strong>Threat Score:</strong> <span style="font-size:1.25rem; font-weight:700; color:#ffffff;">{result['threat_score']} / 100</span>
                    &nbsp;&bull;&nbsp;
                    <strong>Risk Level:</strong> <span style="font-size:1.1rem; font-weight:700; color:#38bdf8;">{result['risk_level'].upper()}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Visual Threat Score Meter
                st.progress(result["threat_score"] / 100.0)
                st.markdown("""
                <div class="score-scale-legend">
                    <span>0 Baseline (Low)</span>
                    <span>32 Moderate (Medium)</span>
                    <span>68 Elevated (High)</span>
                    <span>100 Maximum</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                col_sub1, col_sub2 = st.columns(2)
                with col_sub1:
                    flag_label = "Anomaly" if result["anomaly_predicted"] else "Baseline"
                    st.metric("Isolation Forest Flag", flag_label)
                with col_sub2:
                    st.metric("Raw Decision Score", f"{result['raw_decision_score']:.4f}", help="Negative scores represent statistical outliers; positive scores represent typical clusters.")
                    
        st.write("")
        
        # Contributing Factor Breakdown & Action Recommendations
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            with st.container(border=True):
                st.markdown("#### Why this activity was flagged")
                st.caption("Observable telemetry indicators contributing to the result:")
                
                for name, pts, desc in result["components"]:
                    if pts > 0:
                        item_class = "indicator-elevated"
                        badge_markup = f'<span class="point-badge-elevated">+{pts} pts</span>'
                    else:
                        item_class = "indicator-normal"
                        badge_markup = f'<span class="point-badge-normal">0 pts</span>'
                        
                    st.markdown(f"""
                    <div class="factor-item {item_class}">
                        <div class="factor-info">
                            <div class="factor-title">{name}</div>
                            <div class="factor-desc">{desc}</div>
                        </div>
                        <div>{badge_markup}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("""
                <div class="governance-disclaimer">
                    <strong>Notice:</strong> An anomaly indicates unusual activity, not proof of an attack. The threat score is an advisory estimate blending statistical outlier isolation with observable telemetry indicators.
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            with st.container(border=True):
                st.markdown("#### Recommended response")
                st.caption(f"Practical next steps for {result['risk_level'].lower()} risk activity:")
                
                for idx, rec in enumerate(result["recommendations"], start=1):
                    st.markdown(f"""
                    <div class="response-action-card">
                        <span style="display:inline-block; width:22px; height:22px; border-radius:50%; background:rgba(6,182,212,0.15); color:#38bdf8; text-align:center; line-height:22px; font-weight:700; font-size:0.78rem; margin-right:8px;">{idx}</span>
                        {rec}
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("""
                <div class="safeguard-reminder">
                    <strong>Advisory Scope:</strong> All recommendations are practical next steps for a person to review. The application does not automatically block accounts or devices, disable access, scan networks, or perform security actions.
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: SESSION ACTIVITY LOG
# =============================================================================
with tab_history:
    st.markdown("### Session Activity History")
    st.caption("All activity evaluated during the current browser session is retained in volatile memory. No data is written to disk or external systems.")
    
    if len(st.session_state.history) == 0:
        st.info("No activity records evaluated yet in this session. Head to the 'Activity Analysis & Triage' tab to run your first check.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        
        # Rename for clean presentation
        df_display = df_hist.rename(columns={
            "timestamp": "Time",
            "event_note": "Context Note",
            "logins": "Login Attempts",
            "device": "Device",
            "data_mb": "Data (MB)",
            "timing": "Time Window",
            "model_pred": "AI Prediction",
            "threat_score": "Threat Score",
            "risk_level": "Risk Level"
        })
        
        st.dataframe(
            df_display,
            use_container_width=True,
            column_config={
                "Threat Score": st.column_config.ProgressColumn(
                    "Threat Score",
                    help="Calculated threat score 0 to 100",
                    format="%d",
                    min_value=0,
                    max_value=100
                ),
                "Risk Level": st.column_config.TextColumn("Risk Level"),
                "Data (MB)": st.column_config.NumberColumn("Data (MB)", format="%.1f MB")
            }
        )
        
        c_dl, c_clear = st.columns([1, 4])
        with c_dl:
            csv_data = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Session Log (CSV)",
                data=csv_data,
                file_name=f"cyber_threat_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )


# =============================================================================
# TAB 3: ABOUT, LIMITATIONS & RESPONSIBLE USE
# =============================================================================
with tab_about:
    st.markdown("### System Architecture, Governance & Limitations")
    st.caption("Overview of the detection model, privacy safeguards, and operational boundaries.")
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        with st.container(border=True):
            st.markdown("##### Purpose & Audience")
            st.markdown("""
            This application helps colleges, small organizations, and small businesses review manually entered activity patterns—such as repeated failed logins, access from an unknown device, or unusually high data transfers—without requiring complex or costly monitoring infrastructure.
            
            The application estimates risk, explains the indicators behind each result, and suggests practical next steps for a person to review.
            """)
            
        with st.container(border=True):
            st.markdown("##### Isolation Forest Anomaly Detection")
            st.markdown("""
            - **Unsupervised Learning:** The Scikit-learn Isolation Forest model isolates outliers by recursively partitioning feature dimensions across an ensemble of isolation trees.
            - **Anomaly != Attack:** An anomaly indicates unusual activity that deviates from statistical patterns, not proof of an attack or malicious intent. Benign operations (like large legitimate backups or mistyped passwords) can trigger anomalies.
            - **Synthetic Training:** The model is trained on synthetic probability distributions representing typical session telemetry and simulated outlier clusters.
            """)
            
    with col_a2:
        with st.container(border=True):
            st.markdown("##### Privacy & Data Handling")
            st.markdown("""
            - **Zero Credential Ingestion:** The application never asks for passwords, API keys, authentication tokens, personal information, or confidential organizational data.
            - **Ephemeral Session Memory:** Activity data is retained strictly in volatile browser session memory and is never written to disk or transmitted externally.
            - **No Secret Storage:** Secrets and credentials are never stored in source code.
            """)
            
        with st.container(border=True):
            st.markdown("##### Operational Limitations & Advisory Scope")
            st.markdown("""
            - **Advisory Guidance Only:** Recommendations are practical suggestions for a person to review. The application does not automatically block accounts or devices, disable access, scan networks, or perform security actions.
            - **No Replacement for Professional Monitoring:** The system does not claim production-grade accuracy and is not a replacement for professional security monitoring or dedicated SOC analysts.
            - **Human Corroboration:** Always corroborate flagged activity through approved secondary channels before initiating institutional incident response.
            """)

# Footer
st.divider()
st.caption("Cyber Threat Detection & Intelligent Response | Security Operations Center Activity Triage")
