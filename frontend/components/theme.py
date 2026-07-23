"""Flight-deck instrument theme with a light/dark toggle.

A subject-grounded aerospace identity applied by styling Streamlit's own
components (no custom-HTML rebuilds). Two appearances share the same identity —
a dark "night cockpit" and a light "day cockpit" — plus the real aviation
caution/warning status colors (red/amber/green = High/Medium/Low), an
instrument-cyan accent, a squared avionics display face (Chakra Petch) and
monospace telemetry readouts (IBM Plex Mono).

`apply_theme()` renders the sidebar Light-mode toggle, stores the choice in
`st.session_state["theme_mode"]` (so charts can match it), injects the matching
CSS, and returns the mode. Call once per page after set_page_config.
"""

from __future__ import annotations

import streamlit as st

ACCENT = "#4FD8EB"

# Per-mode design tokens. Status colors (red/amber/green) are shared and live in
# utils.config.RISK_COLORS; they read well on both appearances.
_TOKENS = {
    "dark": {
        "bg": "#0B0E12", "panel": "#141A21", "panel2": "#1B232C",
        "border": "rgba(122,162,186,0.18)", "text": "#E4EAF0", "dim": "#8A97A6",
        "accent": "#4FD8EB", "accentink": "#06222A", "litedge": "rgba(79,216,235,0.35)",
        "field": "#0B0E12", "headerbg": "rgba(11,14,18,0.7)", "sidebar": "#090C10",
        "scroll": "rgba(122,162,186,0.22)",
    },
    "light": {
        "bg": "#EDF1F5", "panel": "#FFFFFF", "panel2": "#F6F8FB",
        "border": "rgba(20,45,65,0.14)", "text": "#0B1A24", "dim": "#5B6B78",
        "accent": "#0898B0", "accentink": "#FFFFFF", "litedge": "rgba(8,152,176,0.55)",
        "field": "#FFFFFF", "headerbg": "rgba(237,241,245,0.75)", "sidebar": "#E3E8EF",
        "scroll": "rgba(20,45,65,0.20)",
    },
}

_STATIC_ROOT = (
    ":root{"
    "--fd-radius:8px;"
    "--fd-display:'Chakra Petch','Space Grotesk',system-ui,sans-serif;"
    "--fd-body:'Inter',system-ui,'Segoe UI',Roboto,sans-serif;"
    "--fd-mono:'IBM Plex Mono',ui-monospace,'SF Mono',Menlo,monospace;"
    "}"
)

_CSS_BODY = """
.stApp { background: var(--fd-bg); }
html, body, .stApp, [class*="css"], p, li, .stMarkdown, label {
  font-family: var(--fd-body) !important; color: var(--fd-text);
  -webkit-font-smoothing: antialiased;
}
.block-container { padding-top: 3rem !important; max-width: 1180px; }

h1, h2, h3, [data-testid="stHeading"] {
  font-family: var(--fd-display) !important; letter-spacing: 0.01em !important;
  color: var(--fd-text) !important;
}
h1 { font-weight: 700 !important; }
h2, h3 { font-weight: 600 !important; }
[data-testid="stCaptionContainer"], .stCaption {
  color: var(--fd-dim) !important; text-transform: uppercase;
  letter-spacing: 0.09em; font-size: 0.72rem !important;
}

[data-testid="stMetric"] {
  background: linear-gradient(180deg, var(--fd-panel2), var(--fd-panel)) !important;
  border: 1px solid var(--fd-border) !important;
  border-top: 1px solid var(--fd-litedge) !important;
  border-radius: var(--fd-radius) !important; padding: 0.9rem 1.1rem !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 6px 20px rgba(0,0,0,0.28);
}
[data-testid="stMetricValue"] {
  font-family: var(--fd-mono) !important; font-weight: 600 !important;
  color: var(--fd-text) !important; letter-spacing: -0.01em;
}
[data-testid="stMetricLabel"] {
  text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.7rem !important;
  color: var(--fd-dim) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] details {
  background: var(--fd-panel) !important; border: 1px solid var(--fd-border) !important;
  border-radius: var(--fd-radius) !important;
}

.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  font-family: var(--fd-display) !important; text-transform: uppercase;
  letter-spacing: 0.06em; font-weight: 600 !important;
  border-radius: var(--fd-radius) !important; border: 1px solid var(--fd-border) !important;
  background: var(--fd-panel2) !important; color: var(--fd-text) !important;
  transition: transform 100ms ease, filter 200ms ease, background 200ms ease;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button {
  background: var(--fd-accent) !important; color: var(--fd-accentink) !important;
  border-color: transparent !important;
  box-shadow: 0 0 0 1px var(--fd-litedge), 0 0 18px var(--fd-litedge);
}
.stButton > button:hover, .stFormSubmitButton > button:hover { filter: brightness(1.08); }
.stButton > button:active, .stFormSubmitButton > button:active { transform: scale(0.97); }

[data-testid="stNumberInput"] input, .stTextInput input {
  font-family: var(--fd-mono) !important; background: var(--fd-field) !important;
  border: 1px solid var(--fd-border) !important; border-radius: 6px !important;
  color: var(--fd-text) !important;
}
[data-baseweb="select"] > div {
  background: var(--fd-field) !important; border: 1px solid var(--fd-border) !important;
  border-radius: 6px !important;
}
[data-testid="stNumberInput"] input:focus, .stTextInput input:focus {
  border-color: var(--fd-accent) !important; box-shadow: 0 0 0 2px var(--fd-litedge) !important;
}

[data-testid="stHeader"] { background: var(--fd-headerbg) !important; backdrop-filter: blur(10px); }
[data-testid="stSidebar"] { background: var(--fd-sidebar) !important; border-right: 1px solid var(--fd-border); }
[data-testid="stSidebarNav"] a span { font-family: var(--fd-display) !important; letter-spacing: 0.03em; }
footer, [data-testid="stFooter"] { display: none !important; }

[data-testid="stAlert"] {
  background: var(--fd-panel) !important; border: 1px solid var(--fd-border) !important;
  border-left: 3px solid var(--fd-accent) !important; border-radius: 6px !important;
  font-family: var(--fd-mono) !important;
}
.stTabs [data-baseweb="tab"] { font-family: var(--fd-display); letter-spacing: 0.03em; }
.stTabs [aria-selected="true"] { color: var(--fd-accent) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--fd-border); border-radius: 6px; overflow: hidden; }
hr { border-color: var(--fd-border) !important; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--fd-scroll); border-radius: 6px; }

/* --- Deep components: keep them on-theme in both appearances ------------ */
/* Header toolbar (Deploy, menu) + sidebar collapse control */
[data-testid="stToolbar"] button, [data-testid="stToolbar"] a,
[data-testid="stMainMenu"] svg, [data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapseButton"] { color: var(--fd-text) !important; fill: var(--fd-text) !important; }

/* Sidebar text follows the theme (nav, labels, selector value) */
[data-testid="stSidebar"] a, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: var(--fd-text) !important; }

/* Selectbox: control field, value text, dropdown popover + options.
   Current Streamlit uses react-aria ComboBox; baseweb kept for older versions. */
.stApp [data-testid="stSelectbox"] .react-aria-ComboBox > div,
.stApp [data-baseweb="select"] > div { background: var(--fd-field) !important; }
.stApp [data-testid="stSelectbox"] .react-aria-ComboBox, .stApp [data-baseweb="select"] div {
  color: var(--fd-text) !important;
}
.react-aria-Popover, .react-aria-ListBox,
[data-baseweb="popover"] ul, [data-baseweb="menu"], ul[role="listbox"] {
  background: var(--fd-panel) !important; border: 1px solid var(--fd-border) !important;
  color: var(--fd-text) !important;
}
.react-aria-Option, li[role="option"] { color: var(--fd-text) !important; }
.react-aria-Option[data-focused], .react-aria-Option[aria-selected="true"],
li[role="option"]:hover, li[role="option"][aria-selected="true"] { background: var(--fd-panel2) !important; }

/* Number-input step buttons */
[data-testid="stNumberInput"] button {
  background: var(--fd-panel2) !important; border-color: var(--fd-border) !important;
  color: var(--fd-text) !important;
}

/* File uploader dropzone + its Browse button */
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] section {
  background: var(--fd-field) !important; border: 1px dashed var(--fd-border) !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--fd-text) !important; }
[data-testid="stFileUploader"] button {
  background: var(--fd-panel2) !important; border: 1px solid var(--fd-border) !important;
  color: var(--fd-text) !important;
}

/* Expander summary, tab underline, tooltips */
[data-testid="stExpander"] summary { color: var(--fd-text) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--fd-accent) !important; }
[data-baseweb="tooltip"], [role="tooltip"] { background: var(--fd-panel2) !important; color: var(--fd-text) !important; }

@media (prefers-reduced-motion: reduce) {
  .stButton > button, .stFormSubmitButton > button { transition: none !important; }
  .stButton > button:active, .stFormSubmitButton > button:active { transform: none !important; }
}
"""


def _root_vars(tokens: dict) -> str:
    return ":root{" + "".join(f"--fd-{k}:{v};" for k, v in tokens.items()) + "}"


def apply_theme() -> str:
    """Render the sidebar Light-mode toggle, inject the matching CSS, return mode.

    The choice persists across pages: Streamlit resets widget state on page
    navigation unless the key is re-assigned to itself each run, so we do that.
    """
    if "__fd_light" in st.session_state:
        st.session_state["__fd_light"] = st.session_state["__fd_light"]
    light = st.sidebar.toggle("Light mode", key="__fd_light")
    mode = "light" if light else "dark"
    st.session_state["theme_mode"] = mode

    css = (
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?"
        "family=Chakra+Petch:wght@500;600;700&family=Inter:wght@400;500;600&"
        "family=IBM+Plex+Mono:wght@400;500;600&display=swap');"
        + _STATIC_ROOT + _root_vars(_TOKENS[mode]) + _CSS_BODY + "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)
    return mode
