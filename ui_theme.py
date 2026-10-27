from __future__ import annotations

import streamlit as st


def inject_executive_theme(theme: str = "dark_executive") -> None:
    """
    Streamlit CSS theme injector.

    Supported:
    - "dark_executive": dark glassmorphism executive look.
    """
    if theme != "dark_executive":
        raise ValueError(f"Unknown theme: {theme}")

    st.markdown(
        """
<style>
/* ---- Executive Dark Glass Theme ---- */
:root{
  --bg0: #070b14;
  --bg1: #0b1220;
  --card: rgba(255,255,255,0.07);
  --card2: rgba(255,255,255,0.10);
  --stroke: rgba(255,255,255,0.14);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.70);
  --accent: #66e3ff;
  --accent2: #2ee59d;
  --danger: #ff4d6d;
  --shadow: 0 12px 40px rgba(0,0,0,0.35);
  --radius: 18px;
}

.stApp{
  background:
    radial-gradient(1200px circle at 10% 10%, rgba(102,227,255,0.12), transparent 60%),
    radial-gradient(900px circle at 90% 20%, rgba(46,229,157,0.10), transparent 60%),
    radial-gradient(700px circle at 80% 90%, rgba(255,77,109,0.10), transparent 55%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 100%);
  color: var(--text);
}

/* Readability: force light text on dark background */
h1, h2, h3, h4, h5, h6{ color: var(--text) !important; }
p, li{ color: var(--text) !important; }
label[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] *{ color: var(--text) !important; }
div[data-testid="stMetricLabel"] *{ color: var(--muted) !important; }
div[data-testid="stMetricValue"] *{ color: var(--text) !important; }
div[data-testid="stMetricDelta"] *{ color: var(--muted) !important; }
div[data-testid="stMarkdownContainer"], div[data-testid="stMarkdownContainer"] *{ color: var(--text) !important; }
div[data-testid="stCaptionContainer"] p{ color: var(--muted) !important; }
div[data-testid="stDataFrame"] *{ color: var(--text) !important; }
div[data-testid="stChatMessage"] *{ color: var(--text) !important; }
div[data-testid="stSlider"] *{ color: var(--text) !important; }
div[data-testid="stTabs"] *{ color: var(--text) !important; }

/* Links */
a{ color: var(--accent) !important; }

/* Reduce Streamlit chrome */
header[data-testid="stHeader"]{ background: transparent; }
footer{ visibility: hidden; }

/* Container width + padding */
.block-container{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

/* Buttons */
.stButton>button{
  border-radius: 12px;
  border: 1px solid var(--stroke);
  background: linear-gradient(135deg, rgba(102,227,255,0.12), rgba(46,229,157,0.10));
  color: var(--text);
  box-shadow: none;
}
.stButton>button:hover{
  border-color: rgba(102,227,255,0.45);
}

/* Inputs */
div[data-baseweb="input"] > div{
  border-radius: 12px;
}
div[data-baseweb="select"] > div{
  border-radius: 12px;
}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-testid="stChatInput"] textarea{
  color: var(--text) !important;
  caret-color: var(--accent) !important;
  background: rgba(255,255,255,0.06) !important;
}
/* Chatbox: darken only the chat input container for readability */
div[data-testid="stChatInput"] > div{
  background: rgba(11,18,32,0.92) !important;
}
div[data-testid="stChatInput"] div[data-baseweb="textarea"] > div,
div[data-testid="stChatInput"] div[data-baseweb="input"] > div{
  background: rgba(11,18,32,0.92) !important;
}
div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] input{
  background: transparent !important;
}
/* Date inputs + multiselect search inputs sometimes bypass baseweb selectors */
div[data-testid="stDateInput"] input,
div[data-testid="stMultiSelect"] input,
div[data-testid="stSelectbox"] input{
  color: var(--text) !important;
  caret-color: var(--accent) !important;
  background: rgba(255,255,255,0.06) !important;
}
div[data-baseweb="select"] input{
  color: var(--text) !important;
  background: transparent !important;
}
div[data-baseweb="popover"] input,
div[data-baseweb="popover"] textarea{
  color: var(--text) !important;
  background: rgba(255,255,255,0.06) !important;
}
input::placeholder, textarea::placeholder{
  color: rgba(255,255,255,0.55) !important;
}
div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
}
div[data-baseweb="select"] *{ color: var(--text) !important; }

/* Dropdown menus (BaseWeb popovers) */
div[data-baseweb="popover"] *{ color: var(--text) !important; }
ul[role="listbox"], div[data-baseweb="menu"]{
  background: rgba(11,18,32,0.98) !important;
  border: 1px solid var(--stroke) !important;
}

/* Metrics as cards */
div[data-testid="stMetric"]{
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  padding: 14px 16px;
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow);
}

/* Expanders */
div[data-testid="stExpander"]{
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  padding: 6px 10px;
  backdrop-filter: blur(10px);
}

/* Dataframes */
div[data-testid="stDataFrame"]{
  background: rgba(0,0,0,0.10);
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
}

/* Chat container */
div[data-testid="stChatMessage"]{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 6px 10px;
}

/* Headings */
h1, h2, h3, h4{ letter-spacing: 0.2px; }
small, .muted{ color: var(--muted); }
</style>
        """,
        unsafe_allow_html=True,
    )


def executive_card(title: str, body_md: str) -> None:
    """
    Simple glass card renderer (HTML wrapper).
    """
    st.markdown(
        f"""
<div style="
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 12px 40px rgba(0,0,0,0.35);
  backdrop-filter: blur(10px);
  margin-bottom: 12px;
">
  <div style="font-weight: 700; margin-bottom: 8px;">{title}</div>
  <div style="color: rgba(255,255,255,0.86);">{body_md}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
