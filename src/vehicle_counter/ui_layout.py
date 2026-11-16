"""Shared presentation and navigation for the Streamlit workspace."""

import streamlit as st

SETUP = "Siapkan analisis"
QUEUE = "Antrean"
RESULTS = "Hasil"
HISTORY = "Riwayat"
WORKSPACE_KEY = "vehicle_workspace"
NEXT_VIEW_KEY = "vehicle_workspace_next"


def open_view(view: str) -> None:
    # Apply the change before Streamlit recreates the navigation widget.
    st.session_state[NEXT_VIEW_KEY] = view
    st.rerun()


def workspace_tabs():
    if NEXT_VIEW_KEY in st.session_state:
        st.session_state[WORKSPACE_KEY] = st.session_state.pop(NEXT_VIEW_KEY)
    return st.tabs((SETUP, QUEUE, RESULTS, HISTORY), key=WORKSPACE_KEY, on_change="rerun")


def show_header() -> None:
    st.html("""
        <style>
        [data-testid="stMainBlockContainer"] {
            max-width: 1380px;
            padding-top: 3.3rem;
        }
        .st-key-vehicle_header {
            background: linear-gradient(110deg, #132b31, #14202b 74%);
            border: 1px solid #2c434d;
            border-left: 4px solid #44d4bb;
            border-radius: 1rem;
            padding: 1rem 1.5rem 1.2rem;
            color: #f0faf8;
        }
        .st-key-vehicle_header h1 {
            color: #f0faf8;
            font-family: Inter, system-ui, sans-serif;
            letter-spacing: -0.045em;
            font-size: clamp(2rem, 4vw, 3.1rem);
            line-height: 1.08;
            padding-bottom: .35rem;
        }
        .st-key-vehicle_header [data-testid="stCaptionContainer"] {
            color: #bed8d8;
        }
        .st-key-vehicle_workspace > [data-testid="stTabs"] > [role="tablist"],
        .st-key-vehicle_workspace > [data-baseweb="tab-list"] {
            margin-bottom: .45rem;
        }
        .st-key-vehicle_workspace [data-baseweb="tab"] {
            min-height: 44px;
        }
        .st-key-vehicle_empty_preview {
            min-height: 18rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: clamp(1rem, 5vw, 3.5rem);
            border-radius: .8rem;
            background: #13212a;
            color: #ebf2f7;
        }
        .st-key-vehicle_empty_preview h3 { color: #ebf2f7; }
        .st-key-vehicle_empty_preview p { color: #b8cad7; }
        @media (max-width: 760px) {
            [data-testid="stMainBlockContainer"] { padding: 3rem 1rem 2rem; }
            .st-key-vehicle_header { padding: .75rem 1rem 1rem; }
            .st-key-vehicle_empty_preview { min-height: 10rem; }
            .st-key-vehicle_setup > [data-testid="stHorizontalBlock"] {
                flex-direction: column;
            }
            .st-key-vehicle_setup > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                min-width: 0;
                width: 100%;
                flex: 1 1 auto;
            }
        }
        </style>
    """)
    with st.container(key="vehicle_header"):
        st.title("Vehicle Tracking", anchor=False)
        st.caption("Hitung kendaraan, periksa kondisi lalu lintas, dan lihat heatmap dari video.")
