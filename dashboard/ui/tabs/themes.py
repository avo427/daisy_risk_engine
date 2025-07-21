import streamlit as st
import pandas as pd

def themes_tab(config, save_config):
    st.subheader("Theme Proxies")
    theme_proxy_df = pd.DataFrame(list(config.get("theme_proxies", {}).items()), columns=["Theme", "ETF Proxy"])
    theme_proxy_height = 35 * (len(theme_proxy_df) + 2)
    theme_proxy_edit = st.data_editor(
        theme_proxy_df,
        num_rows="dynamic",
        use_container_width=True,
        height=theme_proxy_height
    )
    st.subheader("Ticker-to-Theme Map")
    ticker_theme_df = pd.DataFrame(list(config.get("ticker_themes", {}).items()), columns=["Ticker", "Theme"])
    ticker_theme_height = 35 * (len(ticker_theme_df) + 2)
    ticker_theme_edit = st.data_editor(
        ticker_theme_df,
        num_rows="dynamic",
        use_container_width=True,
        height=ticker_theme_height
    )
    if st.button("Save Theme & Ticker Mappings"):
        config["theme_proxies"] = dict(zip(theme_proxy_edit["Theme"], theme_proxy_edit["ETF Proxy"]))
        config["ticker_themes"] = dict(zip(ticker_theme_edit["Ticker"], ticker_theme_edit["Theme"]))
        save_config(config)
        st.success("Theme and ticker mappings saved.")
        st.rerun() 