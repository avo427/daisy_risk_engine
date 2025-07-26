import streamlit as st

def render_sidebar(config, save_config, load_all_data, project_root, paths, user_settings):
    st.header("Engine Controls")
    st.subheader("User Settings")

    years = st.number_input("Years of History", min_value=1, max_value=50, value=user_settings.get("years", 20))
    risk_free_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0,
                                     value=user_settings.get("risk_free_rate", 0.05) * 100, step=0.01, format="%.2f")
    random_state = st.number_input("Random Seed", min_value=0, value=user_settings.get("random_state", 42), step=1)
    monte_carlo_simulations = st.number_input("Monte Carlo Simulations", min_value=1000, max_value=100000, 
                                             value=user_settings.get("monte_carlo_simulations", 10000), step=1000)
    target_volatility = st.number_input("Target Volatility (%)", min_value=5.0, max_value=50.0,
                                       value=user_settings.get("volatility_sizing", {}).get("target_volatility", 0.20) * 100, step=0.01, format="%.2f",
                                       help="Target volatility for volatility-based sizing")
    total_returns = st.checkbox("Use Total Returns", value=user_settings.get("total_returns", True))

    col1, spacer, col2 = st.columns([1, 0.1, 1])
    
    with col1:
        if st.button("Save", use_container_width=True):
            config["user_settings"]["years"] = years
            config["user_settings"]["total_returns"] = total_returns
            config["user_settings"]["risk_free_rate"] = float(risk_free_rate) / 100
            config["user_settings"]["random_state"] = int(random_state)
            config["user_settings"]["monte_carlo_simulations"] = int(monte_carlo_simulations)
            
            # Ensure volatility_sizing section exists
            if "volatility_sizing" not in config["user_settings"]:
                config["user_settings"]["volatility_sizing"] = {}
            config["user_settings"]["volatility_sizing"]["target_volatility"] = float(target_volatility) / 100
            
            save_config(config)
            st.success("Settings saved to config.yaml")
            st.cache_data.clear()
            st.session_state.update(load_all_data())
            st.rerun()
    
    with spacer:
        st.write("")  # Empty space for gap
    
    with col2:
        if st.button("Default", use_container_width=True):
            config["user_settings"]["years"] = 10
            config["user_settings"]["total_returns"] = True
            config["user_settings"]["risk_free_rate"] = 0.05
            config["user_settings"]["random_state"] = 44
            config["user_settings"]["monte_carlo_simulations"] = 10000
            
            # Ensure volatility_sizing section exists
            if "volatility_sizing" not in config["user_settings"]:
                config["user_settings"]["volatility_sizing"] = {}
            config["user_settings"]["volatility_sizing"]["target_volatility"] = 0.20
            
            save_config(config)
            st.success("Settings reset to defaults")
            st.cache_data.clear()
            st.session_state.update(load_all_data())
            st.rerun()

    st.markdown("---")
    st.subheader("Pipeline Configuration")

    pipeline_option = st.radio(
        "Select the pipeline to run",
        ["Full Pipeline", "Risk Analysis", "Factor Exposure", "Stress Testing"],
        index=0,
        help="Full Pipeline: Complete analysis (realized + forecast + factor exposure + stress testing). Risk Analysis: Realized and forecast metrics only. Factor Exposure: Factor returns and portfolio exposures. Stress Testing: Factor exposure + comprehensive stress tests."
    )

    if st.button("Run Daisy Risk Engine"):
        import subprocess
        import sys
        mode_map = {
            "Full Pipeline": "full",
            "Risk Analysis": "risk",
            "Factor Exposure": "factor",
            "Stress Testing": "stress"
        }
        selected_mode = mode_map[pipeline_option]
        with st.spinner(f"Running {pipeline_option}... Please wait."):
            result = subprocess.run(
                [sys.executable, str(project_root / "main.py"), "--mode", selected_mode],
                capture_output=True,
                text=True,
                cwd=str(project_root)  # Set working directory to project root
            )
        if result.returncode == 0:
            st.success("SUCCESS: Run completed successfully.")
            st.cache_data.clear()
            st.session_state.update(load_all_data())
            st.rerun()
        else:
            st.error("ERROR: Pipeline run failed. See logs below.")
            st.code(result.stdout + "\n" + result.stderr, language="bash")

    if st.button("Reload Data Only"):
        st.cache_data.clear()
        st.session_state.update(load_all_data())
        st.rerun()

    st.markdown("---") 