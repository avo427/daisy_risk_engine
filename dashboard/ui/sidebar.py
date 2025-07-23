import streamlit as st

def render_sidebar(config, save_config, load_all_data, project_root, paths, user_settings):
    st.header("Engine Controls")
    st.subheader("User Settings")

    years = st.number_input("Years of History", min_value=1, max_value=50, value=user_settings.get("years", 20))
    risk_free_rate = st.number_input("Risk-Free Rate", min_value=0.0, max_value=0.2,
                                     value=user_settings.get("risk_free_rate", 0.05), step=0.001)
    random_state = st.number_input("Random Seed", min_value=0, value=user_settings.get("random_state", 42), step=1)
    total_returns = st.checkbox("Use Total Returns", value=user_settings.get("total_returns", True))

    if st.button("Save Settings"):
        config["user_settings"]["years"] = years
        config["user_settings"]["total_returns"] = total_returns
        config["user_settings"]["risk_free_rate"] = float(risk_free_rate)
        config["user_settings"]["random_state"] = int(random_state)
        save_config(config)
        st.success("Settings saved to config.yaml")
        st.cache_data.clear()
        st.session_state.update(load_all_data())
        st.rerun()

    st.markdown("---")
    st.subheader("Pipeline Configuration")

    pipeline_option = st.radio(
        "Select the pipeline to run",
        ["Full Pipeline", "Risk Analysis", "Factor Exposure"],
        index=0
    )

    if st.button("Run Daisy Risk Engine"):
        import subprocess
        import sys
        mode_map = {
            "Full Pipeline": "full",
            "Risk Analysis": "risk",
            "Factor Exposure": "factor"
        }
        selected_mode = mode_map[pipeline_option]
        with st.spinner(f"Running {pipeline_option}... Please wait."):
            result = subprocess.run(
                [sys.executable, str(project_root / "main.py"), "--mode", selected_mode],
                capture_output=True,
                text=True
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