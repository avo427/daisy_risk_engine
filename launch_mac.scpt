do shell script "cd /Users/andrewvo/Desktop/INVESTMENTS/daisy_risk_engine && nohup streamlit run dashboard/app.py --server.port 8501 > /tmp/streamlit.log 2>&1 &"
delay 5
tell application "Google Chrome"
    open location "http://localhost:8501"
end tell 