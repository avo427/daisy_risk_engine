Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' === Relative working directory ===
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = currentDir

' === Run Streamlit silently first ===
shell.Run "cmd /c streamlit run dashboard\app.py --server.headless true", 0, False

' === Wait for server to start (3 seconds) ===
WScript.Sleep 3000

' === Then launch Chrome in a new window ===
chromePath = """C:\Program Files\Google\Chrome\Application\chrome.exe"""
url = "http://localhost:8501"
shell.Run chromePath & " --new-window " & url, 0, False

Set shell = Nothing
