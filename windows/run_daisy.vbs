Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' === Get parent directory of "windows" folder ===
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)
projectRoot = fso.GetParentFolderName(currentDir)
shell.CurrentDirectory = projectRoot

' === Run Streamlit silently from parent directory ===
shell.Run "cmd /c streamlit run dashboard\app.py --server.headless true", 0, False

' === Wait for server to start (3 seconds) ===
WScript.Sleep 3000

' === Launch Chrome in new window ===
chromePath = """C:\Program Files\Google\Chrome\Application\chrome.exe"""
url = "http://localhost:8501"
shell.Run chromePath & " --new-window " & url, 0, False

Set shell = Nothing
Set fso = Nothing
