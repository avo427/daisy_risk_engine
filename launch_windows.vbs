Set shell = CreateObject("WScript.Shell")

' Get the directory where this script is located
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Build and run the streamlit command
cmd = "cmd /c ""cd /d """ & scriptDir & """ && streamlit run dashboard\app.py --server.port 8501"""

' Run the command in a new window
shell.Run cmd, 1, False 