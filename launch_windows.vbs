Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Get the directory where this VBS file is located
strScriptPath = objFSO.GetParentFolderName(WScript.ScriptFullName)

' Change to the script directory
objShell.CurrentDirectory = strScriptPath

' Launch streamlit app silently (py is the Windows Python launcher)
objShell.Run "py -m streamlit run dashboard/app.py", 0, False

Set objShell = Nothing
Set objFSO = Nothing 