Set shell = CreateObject("WScript.Shell")

' === Navigate to parent directory ===
parentFolder = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
projectRoot = CreateObject("Scripting.FileSystemObject").GetParentFolderName(parentFolder)

' === Build command ===
cmd = "cmd /k """ & projectRoot & "\.venv\Scripts\activate.bat && streamlit run " & projectRoot & "\dashboard\app.py"""

' === Run the command in new terminal window ===
shell.Run cmd, 1, False
