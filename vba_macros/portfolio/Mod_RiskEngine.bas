Attribute VB_Name = "Mod_RiskEngine"
Sub RunRiskEngine()
    Dim osString As String
    Dim parentPath As String
    Dim appPath As String
    Dim scriptPath As String

    parentPath = ThisWorkbook.Path
    osString = Trim(LCase(Application.OperatingSystem))

    If InStr(osString, "mac") > 0 Then
        ' macOS: Launch .command file
        appPath = parentPath & "/daisy_risk_engine/launch_mac.command"
        If Dir(appPath) = "" Then
            MsgBox "Mac launcher not found:" & vbCrLf & appPath, vbCritical, "File Missing"
            Exit Sub
        End If
        shell "open " & Chr(34) & appPath & Chr(34), vbNormalFocus

    ElseIf InStr(osString, "windows") > 0 Then
        ' Windows: Run VBS script silently
        scriptPath = parentPath & "\daisy_risk_engine\launch_windows.vbs"
        If Dir(scriptPath) = "" Then
            MsgBox "Windows launcher not found:" & vbCrLf & scriptPath, vbCritical, "File Missing"
            Exit Sub
        End If
        shell "wscript """ & scriptPath & """", vbHide

    Else
        MsgBox "Unsupported OS: " & Application.OperatingSystem, vbCritical, "Not Supported"
    End If
End Sub
