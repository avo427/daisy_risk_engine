Attribute VB_Name = "Mod_DashboardSnapshot"
Sub DashboardSnapshot()
    Dim wb As Workbook, tempWb As Workbook, snapWb As Workbook
    Dim dashboardWs As Worksheet, tempWS As Worksheet, snapWs As Worksheet
    Dim snapFile As String, todayStr As String
    Dim ole As OLEObject, shp As Shape
    Dim fullPathNote As String
    Dim rng As Range, cell As Range
    Dim dashboardName As String

    On Error GoTo CleanFail

    Set wb = ThisWorkbook
    todayStr = Format(Date, "yyyymmdd")

    ' === Use relative path to \Data\Portfolio Snapshots.xlsx
    snapFile = wb.Path & "\Data\Portfolio Snapshots.xlsx"
    fullPathNote = snapFile

    ' Capture the active dashboard sheet first
    Set dashboardWs = wb.ActiveSheet
    dashboardName = dashboardWs.Name

    Application.ScreenUpdating = False
    Application.DisplayAlerts = False

    ' Copy sheet to new temporary workbook
    wb.Sheets(dashboardName).Copy
    Set tempWb = ActiveWorkbook
    Set tempWS = tempWb.Sheets(1)

    ' Strip formulas
    Set rng = tempWS.UsedRange
    For Each cell In rng
        If cell.HasFormula Then cell.Formula = cell.Value
    Next cell
    rng.Value = rng.Value

    ' Remove conditional formatting
    rng.FormatConditions.Delete

    ' Delete ActiveX controls
    On Error Resume Next
    For Each ole In tempWS.OLEObjects
        ole.Delete
    Next ole

    ' Delete Form controls and shapes
    tempWS.Buttons.Delete
    For Each shp In tempWS.Shapes
        shp.Delete
    Next shp
    On Error GoTo 0

    ' Create or open snapshot workbook
    If Dir(snapFile) = "" Then
        Set snapWb = Workbooks.Add
        MkDir wb.Path & "\Data"
        snapWb.SaveAs fileName:=snapFile
    Else
        Set snapWb = Workbooks.Open(snapFile)
    End If

    ' Delete existing snapshot sheet if it exists
    On Error Resume Next
    snapWb.Sheets(todayStr).Delete
    On Error GoTo 0

    ' Move copied snapshot into target workbook
    tempWS.Copy Before:=snapWb.Sheets(1)
    Set snapWs = snapWb.Sheets(1)
    snapWs.Name = todayStr
    snapWs.Tab.ColorIndex = xlColorIndexNone

    tempWb.Close SaveChanges:=False
    snapWb.Save
    snapWb.Close

    ' Log success
    AppendToLog "Portfolio Snapshots.xlsx", 0, "Success", fullPathNote

    Application.ScreenUpdating = True
    Application.DisplayAlerts = True

    MsgBox "Snapshot '" & todayStr & "' saved successfully to Portfolio Snapshots.xlsx", vbInformation
    Exit Sub

CleanFail:
    On Error Resume Next
    AppendToLog "Portfolio Snapshots.xlsx", 0, "Failed", fullPathNote
    MsgBox "Snapshot failed. Please check file paths or workbook state.", vbExclamation
    Application.ScreenUpdating = True
    Application.DisplayAlerts = True
End Sub
