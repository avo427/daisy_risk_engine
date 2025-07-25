Attribute VB_Name = "Mod_ImportPositions"
Sub ImportPositions()
    Dim successMerrill As Boolean, successSchwab As Boolean
    Dim ws As Worksheet, portfolioRange As Range
    Dim contractsCol As Long, sharesCol As Long, costCol As Long
    Dim cell As Range, i As Long

    ' === Setup ===
    Set ws = ThisWorkbook.Sheets("Dashboard")
    Set portfolioRange = ws.Range("Portfolio")

    ' === Identify Contracts, Shares, and Unit Cost Columns ===
    For Each cell In portfolioRange.Rows(1).Cells
        Select Case UCase(Trim(cell.Value))
            Case "CONTRACTS": contractsCol = cell.Column
            Case "SHARES": sharesCol = cell.Column
            Case "UNIT COST": costCol = cell.Column
        End Select
    Next cell

    ' === Sanity check ===
    If contractsCol = 0 And sharesCol = 0 And costCol = 0 Then
        MsgBox "No matching column headers found in Portfolio range.", vbCritical
        Exit Sub
    End If

    ' === Disable UI updates for performance ===
    Application.ScreenUpdating = False
    Application.DisplayAlerts = False

    ' === Clear values BEFORE imports ===
    For i = 2 To portfolioRange.Rows.Count
        If contractsCol > 0 Then ws.Cells(portfolioRange.Cells(i, 1).Row, contractsCol).Value = ""
        If sharesCol > 0 Then ws.Cells(portfolioRange.Cells(i, 1).Row, sharesCol).Value = ""
        If costCol > 0 Then ws.Cells(portfolioRange.Cells(i, 1).Row, costCol).Value = ""
    Next i

    ' === Execute imports ===
    successMerrill = ImportMerrill()
    successSchwab = ImportSchwab()

    ' === Check and fallback ===
    If successMerrill And successSchwab Then
        MsgBox "Portfolio updated successfully from both Merrill Edge and Schwab.", vbInformation, "Import Complete"
    Else
        For i = 2 To portfolioRange.Rows.Count
            If sharesCol > 0 Then ws.Cells(portfolioRange.Cells(i, 1).Row, sharesCol).Value = ""
            If costCol > 0 Then ws.Cells(portfolioRange.Cells(i, 1).Row, costCol).Value = ""
            If contractsCol > 0 Then ws.Cells(portfolioRange.Cells(i, 1).Row, contractsCol).Value = ""
        Next i

        MsgBox "Import failed. Shares and Unit Cost cleared to prevent partial update." & vbCrLf & _
               IIf(Not successMerrill, "- Merrill Edge failed" & vbCrLf, "") & _
               IIf(Not successSchwab, "- Schwab failed", ""), vbCritical, "Import Error"
    End If

    ' === Final UI Reset ===
    Application.DisplayAlerts = True
    Application.ScreenUpdating = True
    ws.Activate
End Sub

