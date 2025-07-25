Attribute VB_Name = "Mod_UpdateExposures"
Sub UpdateExposures()
    Dim pt As PivotTable
    Dim ws As Worksheet
    Dim tblRange As Range
    Dim startRow As Long, endRow As Long
    Dim i As Long
    Dim rowField As PivotField
    Dim cell As Range
    Dim pCell As PivotCell

    Set ws = ActiveSheet
    If ws.PivotTables.Count = 0 Then Exit Sub

    Application.ScreenUpdating = False

    ' === Step 1: Refresh and sort by "Market Value $" ===
    For Each pt In ws.PivotTables
        pt.PreserveFormatting = True
        pt.RefreshTable

        On Error Resume Next
        If pt.PivotFields.Count > 0 And pt.RowFields.Count > 0 Then
            Set rowField = pt.RowFields(1)
            If Not rowField Is Nothing Then
                rowField.AutoSort xlDescending, "Market Value $"
            End If
        End If
        On Error GoTo 0
    Next pt

    ' === Step 2: Format each PivotTable ===
    For Each pt In ws.PivotTables
        Set tblRange = pt.TableRange1
        If tblRange Is Nothing Then GoTo NextPT

        ' Center-align all content
        tblRange.HorizontalAlignment = xlCenter

        ' Format first column: left-align, and bold only top-level PivotItems
        With tblRange.Columns(1)
            .HorizontalAlignment = xlLeft
            For Each cell In .Cells
                On Error Resume Next
                Set pCell = cell.PivotCell
                On Error GoTo 0
                If Not pCell Is Nothing Then
                    If pCell.PivotCellType = xlPivotCellPivotItem And pCell.PivotField.Position = 1 Then
                        cell.Font.Bold = True
                    Else
                        cell.Font.Bold = False
                    End If
                Else
                    cell.Font.Bold = False
                End If
            Next cell
        End With

        ' Clear all borders
        tblRange.Borders.LineStyle = xlNone

        ' Thin horizontal lines
        With tblRange.Borders(xlInsideHorizontal)
            .LineStyle = xlContinuous
            .Weight = xlThin
        End With

        ' Remove vertical borders
        tblRange.Borders(xlInsideVertical).LineStyle = xlNone

        ' Thick outer borders
        With tblRange
            .Borders(xlEdgeLeft).LineStyle = xlContinuous: .Borders(xlEdgeLeft).Weight = xlMedium
            .Borders(xlEdgeRight).LineStyle = xlContinuous: .Borders(xlEdgeRight).Weight = xlMedium
            .Borders(xlEdgeTop).LineStyle = xlContinuous: .Borders(xlEdgeTop).Weight = xlMedium
            .Borders(xlEdgeBottom).LineStyle = xlContinuous: .Borders(xlEdgeBottom).Weight = xlMedium
        End With

        ' Enforce row height = 20pt
        startRow = tblRange.Row
        endRow = startRow + tblRange.Rows.Count - 1
        For i = startRow To endRow
            ws.Rows(i).RowHeight = 20
        Next i

NextPT:
    Next pt

    Application.ScreenUpdating = True
End Sub

