Attribute VB_Name = "Mod_ImportMarketData"
Sub ImportMarketData()
    ' === Constants ===
    Const subfolderPath As String = "\Data\Market\"
    Const targetSheet As String = "Market Data"
    Const namedRange As String = "Date_MarketData"

    ' === Declarations ===
    Dim folderPath As String
    Dim latestFile As String
    Dim latestDate As Date
    Dim fso As Object, folder As Object, file As Object
    Dim wbCSV As Workbook, wsCSV As Worksheet, wsTarget As Worksheet
    Dim lastRow As Long, lastCol As Long
    Dim dataRange As Range
    Dim tickerCount As Long, metricsCount As Long
    Dim wasScreenUpdating As Boolean
    Dim note As String

    On Error GoTo HandleError
    wasScreenUpdating = Application.ScreenUpdating
    Application.ScreenUpdating = False
    Application.DisplayAlerts = False

    ' === Construct relative path from workbook folder ===
    folderPath = ThisWorkbook.Path & subfolderPath

    ' === Validate folder exists ===
    Set fso = CreateObject("Scripting.FileSystemObject")
    If Not fso.FolderExists(folderPath) Then
        note = "Folder not found: " & folderPath
        AppendToLog "(none)", 0, "Failed", note
        MsgBox note, vbCritical, "Import Failed"
        GoTo Cleanup
    End If

    Set folder = fso.GetFolder(folderPath)

    ' === Find latest .csv file by DateLastModified ===
    latestDate = #1/1/1900#
    latestFile = ""

    For Each file In folder.Files
        If LCase(Right(file.Name, 4)) = ".csv" Then
            If file.DateLastModified > latestDate Then
                latestDate = file.DateLastModified
                latestFile = file.Name
            End If
        End If
    Next file

    If latestFile = "" Then
        note = "No CSV files found in folder: " & folderPath
        AppendToLog "(none)", 0, "Failed", note
        MsgBox note, vbExclamation, "Import Failed"
        GoTo Cleanup
    End If

    ' === Open CSV file (hidden) ===
    Set wbCSV = Workbooks.Open(folderPath & latestFile, ReadOnly:=True)
    wbCSV.Windows(1).Visible = False
    Set wsCSV = wbCSV.Sheets(1)

    ' === Delete metadata rows ===
    If wsCSV.UsedRange.Rows.Count >= 3 Then
        wsCSV.Rows("1:3").Delete Shift:=xlUp
    End If

    ' === Parse as comma-separated values ===
    With wsCSV
        .Columns("A").TextToColumns _
            Destination:=.Range("A1"), _
            DataType:=xlDelimited, _
            TextQualifier:=xlTextQualifierDoubleQuote, _
            ConsecutiveDelimiter:=False, _
            Tab:=False, _
            Semicolon:=False, _
            Comma:=True
    End With

    ' === Identify full data range ===
    lastRow = wsCSV.Cells(wsCSV.Rows.Count, 1).End(xlUp).Row
    lastCol = wsCSV.Cells(1, wsCSV.Columns.Count).End(xlToLeft).Column
    Set dataRange = wsCSV.Range(wsCSV.Cells(1, 1), wsCSV.Cells(lastRow, lastCol))

    ' === Paste to target sheet ===
    Set wsTarget = ThisWorkbook.Sheets(targetSheet)
    wsTarget.Cells.ClearContents
    wsTarget.Range("A1").Resize(dataRange.Rows.Count, dataRange.Columns.Count).Value = dataRange.Value

    ' === Count stats ===
    tickerCount = Application.WorksheetFunction.Max(0, Application.WorksheetFunction.CountA(wsTarget.Columns(1)) - 1)
    metricsCount = Application.WorksheetFunction.Max(0, dataRange.Columns.Count - 1)

    ' === Update named range ===
    ThisWorkbook.Names(namedRange).RefersToRange.Value = Format(latestDate, "mm/dd/yyyy")

    ' === Close source file ===
    wbCSV.Close SaveChanges:=False

    ' === Log and final message ===
    AppendToLog latestFile, tickerCount, "Success", folderPath & latestFile

    MsgBox "Market Data update complete." & vbCrLf & _
           "File: " & latestFile & vbCrLf & _
           "Tickers updated: " & tickerCount & vbCrLf & _
           "Date set: " & Format(latestDate, "mm/dd/yyyy"), _
           vbInformation, "Import Complete"

    GoTo Cleanup

HandleError:
    note = "Unexpected error: " & Err.Description
    AppendToLog latestFile, 0, "Failed", note
    MsgBox "Error: " & note, vbCritical, "Import Error"
    On Error Resume Next
    If Not wbCSV Is Nothing Then wbCSV.Close SaveChanges:=False

Cleanup:
    Application.DisplayAlerts = True
    Application.ScreenUpdating = wasScreenUpdating
End Sub
