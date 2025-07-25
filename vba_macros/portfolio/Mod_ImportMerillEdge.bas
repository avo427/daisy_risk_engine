Attribute VB_Name = "Mod_ImportMerillEdge"
Function ImportMerrill() As Boolean
    On Error GoTo FailHandler

    Const overrideSGOVCost As Double = 100
    Dim folderPath As String
    folderPath = ThisWorkbook.Path & "\Data\Positions\Merrill\"

    Dim latestFile As String, latestDate As Date, fileDate As Date
    Dim fso As Object, folder As Object, file As Object
    Set fso = CreateObject("Scripting.FileSystemObject")

    If Not fso.FolderExists(folderPath) Then
        MsgBox "Folder not found: " & folderPath, vbCritical
        GoTo FailHandler
    End If

    Set folder = fso.GetFolder(folderPath)

    For Each file In folder.Files
        If file.Name Like "ExportData*.csv" Then
            If file.DateLastModified > latestDate Then
                latestDate = file.DateLastModified
                latestFile = file.Path
                fileDate = file.DateLastModified
            End If
        End If
    Next file

    If latestFile = "" Then GoTo FailHandler

    ' === Parse Ticker Table ===
    Dim tickerMap As Object: Set tickerMap = CreateObject("Scripting.Dictionary")
    Dim tickerColCSV As Long: tickerColCSV = -1
    Dim qtyColCSV As Long: qtyColCSV = -1
    Dim costColCSV As Long: costColCSV = -1

    Dim parser As Object: Set parser = fso.OpenTextFile(latestFile, 1)
    Dim foundHeader As Boolean: foundHeader = False

    Do While Not parser.AtEndOfStream
        Dim fields() As String
        fields = SplitCSVLine(parser.ReadLine)

        If Not foundHeader Then
            Dim i As Long
            For i = LBound(fields) To UBound(fields)
                Select Case UCase(Trim(fields(i)))
                    Case "SYMBOL": tickerColCSV = i
                    Case "QUANTITY": qtyColCSV = i
                    Case "UNIT COST": costColCSV = i
                End Select
            Next i
            If tickerColCSV >= 0 And qtyColCSV >= 0 And costColCSV >= 0 Then
                foundHeader = True
            End If
        Else
            If UBound(fields) >= costColCSV Then
                Dim rawTicker As String: rawTicker = UCase(Trim(CleanTicker(fields(tickerColCSV))))
                Dim qty As Double: qty = CleanNumber(fields(qtyColCSV))
                Dim cost As Double: cost = CleanNumber(fields(costColCSV))

                If rawTicker <> "" And qty > 0 And (cost > 0 Or rawTicker = "SGOV") Then
                    If rawTicker = "SGOV" Then cost = overrideSGOVCost
                    If Not tickerMap.exists(rawTicker) Then
                        tickerMap.Add rawTicker, Array(qty, cost)
                    End If
                End If
            End If
        End If
    Loop
    parser.Close

    If tickerColCSV = -1 Or qtyColCSV = -1 Or costColCSV = -1 Then GoTo FailHandler

    ' === Parse Balances ===
    Dim moneyAccounts As Double, pendingActivity As Double
    Dim cashBalance As Double, marginBalance As Double
    Dim balParser As Object: Set balParser = fso.OpenTextFile(latestFile, 1)

    Do While Not balParser.AtEndOfStream
        Dim balLine As String: balLine = balParser.ReadLine
        Dim parts() As String

        If InStr(UCase(balLine), "MONEY ACCOUNT") > 0 Then
            parts = SplitCSVLine(balLine)
            If UBound(parts) >= 4 Then moneyAccounts = moneyAccounts + CleanNumber(parts(4))
        ElseIf InStr(UCase(balLine), "PENDING ACTIVITY") > 0 Then
            parts = SplitCSVLine(balLine)
            If UBound(parts) >= 4 Then pendingActivity = pendingActivity + CleanNumber(parts(4))
        ElseIf InStr(UCase(balLine), "CASH BALANCE") > 0 Then
            parts = SplitCSVLine(balLine)
            If UBound(parts) >= 4 Then cashBalance = cashBalance + CleanNumber(parts(4))
        ElseIf InStr(UCase(balLine), "MARGIN BALANCE") > 0 Then
            parts = SplitCSVLine(balLine)
            If UBound(parts) >= 4 Then marginBalance = marginBalance + CleanNumber(parts(4))
        End If
    Loop
    balParser.Close

    Dim totalCash As Double: totalCash = moneyAccounts + pendingActivity + cashBalance + marginBalance
    If Abs(totalCash) > 0.01 Then tickerMap("$") = Array(totalCash, 1#)

    ' === Update Dashboard ===
    Dim wsPortfolio As Worksheet: Set wsPortfolio = ThisWorkbook.Sheets("Dashboard")
    Dim portfolioRange As Range: Set portfolioRange = wsPortfolio.Range("Portfolio")

    Dim headerRow As Range: Set headerRow = portfolioRange.Rows(1)
    Dim tickerCol As Long, sharesCol As Long, costCol As Long, accountCol As Long
    Dim cell As Range

    For Each cell In headerRow.Cells
        Select Case UCase(Trim(cell.Value))
            Case "TICKER": tickerCol = cell.Column
            Case "SHARES": sharesCol = cell.Column
            Case "UNIT COST": costCol = cell.Column
            Case "ACCOUNT": accountCol = cell.Column
        End Select
    Next cell

    If tickerCol = 0 Or sharesCol = 0 Or costCol = 0 Or accountCol = 0 Then
        MsgBox "Missing column headers in Portfolio range.", vbCritical
        GoTo FailHandler
    End If

    Dim updatedCount As Long, r As Long
    Dim rowTicker As String, rowAccount As String

    For r = 2 To portfolioRange.Rows.Count
        rowTicker = UCase(Trim(CleanTicker(wsPortfolio.Cells(r, tickerCol).Value)))
        rowAccount = UCase(Trim(wsPortfolio.Cells(r, accountCol).Value))

        If rowAccount = "MERRILL EDGE" And tickerMap.exists(rowTicker) Then
            Dim dataArr As Variant: dataArr = tickerMap(rowTicker)
            wsPortfolio.Cells(r, sharesCol).Value = dataArr(0)
            wsPortfolio.Cells(r, costCol).Value = dataArr(1)
            updatedCount = updatedCount + 1

            Debug.Print "? Updated " & rowTicker & ": " & dataArr(0) & " shares @ " & dataArr(1)
        End If
    Next r

    ' === Update Date Cell ===
    Dim dateCell As Range: Set dateCell = Range("Date_MerrillEdge")
    If Not dateCell Is Nothing Then dateCell.Value = Format(fileDate, "mm/dd/yyyy")

    ThisWorkbook.Sheets("Dashboard").Activate
    Call AppendToLog(latestFile, updatedCount, "Success", "Imported Merrill Holdings")

    MsgBox "Merrill Edge update complete." & vbCrLf & _
           "File: " & Dir(latestFile) & vbCrLf & _
           "Tickers updated: " & updatedCount & vbCrLf & _
           "Date set: " & Format(fileDate, "mm/dd/yyyy"), vbInformation

    ImportMerrill = True
    Exit Function

FailHandler:
    ImportMerrill = False
    MsgBox "Merrill import failed. Check format, headers, or file presence.", vbCritical
End Function

