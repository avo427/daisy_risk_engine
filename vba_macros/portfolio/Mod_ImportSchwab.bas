Attribute VB_Name = "Mod_ImportSchwab"
Function ImportSchwab() As Boolean
    On Error GoTo FailHandler

    ' === Config ===
    Dim folderPath As String
    folderPath = ThisWorkbook.Path & "\Data\Positions\Schwab\"

    Const portfolioSheetName As String = "Dashboard"
    Const namedRange As String = "Portfolio"

    ' === Declarations ===
    Dim fileName As String, csvPath As String
    Dim csvWS As Worksheet, tempWS As Worksheet
    Dim portfolioWS As Worksheet, portfolioRange As Range
    Dim fso As Object, folder As Object, file As Object
    Dim latestDate As Date, i As Long, j As Long
    Dim acctLabel As Variant, acctName As String
    Dim posTableStart As Long, posTableEnd As Long
    Dim tickerCol As Long, acctCol As Long, sharesCol As Long, costCol As Long
    Dim ticker As String, quantity As Variant, cost As Variant, cashVal As Variant
    Dim wsRow As Long, tickerCount As Long
    Dim excelTicker As String, excelAcct As String
    Dim cell As Range, fileDate As Date

    Application.ScreenUpdating = False
    Application.DisplayAlerts = False

    ' === Step 1: Find latest Schwab CSV ===
    Set fso = CreateObject("Scripting.FileSystemObject")
    If Not fso.FolderExists(folderPath) Then
        MsgBox "Folder not found: " & folderPath, vbCritical
        GoTo FailHandler
    End If
    
    Set folder = fso.GetFolder(folderPath)
    latestDate = DateSerial(1900, 1, 1)

    For Each file In folder.Files
        If file.Name Like "All-Accounts-Positions-*.csv" Then
            If file.DateLastModified > latestDate Then
                latestDate = file.DateLastModified
                fileName = file.Name
            End If
        End If
    Next file

    If fileName = "" Then GoTo FailHandler
    csvPath = folderPath & fileName

    ' === Step 2: Load CSV into temp sheet ===
    Set tempWS = ThisWorkbook.Sheets.Add(After:=Sheets(Sheets.Count))
    tempWS.Name = "TempSchwab_" & Format(Now, "hhmmss")

    With tempWS.QueryTables.Add(Connection:="TEXT;" & csvPath, Destination:=tempWS.Range("A1"))
        .TextFileParseType = xlDelimited
        .TextFileCommaDelimiter = True
        .Refresh BackgroundQuery:=False
    End With
    Set csvWS = tempWS

    ' === Step 3: Load Portfolio ===
    Set portfolioWS = ThisWorkbook.Sheets(portfolioSheetName)
    Set portfolioRange = ThisWorkbook.Names(namedRange).RefersToRange

    For Each cell In portfolioRange.Rows(1).Cells
        Select Case Trim(cell.Value)
            Case "Ticker": tickerCol = cell.Column
            Case "Account": acctCol = cell.Column
            Case "Shares": sharesCol = cell.Column
            Case "Unit Cost": costCol = cell.Column
        End Select
    Next cell

    ' === Step 4–5: Process Individual & Roth Tables ===
    For Each acctLabel In Array("Individual", "Roth")
        acctName = IIf(acctLabel = "Individual", "Schwab Brokerage", "Schwab Roth IRA")

        ' Find start
        posTableStart = 0
        For i = 1 To csvWS.Cells(Rows.Count, 1).End(xlUp).Row
            If LCase(Trim(csvWS.Cells(i, 1).Value)) Like LCase(acctLabel & "*") Then
                posTableStart = i + 1
                Exit For
            End If
        Next i
        If posTableStart = 0 Then GoTo NextTable

        ' Find end
        For i = posTableStart + 1 To csvWS.Cells(Rows.Count, 1).End(xlUp).Row
            If Trim(csvWS.Cells(i, 1).Value) = "Account Total" Then
                posTableEnd = i - 1
                Exit For
            End If
        Next i

        ' === Step 5: Loop through positions ===
        For i = posTableStart + 1 To posTableEnd
            ticker = NormalizeTicker(csvWS.Cells(i, 1).Value)
            quantity = csvWS.Cells(i, 3).Value
            cost = csvWS.Cells(i, 15).Value

            ' === Cash ===
            If ticker = "CASH & CASH INVESTMENTS" Then
                cashVal = csvWS.Cells(i, 7).Value
                For j = 2 To portfolioRange.Rows.Count
                    wsRow = portfolioRange.Cells(j, 1).Row
                    excelTicker = NormalizeTicker(portfolioWS.Cells(wsRow, tickerCol).Value)
                    excelAcct = Trim(UCase(portfolioWS.Cells(wsRow, acctCol).Value))

                    If excelTicker = "$" And excelAcct = Trim(UCase(acctName)) Then
                        portfolioWS.Cells(wsRow, sharesCol).Value = CleanNumber(cashVal)
                        portfolioWS.Cells(wsRow, costCol).Value = 1#
                        tickerCount = tickerCount + 1

                        Debug.Print "?? Updated CASH for " & acctName & ": " & cashVal
                    End If
                Next j

            ' === Ticker positions ===
            ElseIf ticker <> "" And quantity <> "" Then
                For j = 2 To portfolioRange.Rows.Count
                    wsRow = portfolioRange.Cells(j, 1).Row
                    excelTicker = NormalizeTicker(portfolioWS.Cells(wsRow, tickerCol).Value)
                    excelAcct = Trim(UCase(portfolioWS.Cells(wsRow, acctCol).Value))

                    If excelTicker = ticker And excelAcct = Trim(UCase(acctName)) Then
                        portfolioWS.Cells(wsRow, sharesCol).Value = CleanNumber(quantity)
                        portfolioWS.Cells(wsRow, costCol).Value = CleanNumber(cost)
                        tickerCount = tickerCount + 1

                        Debug.Print "? Updated " & ticker & ": " & quantity & " @ " & cost & " in " & acctName
                    End If
                Next j
            End If
        Next i
NextTable:
    Next acctLabel

    ' === Step 6: Set Date_Schwab ===
    Dim parts() As String
    parts = Split(fileName, "-")
    If UBound(parts) >= 5 Then
        Dim yyyy As Integer: yyyy = CInt(parts(3))
        Dim mm As Integer: mm = CInt(parts(4))
        Dim dd As Integer: dd = CInt(Split(parts(5), "-")(0))
        fileDate = DateSerial(yyyy, mm, dd)
        ThisWorkbook.Names("Date_Schwab").RefersToRange.Value = fileDate
    End If

    ' === Step 7: Final Log and Cleanup ===
    ThisWorkbook.Sheets("Dashboard").Activate
    Call AppendToLog(fileName, tickerCount, "Success", "Imported Schwab Holdings")
    MsgBox "Schwab update complete." & vbCrLf & _
           "File: " & fileName & vbCrLf & _
           "Tickers updated: " & tickerCount & vbCrLf & _
           "Date set: " & Format(fileDate, "mm/dd/yyyy"), vbInformation

    If Not tempWS Is Nothing Then tempWS.Delete
    Application.DisplayAlerts = True
    Application.ScreenUpdating = True
    ImportSchwab = True
    Exit Function

FailHandler:
    On Error Resume Next
    If Not tempWS Is Nothing Then tempWS.Delete
    Application.DisplayAlerts = True
    Application.ScreenUpdating = True
    MsgBox "Schwab import failed.", vbCritical
    ImportSchwab = False
End Function
