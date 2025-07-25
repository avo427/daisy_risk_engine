Attribute VB_Name = "Mod_UpdateTracker"
Sub UpdateTracker_Weekly()
    Dim snapshotWb As Workbook
    Dim snapshotPath As String
    Dim snapshotFile As String
    Dim wsTarget As Worksheet
    Dim tickers As Variant, marketVals As Variant
    Dim i As Long, j As Long, rowIdx As Long, colIdx As Long
    Dim sheetDate As Date
    Dim sheetName As String
    Dim allDates As Object ' Dictionary for unique dates
    Dim ws As Worksheet
    Dim tickerList As Object ' Dictionary for tickers with any non-zero MV
    Dim ticker As Variant
    
    On Error GoTo Cleanup ' Ensure cleanup on error
    
    Application.ScreenUpdating = False
    Application.DisplayAlerts = False
    
    ' === Setup paths and target sheet ===
    snapshotPath = ThisWorkbook.Path & "\..\data\"
    snapshotFile = "Portfolio Snapshots.xlsx"
    Set wsTarget = ThisWorkbook.Sheets("Tracker")
    
    ' Clear Tracker range (covers both tickers and dates)
    wsTarget.Range("Ticker_Range").ClearContents
    
    ' Open snapshot workbook read-only
    Set snapshotWb = Workbooks.Open(snapshotPath & snapshotFile, ReadOnly:=True)
    
    ' === Collect all valid last trading day sheets in the snapshot workbook ===
    Set allDates = CreateObject("Scripting.Dictionary")
    For Each ws In snapshotWb.Sheets
        sheetName = ws.Name
        ' Check if sheet name is 8 digits and numeric (yyyymmdd)
        If Len(sheetName) = 8 And IsNumeric(sheetName) Then
            sheetDate = DateSerial(CInt(Left(sheetName, 4)), CInt(Mid(sheetName, 5, 2)), CInt(Right(sheetName, 2)))
            ' Check if this date is a last trading day of the week (Friday or Thursday fallback)
            If IsLastTradingDayOfWeek(sheetDate) Then
                If Not allDates.Exists(sheetDate) Then
                    allDates.Add sheetDate, True
                End If
            End If
        End If
    Next ws
    
    ' Convert dictionary keys to array and sort ascending
    Dim sortedDates() As Date
    Dim keyArr() As Variant
    Dim n As Long
    keyArr = allDates.Keys
    n = UBound(keyArr) - LBound(keyArr) + 1
    ReDim sortedDates(1 To n)
    For i = 1 To n
        sortedDates(i) = keyArr(i - 1)
    Next i
    Call QuickSortDates(sortedDates, LBound(sortedDates), UBound(sortedDates))
    
    ' === Build a master list of tickers with non-zero market value on any date ===
    Set tickerList = CreateObject("Scripting.Dictionary")
    
    For i = 1 To UBound(sortedDates)
        sheetName = Format(sortedDates(i), "yyyymmdd")
        Set ws = snapshotWb.Sheets(sheetName)
        tickers = ws.Range("Portfolio_Tickers").Value
        marketVals = ws.Range("Portfolio_MarketValues").Value
        For j = 1 To UBound(tickers, 1)
            ticker = tickers(j, 1)
            If Not IsEmpty(ticker) Then
                If marketVals(j, 1) <> 0 And Not tickerList.Exists(ticker) Then
                    tickerList.Add ticker, True
                End If
            End If
        Next j
    Next i
    
    ' === Get market values for latest date ===
    Dim latestDateIndex As Long
    latestDateIndex = UBound(sortedDates)
    sheetName = Format(sortedDates(latestDateIndex), "yyyymmdd")
    Set ws = snapshotWb.Sheets(sheetName)
    tickers = ws.Range("Portfolio_Tickers").Value
    marketVals = ws.Range("Portfolio_MarketValues").Value
    
    ' Build dictionary of ticker to market value on latest date
    Dim latestDateValues As Object
    Set latestDateValues = CreateObject("Scripting.Dictionary")
    For j = 1 To UBound(tickers, 1)
        ticker = tickers(j, 1)
        If Not IsEmpty(ticker) Then
            If latestDateValues.Exists(ticker) Then
                latestDateValues(ticker) = latestDateValues(ticker) + marketVals(j, 1)
            Else
                latestDateValues.Add ticker, marketVals(j, 1)
            End If
        End If
    Next j
    
    ' === Build array for sorting (ticker, market value on latest date) ===
    Dim tickerArray() As Variant
    Dim idx As Long
    ReDim tickerArray(1 To tickerList.Count, 1 To 2)
    
    idx = 1
    For Each ticker In tickerList.Keys
        tickerArray(idx, 1) = ticker
        If latestDateValues.Exists(ticker) Then
            tickerArray(idx, 2) = latestDateValues(ticker)
        Else
            tickerArray(idx, 2) = 0
        End If
        idx = idx + 1
    Next ticker
    
    ' === Sort tickerArray descending by market value (column 2) ===
    Call Sort2DArrayByColumn(tickerArray, 2, False) ' False = descending
    
    ' === Write sorted tickers to Ticker_Range ===
    rowIdx = 1
    For idx = 1 To UBound(tickerArray)
        wsTarget.Range("Ticker_Range").Cells(rowIdx, 1).Value = tickerArray(idx, 1)
        rowIdx = rowIdx + 1
        If rowIdx > wsTarget.Range("Ticker_Range").Rows.Count Then Exit For
    Next idx
    
    ' === Write all dates to Date_Range header row ===
    colIdx = 1
    For i = 1 To UBound(sortedDates)
        wsTarget.Range("Date_Range").Cells(1, colIdx).Value = Format(sortedDates(i), "mm/dd/yyyy")
        colIdx = colIdx + 1
        If colIdx > wsTarget.Range("Date_Range").Columns.Count Then Exit For
    Next i
    
    ' === For each date, read tickers + values and write them aligned in the matrix ===
    Dim tickerRow As Long
    Dim tickerValueDict As Object
    
    For i = 1 To UBound(sortedDates)
        sheetName = Format(sortedDates(i), "yyyymmdd")
        Set ws = snapshotWb.Sheets(sheetName)
        tickers = ws.Range("Portfolio_Tickers").Value
        marketVals = ws.Range("Portfolio_MarketValues").Value
        
        Set tickerValueDict = CreateObject("Scripting.Dictionary")
        For j = 1 To UBound(tickers, 1)
            If Not IsEmpty(tickers(j, 1)) Then
                If tickerValueDict.Exists(tickers(j, 1)) Then
                    tickerValueDict(tickers(j, 1)) = tickerValueDict(tickers(j, 1)) + marketVals(j, 1)
                Else
                    tickerValueDict.Add tickers(j, 1), marketVals(j, 1)
                End If
            End If
        Next j
        
        colIdx = i
        For tickerRow = 1 To wsTarget.Range("Ticker_Range").Rows.Count
            ticker = wsTarget.Range("Ticker_Range").Cells(tickerRow, 1).Value
            If ticker = "" Then Exit For
            If tickerValueDict.Exists(ticker) Then
                wsTarget.Cells(wsTarget.Range("Ticker_Range").Row + tickerRow - 1, _
                    wsTarget.Range("Date_Range").Column + colIdx - 1).Value = tickerValueDict(ticker)
            Else
                wsTarget.Cells(wsTarget.Range("Ticker_Range").Row + tickerRow - 1, _
                    wsTarget.Range("Date_Range").Column + colIdx - 1).Value = 0
            End If
        Next tickerRow
    Next i
    
    ' === Close snapshot workbook ===
    snapshotWb.Close False
    
    ' === Log imports with date formatted as mm/dd/yyyy ===
    For i = 1 To UBound(sortedDates)
        Call AppendToLog(Format(sortedDates(i), "mm/dd/yyyy"), tickerList.Count, "Imported", "Success")
    Next i
    
    Application.StatusBar = "? Imported snapshots for " & UBound(sortedDates) & " weeks."
    
Cleanup:
    Application.ScreenUpdating = True
    Application.DisplayAlerts = True
    If Err.Number <> 0 Then MsgBox "Error " & Err.Number & ": " & Err.Description, vbCritical
End Sub
