Attribute VB_Name = "Mod_HelperFunctions"
Function IsNasdaqTradingDayYahoo(d As Date, Optional ticker As String = "^NDX") As Boolean
    Dim http As Object
    Dim url As String, jsonText As String
    Dim epoch As Long

    epoch = (d - DateSerial(1970, 1, 1)) * 86400
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" & ticker & _
          "?interval=1d&period1=" & epoch & "&period2=" & (epoch + 86400)

    Set http = CreateObject("MSXML2.XMLHTTP")
    On Error GoTo FailSafe
    http.Open "GET", url, False
    http.Send

    If http.status = 200 Then
        jsonText = http.ResponseText
        If InStr(jsonText, "timestamp") > 0 Then
            IsNasdaqTradingDayYahoo = True
        Else
            IsNasdaqTradingDayYahoo = False
        End If
    Else
        IsNasdaqTradingDayYahoo = False
    End If
    Exit Function

FailSafe:
    IsNasdaqTradingDayYahoo = False
End Function

Function IsLastTradingDayOfWeek(d As Date) As Boolean
    Dim wDay As Integer
    wDay = Weekday(d, vbMonday) ' Monday=1, Sunday=7
    If wDay = 5 Then ' Friday
        IsLastTradingDayOfWeek = True
        Exit Function
    End If
    If wDay = 4 Then ' Thursday fallback if Friday not trading
        Dim nextDay As Date
        nextDay = d + 1
        If Not IsNasdaqTradingDayYahoo(nextDay) Then
            IsLastTradingDayOfWeek = True
            Exit Function
        End If
    End If
    IsLastTradingDayOfWeek = False
End Function

Sub QuickSortDates(arr() As Date, first As Long, last As Long)
    Dim pivot As Date, tmp As Date
    Dim i As Long, j As Long
    If first >= last Then Exit Sub
    pivot = arr((first + last) \ 2)
    i = first
    j = last
    Do While i <= j
        Do While arr(i) < pivot
            i = i + 1
        Loop
        Do While arr(j) > pivot
            j = j - 1
        Loop
        If i <= j Then
            tmp = arr(i)
            arr(i) = arr(j)
            arr(j) = tmp
            i = i + 1
            j = j - 1
        End If
    Loop
    If first < j Then QuickSortDates arr, first, j
    If i < last Then QuickSortDates arr, i, last
End Sub

Sub Sort2DArrayByColumn(ByRef arr As Variant, ByVal col As Long, ByVal ascending As Boolean)
    Dim i As Long, j As Long
    Dim temp1 As Variant, temp2 As Variant
    Dim swapped As Boolean
    Dim n As Long
    n = UBound(arr, 1)
    For i = 1 To n - 1
        swapped = False
        For j = 1 To n - i
            If ascending Then
                If arr(j, col) > arr(j + 1, col) Then
                    temp1 = arr(j, 1)
                    temp2 = arr(j, 2)
                    arr(j, 1) = arr(j + 1, 1)
                    arr(j, 2) = arr(j + 1, 2)
                    arr(j + 1, 1) = temp1
                    arr(j + 1, 2) = temp2
                    swapped = True
                End If
            Else
                If arr(j, col) < arr(j + 1, col) Then
                    temp1 = arr(j, 1)
                    temp2 = arr(j, 2)
                    arr(j, 1) = arr(j + 1, 1)
                    arr(j, 2) = arr(j + 1, 2)
                    arr(j + 1, 1) = temp1
                    arr(j + 1, 2) = temp2
                    swapped = True
                End If
            End If
        Next j
        If Not swapped Then Exit For
    Next i
End Sub


Sub AppendToLog(sheetDate As String, tickerCount As Long, status As String, Optional note As String = "-")
    Dim logWS As Worksheet
    Dim nextRow As Long

    On Error Resume Next
    Set logWS = ThisWorkbook.Sheets("Hidden Log")
    If logWS Is Nothing Then
        Set logWS = ThisWorkbook.Sheets.Add(After:=Sheets(Sheets.Count))
        logWS.Name = "Hidden Log"
        logWS.Range("A1:E1").Value = Array("Timestamp", "Sheet Date", "Tickers", "Status", "Notes")
        logWS.Visible = xlSheetVeryHidden
    End If
    On Error GoTo 0

    nextRow = logWS.Cells(logWS.Rows.Count, "A").End(xlUp).Row + 1
    logWS.Cells(nextRow, 1).Value = Now
    logWS.Cells(nextRow, 2).Value = sheetDate
    logWS.Cells(nextRow, 3).Value = tickerCount
    logWS.Cells(nextRow, 4).Value = status
    logWS.Cells(nextRow, 5).Value = note
End Sub

