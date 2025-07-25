Attribute VB_Name = "Mod_HelperFunctions"
Function CleanTicker(s As String) As String
    Dim exclPos As Long: exclPos = InStr(s, "!")
    If exclPos > 0 Then s = Left(s, exclPos - 1)
    CleanTicker = Trim(s)
End Function

Sub AppendToLog(fileName As String, tickerCount As Long, status As String, Optional note As String = "-")
    Dim logWS As Worksheet
    Dim nextRow As Long

    On Error Resume Next
    Set logWS = ThisWorkbook.Sheets("Hidden Log")
    If logWS Is Nothing Then
        Set logWS = ThisWorkbook.Sheets.Add(After:=Sheets(Sheets.Count))
        logWS.Name = "Hidden Log"
        logWS.Range("A1:E1").Value = Array("Timestamp", "Filename", "Tickers", "Status", "Notes")
        logWS.Visible = xlSheetVeryHidden
    End If
    On Error GoTo 0

    nextRow = logWS.Cells(logWS.Rows.Count, "A").End(xlUp).Row + 1
    logWS.Cells(nextRow, 1).Value = Now
    logWS.Cells(nextRow, 2).Value = fileName
    logWS.Cells(nextRow, 3).Value = tickerCount
    logWS.Cells(nextRow, 4).Value = status
    logWS.Cells(nextRow, 5).Value = note
    
End Sub

Function CleanNumber(val As Variant) As Double
    Dim s As String
    On Error GoTo Fail

    s = CStr(val)
    s = Replace(s, "$", "")
    s = Replace(s, ",", "")
    s = Replace(s, """", "")
    s = Trim(s)

    If IsNumeric(s) Then
        CleanNumber = CDbl(s)
    Else
        CleanNumber = 0
    End If
    Exit Function

Fail:
    CleanNumber = 0
End Function

Function NormalizeTicker(t As Variant) As String
    NormalizeTicker = Replace(UCase(Trim(CStr(t))), "/", "-")
End Function

Function SplitCSVLine(line As String) As String()
    Dim result() As String
    Dim current As String
    Dim inQuotes As Boolean
    Dim i As Long, ch As String
    ReDim result(0)

    current = ""
    inQuotes = False

    For i = 1 To Len(line)
        ch = Mid(line, i, 1)
        Select Case ch
            Case """"
                inQuotes = Not inQuotes
            Case ","
                If inQuotes Then
                    current = current & ch
                Else
                    result(UBound(result)) = current
                    ReDim Preserve result(UBound(result) + 1)
                    current = ""
                End If
            Case Else
                current = current & ch
        End Select
    Next i

    result(UBound(result)) = current
    SplitCSVLine = result
End Function
