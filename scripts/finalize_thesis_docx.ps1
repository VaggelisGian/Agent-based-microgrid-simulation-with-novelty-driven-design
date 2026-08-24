<#
Finalize the built thesis .docx with Microsoft Word: update every field (TOC, lists of
figures and tables, STYLEREF/SEQ caption numbers, PAGE), repaginate, record page counts
per top-level heading, save in place, and write a JSON page report next to the file.

Usage:
    pwsh -File scripts/finalize_thesis_docx.ps1 -Path docs/thesis/submission/thesis.docx

Page counting: the body is counted from the first numbered chapter heading to the
paragraph before the bibliography heading, which is the regulation's "text excluding
appendices" (the bibliography is listed separately so the reader can apply either reading).
#>
param(
    [Parameter(Mandatory = $true)][string]$Path
)

$ErrorActionPreference = "Stop"
$full = (Resolve-Path $Path).Path
$report = [System.IO.Path]::ChangeExtension($full, ".pages.json")

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($full, $false, $false)
    # update twice: page-dependent fields (TOC page numbers) settle on the second pass
    for ($pass = 0; $pass -lt 2; $pass++) {
        $null = $doc.Fields.Update()
        foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
        foreach ($tof in $doc.TablesOfFigures) { $tof.Update() }
        $doc.Repaginate()
    }
    $total = $doc.ComputeStatistics(2)  # wdStatisticPages
    $headings = @()
    foreach ($p in $doc.Paragraphs) {
        $style = $p.Style.NameLocal
        if ($style -eq "Heading 1" -or $style -eq "Επικεφαλίδα 1" -or $style -eq "Appendix") {
            $text = $p.Range.Text.Trim()
            if ($text.Length -eq 0) { continue }
            $pg = $p.Range.Information(3)          # wdActiveEndPageNumber: physical sheet
            $printed = $p.Range.Information(1)     # wdActiveEndAdjustedPageNumber: printed folio
            $listString = ""
            try { $listString = $p.Range.ListFormat.ListString } catch { $listString = "" }
            $headings += [pscustomobject]@{ heading = $text; list = $listString; page = $pg; printed_page = $printed; style = $style }
        }
    }
    # body = first numbered chapter to the page before the bibliography, the regulation's
    # "text excluding appendices" with the bibliography excluded as the supervisor requires
    $firstChapter = $headings | Where-Object { $_.list -match "^\s*Κεφάλαιο" } | Select-Object -First 1
    $bibliography = $headings | Where-Object { $_.heading -match "ΒΙΒΛΙΟΓΡΑΦΙΑ" } | Select-Object -First 1
    $bodyPages = $null
    if ($firstChapter -and $bibliography) { $bodyPages = $bibliography.page - $firstChapter.page }
    $out = [pscustomobject]@{
        file            = $full
        total_pages     = $total
        body_pages      = $bodyPages
        body_first_page = if ($firstChapter) { $firstChapter.page } else { $null }
        headings        = $headings
        word_count      = $doc.ComputeStatistics(0)
    }
    $out | ConvertTo-Json -Depth 5 | Set-Content -Path $report -Encoding UTF8
    $doc.Save()
    $doc.Close()
    Write-Output ("pages={0} body={1} report={2}" -f $total, $bodyPages, $report)
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
