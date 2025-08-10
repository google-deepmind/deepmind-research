# PowerShell script for downloading Learning to Simulate datasets
# Usage: .\download_dataset.ps1 -DatasetName "WaterDrop" -OutputDir ".\datasets"

param(
    [Parameter(Mandatory=$true)]
    [string]$DatasetName,
    
    [Parameter(Mandatory=$true)]
    [string]$OutputDir
)

function Download-FileWithProgress {
    param(
        [string]$Url,
        [string]$FilePath
    )
    
    try {
        Write-Host "📥 Downloading $(Split-Path $FilePath -Leaf)..." -ForegroundColor Blue
        
        # Use Invoke-WebRequest with progress
        $response = Invoke-WebRequest -Uri $Url -OutFile $FilePath -PassThru
        
        if (Test-Path $FilePath) {
            $fileSize = (Get-Item $FilePath).Length
            Write-Host "✅ Downloaded $(Split-Path $FilePath -Leaf) ($([math]::Round($fileSize/1MB, 2)) MB)" -ForegroundColor Green
            return $true
        }
        return $false
    }
    catch {
        Write-Host "❌ Error downloading $(Split-Path $FilePath -Leaf): $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Main script
$baseUrl = "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/$DatasetName/"
$files = @('metadata.json', 'train.tfrecord', 'valid.tfrecord', 'test.tfrecord')

# Create output directory
$datasetDir = Join-Path $OutputDir $DatasetName
if (!(Test-Path $datasetDir)) {
    New-Item -ItemType Directory -Path $datasetDir -Force | Out-Null
}

Write-Host "🔄 Downloading dataset '$DatasetName' to $datasetDir" -ForegroundColor Yellow

$successCount = 0
foreach ($filename in $files) {
    $url = $baseUrl + $filename
    $filepath = Join-Path $datasetDir $filename
    
    if (Download-FileWithProgress -Url $url -FilePath $filepath) {
        $successCount++
    }
}

Write-Host "`n🎉 Download complete! $successCount/$($files.Count) files downloaded successfully." -ForegroundColor Cyan

if ($successCount -eq $files.Count) {
    Write-Host "✅ All files downloaded to: $datasetDir" -ForegroundColor Green
} else {
    Write-Host "⚠️ Some downloads failed. Check your internet connection and try again." -ForegroundColor Yellow
}
