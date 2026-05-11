$pairs = @(
    @{ s = "EURUSD"; q = "USD" },
    @{ s = "GBPUSD"; q = "USD" },
    @{ s = "AUDUSD"; q = "USD" },
    @{ s = "NZDUSD"; q = "USD" },
    @{ s = "USDJPY"; q = "JPY" },
    @{ s = "USDCHF"; q = "CHF" },
    @{ s = "USDCAD"; q = "CAD" },
    @{ s = "GBPJPY"; q = "JPY" },
    @{ s = "EURJPY"; q = "JPY" } 
)

foreach ($p in $pairs) {
    Write-Host "=== Fetching $($p.s) ===" -ForegroundColor Cyan
    fetch-ib-data $p.s --sec-type CASH --exchange IDEALPRO --currency $p.q `
        --bar-size "5 mins" --start 2020-01-01 `
        --no-use-rth --what-to-show MIDPOINT `
        --output "E:\data_dump\market_data\raw\ib\5\$($p.s)_5m.parquet"
}