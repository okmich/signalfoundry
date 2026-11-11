@REM  TopOneTrader-MT5
fetch-mt5-data --broker toponetrader --account demo  --symbols "AUDUSD,BTCUSD,ETHUSD,EURGBP,EURJPY,EURUSD,GBPJPY,GBPUSD,GER40,JPN225,US30,US500,USDCAD,USDJPY,US100,XAGUSD,XAUUSD" --timeframe M5  --start-date 2018-01-01 --output D:\data_dump\market_data\raw
@REM  FXIFY
fetch-mt5-data --broker fxify --account demo  --symbols "AUDCAD.r,AUDJPY.r,AUDUSD.r,BTCUSDT.r,DE30.r,DJ30.r,ETHUSDT.r,EURCAD.r,EURCHF.r,EURGBP.r,EURJPY.r,EURUSD.r,GBPCAD.r,GBPJPY.r,GBPUSD.r,HKG50.r,JPN225.r,US500.r,USDCAD.r,USDCHF.r,USDJPY.r,USOil.r,USTEC.r,XAGUSD.r,XAUUSD.r,VIX.r,RUS2000.r" --timeframe M5  --start-date 2024-06-05 --output D:\data_dump\market_data\raw
@REM  FXIFY / FXPIG-Server bar-data audit: raw-data-audit (okmich-quant-utils). Samples for reference, not run by this file.
@REM  Data root: %OKMICH_QUANT_MARKET_DATA_DIR% (or --data-dir). State folder: %OKMICH_QUANT_DATA_STATE_DIR% (or --state-dir),
@REM  outside the data tree. Nothing is ever written into the market-data tree; --dry-run writes nothing at all.
@REM  --registry: your drop/refetch/inspect decisions (CSV). Build it by auditing without one, reviewing the unregistered
@REM  CRITICAL/MAJOR findings, recording decisions, and re-auditing with it (raw-data-audit README, "Building a registry").
@REM  Audit the raw files (read-only report):
@REM  raw-data-audit audit     --broker FXPIG-Server --out <report folder>
@REM  raw-data-audit audit     --broker FXPIG-Server --registry <registry.csv> --symbols EURUSD GBPUSD --no-write
@REM  Remediate into the state folder (plan only, then for real):
@REM  raw-data-audit remediate --broker FXPIG-Server --registry <registry.csv> --dry-run
@REM  raw-data-audit remediate --broker FXPIG-Server --registry <registry.csv>
@REM  Refetch holes, dropped windows and spikes from the FXIFY terminal (read-only login; dry run first):
@REM  raw-data-audit remediate --broker FXPIG-Server --registry <registry.csv> --fetch --dry-run
@REM  raw-data-audit remediate --broker FXPIG-Server --registry <registry.csv> --fetch
@REM  Hole ledger, snapshot of raw/, and the history-rewrite diff against the latest snapshot:
@REM  raw-data-audit holes     --broker FXPIG-Server --all
@REM  raw-data-audit snapshot  --broker FXPIG-Server
@REM  raw-data-audit diff      --broker FXPIG-Server
@REM  Deriv-Demo
fetch-mt5-data --broker deriv --account demo  --symbols "Step Index,Step Index 200,Volatility 75 Index,Volatility 100 Index" --timeframe M5  --start-date 2021-01-01 --output D:\data_dump\market_data\raw
@REM  Deriv-Server Live
fetch-mt5-data --broker deriv --account live  --symbols "Step Index,Step Index 200,Volatility 75 Index,Volatility 100 Index" --timeframe M5  --start-date 2021-01-01 --output D:\data_dump\market_data\raw
