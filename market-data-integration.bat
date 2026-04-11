@REM  TopOneTrader-MT5
fetch-mt5-data --broker toponetrader --account demo  --symbols "AUDUSD,BTCUSD,ETHUSD,EURGBP,EURJPY,EURUSD,GBPJPY,GBPUSD,GER40,JPN225,US30,US500,USDCAD,USDJPY,US100,XAGUSD,XAUUSD" --timeframe M5  --start-date 2018-01-01 --output D:\data_dump\market_data\raw --env-dir .\systems\
@REM  FXIFY
fetch-mt5-data --broker fxify --account demo  --symbols "AUDCAD.r,AUDJPY.r,AUDUSD.r,BTCUSDT.r,DE30.r,DJ30.r,ETHUSDT.r,EURCAD.r,EURCHF.r,EURGBP.r,EURJPY.r,EURUSD.r,GBPCAD.r,GBPJPY.r,GBPUSD.r,HKG50.r,JPN225.r,US500.r,USDCAD.r,USDCHF.r,USDJPY.r,USOil.r,USTEC.r,XAGUSD.r,XAUUSD.r,VIX.r,RUS2000.r" --timeframe M5  --start-date 2024-06-05 --output D:\data_dump\market_data\raw --env-dir .\systems\
@REM  Deriv-Demo
fetch-mt5-data --broker deriv --account demo  --symbols "Step Index,Step Index 200,Volatility 75 Index,Volatility 100 Index" --timeframe M5  --start-date 2021-01-01 --output D:\data_dump\market_data\raw --env-dir .\systems\
@REM  Deriv-Server Live
fetch-mt5-data --broker deriv --account live  --symbols "Step Index,Step Index 200,Volatility 75 Index,Volatility 100 Index" --timeframe M5  --start-date 2021-01-01 --output D:\data_dump\market_data\raw --env-dir .\systems\
