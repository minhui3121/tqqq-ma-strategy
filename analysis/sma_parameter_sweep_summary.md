# SMA Full-Period Sweep

Period: 2005-01-01 to 2026-01-01

## Best Candidate (full period)

short                8.400000e+01
long                 1.900000e+02
total_return         2.321822e+02
annualized_return    4.103085e-01
max_drawdown        -4.813654e-01
sharpe_ratio         9.963153e-01
trade_days           5.900000e+01
final_value          2.331822e+06
calmar_proxy         8.523847e-01

## Top 10 (by Sharpe)

 short  long  total_return  annualized_return  max_drawdown  sharpe_ratio  trade_days  final_value  calmar_proxy
    84   190    232.182171           0.410308     -0.481365      0.996315          59 2.331822e+06      0.852385
    85   190    232.182171           0.410308     -0.481365      0.996315          59 2.331822e+06      0.852385
    86   190    232.182171           0.410308     -0.481365      0.996315          59 2.331822e+06      0.852385
    87   190    232.182171           0.410308     -0.481365      0.996315          59 2.331822e+06      0.852385
    74   190    230.017155           0.409479     -0.481365      0.994903          59 2.310172e+06      0.850662
    75   190    226.672916           0.408184     -0.481365      0.993176          59 2.276729e+06      0.847970
    76   190    226.672916           0.408184     -0.481365      0.993176          59 2.276729e+06      0.847970
    77   190    226.672916           0.408184     -0.481365      0.993176          59 2.276729e+06      0.847970
    78   190    226.672916           0.408184     -0.481365      0.993176          59 2.276729e+06      0.847970
    79   190    226.672916           0.408184     -0.481365      0.993176          59 2.276729e+06      0.847970

## Notes

- This single-period sweep selects the best-performing SMA pair over the entire given range (no train/test split).
- Use with caution: single-period winners can be prone to overfitting to the chosen window.
