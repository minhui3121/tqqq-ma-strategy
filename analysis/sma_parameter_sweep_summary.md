# SMA Full-Period Sweep

Period: 2005-01-01 to 2026-01-01

## Best Candidate (full period)

short                8.000000e+01
long                 1.900000e+02
total_return         2.266729e+02
annualized_return    4.081836e-01
max_drawdown        -4.813654e-01
sharpe_ratio         9.931763e-01
trade_days           5.900000e+01
final_value          2.276729e+06
calmar_proxy         8.479703e-01

## Top 10 (by Sharpe)

 short  long  total_return  annualized_return  max_drawdown  sharpe_ratio  trade_days  final_value  calmar_proxy
    80   190    226.672916           0.408184     -0.481365      0.993176          59 2.276729e+06      0.847970
    60   190    211.813248           0.402202     -0.481365      0.983356          57 2.128132e+06      0.835545
   100   190    208.023522           0.400614     -0.553139      0.979926          61 2.090235e+06      0.724257
   110   190    208.485038           0.400809     -0.553139      0.979569          63 2.094850e+06      0.724609
    90   190    206.544966           0.399988     -0.525363      0.979483          61 2.075450e+06      0.761354
    70   190    202.162570           0.398105     -0.481365      0.976680          59 2.031626e+06      0.827032
    50   190    188.619852           0.392036     -0.481365      0.966538          61 1.896199e+06      0.814424
    40   190    175.513909           0.385762     -0.523769      0.954040          69 1.765139e+06      0.736512
    20   190    166.163818           0.381014     -0.551961      0.943948          73 1.671638e+06      0.690291
   120   190    162.529396           0.379101     -0.543829      0.942047          69 1.635294e+06      0.697096

## Notes

- This single-period sweep selects the best-performing SMA pair over the entire given range (no train/test split).
- Use with caution: single-period winners can be prone to overfitting to the chosen window.
