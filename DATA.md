# Data

The data are from Bloomberg and are licensed, so they are not in this repository. With Bloomberg access, rebuild each file at the path shown. Hashes of the frozen panels are in `DATA_FREEZE.txt`.

| File | Content | Bloomberg source |
|---|---|---|
| `data/thesis_data.csv` | Main weekly panel, January 1999 to September 2026: Friday levels and weekly log returns | `PX_LAST`, weekly, for the tickers below |
| `data/thesis_data_domestic.csv` | Domestic-risk panel from December 2002 | As above, adding `REPSOU CDS USD SR 5Y D14 Corp`, `M1FMEAFZ Index` and `ALBIAY Index` |
| `transformed_data.csv` | Panel built by `00_transform.py` from a raw export (`raw_data.csv`) | `PX_LAST`, weekly |
| `bbg_prices_weekly.csv` | Weekly closes of both lines of each dual-listed constituent, the currencies, and South African single stocks; columns are named by company and exchange | `PX_LAST`, weekly |
| `top40_monthly_class.csv`, `members_classified.csv` | Top 40 constituent weights with the author's category | `INDX_MWEIGHT_HIST` on `TOP40 Index`, month-end from January 2003 |

## Tickers in the main panel

`JALSH Index`, `TOP40 Index`, `RESI20 Index`, `FINI15 Index`, `INDI25 Index`, `JGENR Index`, `JMNNG Index`, `MXWO Index`, `S5MATRX Index`, `S5METL Index`, `S5INDU Index`, `S5FINL Index`, `S5CONS Index`, `S5COND Index`, `MXWO0MM Index`, `MXWO0MT Index`, `MXWO0CS Index`, `MXWO000V Index`, `MXWO000G Index`, `MXWO0FN Index`, `MXWO0CD Index`, `M1WOQU Index`, `SPX Index`, `SX5E Index`, `UKX Index`, `USDZAR Curncy`, `ZAR Curncy`, `DXY Curncy`, `XAU Curncy`, `XPT Curncy`, `XPD Curncy`, `LMCADS03 Comdty`, `HG1 Comdty`, `CL1 Comdty`, `BCOM Index`, `BCOMPR Index`, `USGG10YR Index`, `SARPRT Index`, `MXEF Index`, `VIX Index`, `MXZA Index`, `MXZA0IN Index`, `MXZA0CD Index`, `MXZA0CS Index`


The dual-listed lines are: Anglo JSE, Anglo London, Richemont JSE, Richemont Zurich, BAT JSE, BAT London, Glencore JSE, Glencore London, Mondi JSE, Mondi London, BHP JSE, BHP Sydney, South32 JSE, South32 Sydney, Prosus JSE, Prosus Amsterdam, AB InBev JSE, AB InBev Brussels, NEPI JSE, NEPI Amsterdam, AngloGold JSE, AngloGold New York.
