-- ============================================================================
-- sql/queries.sql
-- Advanced SQL analysis for the Fortune 500 Growth & Performance panel dataset.
-- Target: SQLite (data/fortune500_panel.db), table: fortune500
-- Run with:  sqlite3 data/fortune500_panel.db < sql/queries.sql
-- Or open the .db in DB Browser for SQLite / load into Tableau / Power BI as
-- an ODBC/SQLite source for the dashboarding portion of this project.
-- ============================================================================

-- 1. Revenue CAGR (2014 -> 2023) per company, using window functions to grab
--    first/last year revenue per company without a self-join.
WITH bounds AS (
    SELECT
        company_id,
        company_name,
        sector,
        FIRST_VALUE(revenue_musd) OVER (PARTITION BY company_id ORDER BY fiscal_year
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS rev_2014,
        LAST_VALUE(revenue_musd) OVER (PARTITION BY company_id ORDER BY fiscal_year
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS rev_2023,
        fiscal_year
    FROM fortune500
)
SELECT
    company_id,
    company_name,
    sector,
    ROUND(rev_2014, 1) AS revenue_2014_musd,
    ROUND(rev_2023, 1) AS revenue_2023_musd,
    ROUND((POWER(rev_2023 / rev_2014, 1.0/9) - 1) * 100, 2) AS revenue_cagr_pct
FROM bounds
WHERE fiscal_year = 2023
ORDER BY revenue_cagr_pct DESC
LIMIT 25;

-- 2. Sector-level performance summary: total revenue, YoY growth, avg margin,
--    and a market-cap-weighted average margin for the latest year.
SELECT
    sector,
    COUNT(DISTINCT company_id)                         AS num_companies,
    ROUND(SUM(revenue_musd), 1)                         AS total_revenue_musd,
    ROUND(AVG(net_margin_pct), 2)                       AS avg_net_margin_pct,
    ROUND(SUM(net_income_musd * revenue_musd) / SUM(revenue_musd), 2) AS revenue_weighted_margin_pct,
    ROUND(AVG(revenue_growth_pct), 2)                   AS avg_yoy_growth_pct
FROM fortune500
WHERE fiscal_year = 2023
GROUP BY sector
ORDER BY total_revenue_musd DESC;

-- 3. Year-over-year growth ranking: which companies had the most consistent
--    top-quartile growth across ALL 10 years (a "consistent compounder" screen).
WITH yearly_quartile AS (
    SELECT
        company_id,
        company_name,
        fiscal_year,
        revenue_growth_pct,
        NTILE(4) OVER (PARTITION BY fiscal_year ORDER BY revenue_growth_pct DESC) AS growth_quartile
    FROM fortune500
    WHERE fiscal_year > 2014   -- first year has no growth figure
)
SELECT
    company_id,
    company_name,
    COUNT(*) AS years_in_top_quartile
FROM yearly_quartile
WHERE growth_quartile = 1
GROUP BY company_id, company_name
HAVING years_in_top_quartile >= 5
ORDER BY years_in_top_quartile DESC, company_name;

-- 4. Rolling 3-year average revenue growth per company (trend smoothing) using
--    a window frame — mirrors the smoothing done in the Python notebook.
SELECT
    company_id,
    company_name,
    fiscal_year,
    revenue_growth_pct,
    ROUND(AVG(revenue_growth_pct) OVER (
        PARTITION BY company_id ORDER BY fiscal_year
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_3yr_avg_growth_pct
FROM fortune500
WHERE company_id = 'C0001'
ORDER BY fiscal_year;

-- 5. Fortune500 rank volatility: standard deviation of rank across the decade,
--    identifying the most volatile vs. most stable companies by rank.
SELECT
    company_id,
    company_name,
    sector,
    MIN(fortune500_rank) AS best_rank,
    MAX(fortune500_rank) AS worst_rank,
    ROUND(AVG(fortune500_rank), 1) AS avg_rank,
    ROUND(
        SQRT(AVG(fortune500_rank * fortune500_rank) - AVG(fortune500_rank) * AVG(fortune500_rank)),
         2
    ) AS rank_stddev
FROM fortune500
GROUP BY company_id, company_name, sector
ORDER BY rank_stddev DESC
LIMIT 20;

-- 6. Profitability vs. R&D intensity correlation input: bucket companies by
--    R&D-to-revenue intensity and show average margin/growth per bucket.
SELECT
    CASE
        WHEN rd_spend_musd / revenue_musd < 0.01 THEN '0. <1% R&D'
        WHEN rd_spend_musd / revenue_musd < 0.05 THEN '1. 1-5% R&D'
        WHEN rd_spend_musd / revenue_musd < 0.10 THEN '2. 5-10% R&D'
        ELSE '3. >10% R&D'
    END AS rd_intensity_bucket,
    COUNT(*) AS company_years,
    ROUND(AVG(net_margin_pct), 2) AS avg_margin_pct,
    ROUND(AVG(revenue_growth_pct), 2) AS avg_growth_pct
FROM fortune500
WHERE fiscal_year = 2023
GROUP BY rd_intensity_bucket
ORDER BY rd_intensity_bucket;

-- 7. State/HQ concentration of Fortune 500 revenue (geographic view for maps
--    in Tableau/Power BI).
SELECT
    hq_state,
    fiscal_year,
    COUNT(DISTINCT company_id) AS num_companies,
    ROUND(SUM(revenue_musd), 1) AS total_revenue_musd
FROM fortune500
WHERE fiscal_year = 2023
GROUP BY hq_state, fiscal_year
ORDER BY total_revenue_musd DESC;

-- 8. Year-over-year "new entrants vs. dropouts" from the top 100 by revenue
--    rank, comparing 2019 vs 2023 (pre/post pandemic shakeout).
WITH top100_2019 AS (
    SELECT company_id FROM fortune500 WHERE fiscal_year = 2019 AND fortune500_rank <= 100
),
top100_2023 AS (
    SELECT company_id FROM fortune500 WHERE fiscal_year = 2023 AND fortune500_rank <= 100
)
SELECT
    (SELECT COUNT(*) FROM top100_2023 WHERE company_id NOT IN (SELECT company_id FROM top100_2019)) AS new_entrants_since_2019,
    (SELECT COUNT(*) FROM top100_2019 WHERE company_id NOT IN (SELECT company_id FROM top100_2023)) AS dropouts_since_2019;
