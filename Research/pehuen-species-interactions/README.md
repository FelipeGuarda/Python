# pehuen-species-interactions

Prototype analysis of species distribution and temporal interactions in Bosque
Pehuen, based on camera-trap data from the Otoño 2025 and Primavera 2025
campaigns.  Written in R using `camtrapR` and `overlap`.

---

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate pehuen-analysis
```

If any package fails to resolve from conda-forge, run the CRAN fallback:

```bash
conda activate pehuen-analysis
Rscript setup_packages.R
```

### 2. Verify data paths

Open `R/01_load_data.R` and confirm `PATH_OTONO`, `PATH_PRIMAVERA`,
`PATH_GEOJSON`, and `PATH_BOUNDARY` point to the correct locations on this
machine.

---

## Running the analysis

Run scripts in order — each script saves `.rds` files that the next one reads.

```bash
conda activate pehuen-analysis

Rscript R/01_load_data.R          # reads CSVs + GeoJSON, saves data/*.rds
Rscript R/02_detection_summary.R  # bar charts: counts, rates, occupancy
Rscript R/03_activity_patterns.R  # 24h kernel density activity curves
Rscript R/04_temporal_overlap.R   # Dhat4 pairwise overlap + CI
Rscript R/05_spatial_distribution.R  # detection bubble maps
```

All figures are written to `figures/`.

---

## Adding the Primavera-Verano (or future) campaign

1. In `R/01_load_data.R`, add `PATH_NEW_CAMPAIGN <- "..."` in the Paths block.
2. Call `read_campaign_csv(PATH_NEW_CAMPAIGN, "PrimaveraVerano_2025")`.
3. Apply the appropriate station ID standardisation block (copy the Otoño or
   Primavera block and adapt the regex for the new format).
4. Add the new dataframe to the `bind_rows()` call.
5. Re-run all scripts.

---

## Station ID mapping

| tc# | GeoJSON id | Otoño CSV    | Primavera CSV | Notes          |
|:---:|:----------:|:------------:|:-------------:|:--------------|
| 1   | TC-01      | CT01         | TC1_M7.2      | sd validated  |
| 2   | TC-02      | CT02         | —             | Otoño only    |
| 3   | TC-03      | CT03         | —             | Otoño only    |
| 4   | TC-04      | CT04         | TC4_M5.2      | sd validated  |
| 5   | TC-05      | CT05         | —             | Otoño only    |
| 6   | TC-06      | CT06         | TC6_M1.2      | sd validated  |
| 7   | TC-07      | CT07         | —             | Otoño only    |
| 8   | TC-08      | CT08         | —             | Otoño only    |
| 9   | TC-09      | CT09         | TC9_M2.2      | sd validated  |
| 10  | TC-10      | CT10         | TC10_M3.2     | sd validated  |
| 11  | TC-11      | CT11         | TC11_M15.2    | sd validated  |
| 12  | TC-12      | CT12         | TC12_M17.2    | sd validated  |
| 13  | TC-13      | CT13         | TC13_M16.2    | sd validated  |
| 14  | TC-14      | CT14         | —             | Otoño only    |
| 15  | TC-15      | CT15         | TC15_M12.2    | sd validated  |
| 16  | TC-16      | CT16         | TC16_M13.2    | sd validated  |
| 17  | TC-17      | CT17         | —             | Otoño only    |
| 18  | TC-18      | CT18         | TC18_M15.2    | sd validated  |
| 19  | TC-19      | CT19         | TC19_M16.2    | sd validated  |
| 20  | TC-20      | CT20         | TC20_M17.2    | sd validated  |
| 21–26 | TC-21…TC-26 | —         | —             | GeoJSON only  |

`100EK113` (Primavera): anomalous entry, filtered out in 01_load_data.R.

---

## Focal species

| Spanish       | Latin                   | Guild    |
|:--------------|:------------------------|:---------|
| Puma          | Puma concolor           | Native   |
| Guiña         | Leopardus guigna        | Native   |
| Zorro culpeo  | Lycalopex culpaeus      | Native   |
| Jabalí        | Sus scrofa              | Invasive |
| Liebre        | Lepus europaeus         | Invasive |
| Perro         | Canis lupus familiaris  | Invasive |
