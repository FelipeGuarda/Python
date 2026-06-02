==============================================================================
list_ciervo_guina_images.py — Manual review for ciervo & güiña
==============================================================================

Total rows across 3 campaigns: 3,582
Rows tagged ciervo or güiña  : 37

Wrote → C:\Users\USUARIO\Dev\Python\camera-traps\Anual-reports\2025\data\manual_review_ciervo_guina.csv

# Per-camera image counts (raw rows in CSVs, before report dedup/event collapse)
  scientificName  camera_num  n_images                   campaigns
  Cervus elaphus         4.0         3                pv_2025_2026
  Cervus elaphus         7.0         1                  otono_2025
  Cervus elaphus         9.0         1                  otono_2025
  Cervus elaphus        13.0         6 primavera_2025,pv_2025_2026
  Cervus elaphus        14.0         1                pv_2025_2026
  Cervus elaphus        15.0         3                  otono_2025
  Cervus elaphus        20.0         1              primavera_2025
Leopardus guigna         3.0         2                  otono_2025
Leopardus guigna         5.0         5                pv_2025_2026
Leopardus guigna         8.0         2                  otono_2025
Leopardus guigna        14.0         1                pv_2025_2026
Leopardus guigna        16.0         3                  otono_2025
Leopardus guigna        25.0         1                pv_2025_2026
Leopardus guigna        26.0         1                pv_2025_2026


## Ciervo rojo (Cervus elaphus) — 16 image rows

| Campaign | CT | Deployment | Date (corrected) | File | Thumbnail (on Windows) | Source filePath | Review | Conflict? |
|---|---|---|---|---|---|---|---|---|
| pv_2025_2026 | CT04 | TC4_M5.2 | 2025-09-06 00:28:57 | `09060102.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Ciervo_rojo_Cervus_elaphus\TC4_M5_2_09060102.jpg` ✗ missing | `` | corrected | ⚠️ primavera=Lepus europaeus | pv=Cervus elaphus |
| pv_2025_2026 | CT04 | TC4_M5.2 | 2025-10-15 06:28:37 | `10150189.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Ciervo_rojo_Cervus_elaphus\TC4_M5_2_10150189.jpg` ✓ | `` | corrected | ⚠️ primavera=Canis lupus familiaris | pv=Cervus elaphus |
| pv_2025_2026 | CT04 | TC4_M5.2 | 2025-11-04 19:35:47 | `11040253.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Ciervo_rojo_Cervus_elaphus\TC4_M5_2_11040253.jpg` ✗ missing | `` | corrected | ⚠️ primavera=animal sin especie | pv=Cervus elaphus |
| otono_2025 | CT07 | CT07 | 2025-06-02 19:35:14 | `06020042.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Ciervo_rojo_Cervus_elaphus\CT07_06020042.jpg` ✓ | `CT07\06020042.JPG` | corrected |  |
| otono_2025 | CT09 | CT09 | 2025-04-03 20:26:35 | `04030431.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Ciervo_rojo_Cervus_elaphus\CT09_04030431.jpg` ✓ | `CT09\04030431.JPG` | corrected |  |
| primavera_2025 | CT13 | TC13_M16.2 | 2025-10-25 18:47:29 | `10250085.JPG` | `_(no export — Linux side)_` — | `TC13_M16.2\10250085.JPG` | corrected |  |
| pv_2025_2026 | CT13 | TC13_M16.2 | 2025-10-25 18:47:29 | `10250085.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Ciervo_rojo_Cervus_elaphus\TC13_M16_2_10250085.jpg` ✓ | `` | corrected |  |
| primavera_2025 | CT13 | TC13_M16.2 | 2025-10-25 18:47:30 | `10250086.JPG` | `_(no export — Linux side)_` — | `TC13_M16.2\10250086.JPG` | corrected |  |
| primavera_2025 | CT13 | TC13_M16.2 | 2025-10-25 18:47:30 | `10250087.JPG` | `_(no export — Linux side)_` — | `TC13_M16.2\10250087.JPG` | corrected |  |
| pv_2025_2026 | CT13 | TC13_M16.2 | 2025-10-25 18:47:30 | `10250086.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Ciervo_rojo_Cervus_elaphus\TC13_M16_2_10250086.jpg` ✓ | `` | corrected |  |
| pv_2025_2026 | CT13 | TC13_M16.2 | 2025-10-25 18:47:30 | `10250087.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Ciervo_rojo_Cervus_elaphus\TC13_M16_2_10250087.jpg` ✓ | `` | corrected |  |
| pv_2025_2026 | CT14 | TC14_M11.2 | 2025-10-02 09:52:37 | `10020834.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Ciervo_rojo_Cervus_elaphus\TC14_M11_2_10020834.jpg` ✓ | `` | corrected |  |
| otono_2025 | CT15 | CT15 | 2025-01-23 18:02:37 (+8yr) | `01230193.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Ciervo_rojo_Cervus_elaphus\CT15_01230193.jpg` ✓ | `CT15\01230193.JPG` | corrected |  |
| otono_2025 | CT15 | CT15 | 2025-01-23 18:02:38 (+8yr) | `01230194.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Ciervo_rojo_Cervus_elaphus\CT15_01230194.jpg` ✓ | `CT15\01230194.JPG` | corrected |  |
| otono_2025 | CT15 | CT15 | 2025-01-29 15:21:02 (+8yr) | `01290209.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Ciervo_rojo_Cervus_elaphus\CT15_01290209.jpg` ✓ | `CT15\01290209.JPG` | corrected |  |
| primavera_2025 | CT20 | TC20_M17.2 | 2025-11-14 05:05:04 | `11140392.JPG` | `_(no export — Linux side)_` — | `TC20_M17.2\11140392.JPG` | corrected | ⚠️ primavera=Cervus elaphus | pv=Puma concolor |


## Güiña (Leopardus guigna) — 21 image rows

| Campaign | CT | Deployment | Date (corrected) | File | Thumbnail (on Windows) | Source filePath | Review | Conflict? |
|---|---|---|---|---|---|---|---|---|
| otono_2025 | CT03 | CT03 | 2024-10-18 12:22:20 | `10180013.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Guina_Leopardus_guigna\CT03_10180013.jpg` ✓ | `CT03\10180013.JPG` | corrected |  |
| otono_2025 | CT03 | CT03 | 2024-10-18 12:22:21 | `10180014.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Guina_Leopardus_guigna\CT03_10180014.jpg` ✓ | `CT03\10180014.JPG` | confirmed |  |
| pv_2025_2026 | CT05 | TC5_M9.2 | 2025-07-11 20:24:09 | `07110001.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Guina_Leopardus_guigna\TC5_M9_2_07110001.jpg` ✓ | `` | corrected |  |
| pv_2025_2026 | CT05 | TC5_M9.2 | 2025-08-30 02:38:10 | `08300093.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Guina_Leopardus_guigna\TC5_M9_2_08300093.jpg` ✗ missing | `` | corrected |  |
| pv_2025_2026 | CT05 | TC5_M9.2 | 2025-10-18 23:37:43 | `10180169.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Guina_Leopardus_guigna\TC5_M9_2_10180169.jpg` ✓ | `` | corrected |  |
| pv_2025_2026 | CT05 | TC5_M9.2 | 2025-10-18 23:37:44 | `10180170.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Guina_Leopardus_guigna\TC5_M9_2_10180170.jpg` ✗ missing | `` | confirmed |  |
| pv_2025_2026 | CT05 | TC5_M9.2 | 2025-10-30 20:59:53 | `10300189.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Guina_Leopardus_guigna\TC5_M9_2_10300189.jpg` ✓ | `` | confirmed |  |
| otono_2025 | CT08 | CT08 | 2024-09-26 05:30:04 | `09260026.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Guina_Leopardus_guigna\CT08_09260026.jpg` ✓ | `CT08\09260026.JPG` | corrected |  |
| otono_2025 | CT08 | CT08 | 2024-11-26 22:13:06 | `11260099.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Guina_Leopardus_guigna\CT08_11260099.jpg` ✗ missing | `CT08\11260099.JPG` | corrected |  |
| pv_2025_2026 | CT14 | TC14_M11.2 | 2026-01-03 15:30:01 | `01030967.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Guina_Leopardus_guigna\TC14_M11_2_01030967.jpg` ✗ missing | `` | corrected |  |
| otono_2025 | CT16 | CT16 | 2025-01-03 20:38:07 (+8yr) | `09100029.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Guina_Leopardus_guigna\CT16_09100029.jpg` ✓ | `CT16\09100029.JPG` | corrected |  |
| otono_2025 | CT16 | CT16 | 2025-01-03 20:38:08 (+8yr) | `09100030.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Guina_Leopardus_guigna\CT16_09100030.jpg` ✓ | `CT16\09100030.JPG` | corrected |  |
| otono_2025 | CT16 | CT16 | 2025-02-06 06:29:27 (+8yr) | `02060086.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Otoño 2025\species\Guina_Leopardus_guigna\CT16_02060086.jpg` ✗ missing | `CT16\02060086.JPG` | corrected |  |
| pv_2025_2026 | CT25 | TC25_M22.2 | 2025-06-26 02:51:11 | `06260001.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Guina_Leopardus_guigna\TC25_M22_2_06260001.jpg` ✓ | `` | corrected |  |
| pv_2025_2026 | CT26 | TC26_M23.2 | 2025-07-18 03:13:04 | `07180010.JPG` | `C:\Users\USUARIO\Dev\Python\camera-traps\exports\Primavera-verano 2025-2026\species\Guina_Leopardus_guigna\TC26_M23_2_07180010.jpg` ✓ | `` | corrected |  |

**Unmappable deployments (CT number could not be parsed — dropped by the report):**

| Campaign | CT | Deployment | Date (corrected) | File | Thumbnail (on Windows) | Source filePath | Review | Conflict? |
|---|---|---|---|---|---|---|---|---|
| primavera_2025 | **?** | 100EK113 | 2025-07-11 20:24:09 | `07110001.JPG` | _(no export)_ — | `100EK113\07110001.JPG` | corrected | (suspected CT05 per session log) |
| primavera_2025 | **?** | 100EK113 | 2025-08-30 02:38:10 | `08300093.JPG` | _(no export)_ — | `100EK113\08300093.JPG` | corrected | (suspected CT05 per session log) |
| primavera_2025 | **?** | 100EK113 | 2025-10-18 23:37:43 | `10180169.JPG` | _(no export)_ — | `100EK113\10180169.JPG` | corrected | (suspected CT05 per session log) |
| primavera_2025 | **?** | 100EK113 | 2025-10-18 23:37:44 | `10180170.JPG` | _(no export)_ — | `100EK113\10180170.JPG` | confirmed | (suspected CT05 per session log) |
| primavera_2025 | **?** | 100EK113 | 2025-10-30 20:59:53 | `10300189.JPG` | _(no export)_ — | `100EK113\10300189.JPG` | confirmed | (suspected CT05 per session log) |
| primavera_2025 | **?** | 100EK113 | 2025-10-30 20:59:54 | `10300190.JPG` | _(no export)_ — | `100EK113\10300190.JPG` | corrected | (suspected CT05 per session log) |
