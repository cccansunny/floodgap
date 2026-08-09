# FloodGap

**Where should the next flood sensors go?**

A data-driven flood-sensor siting model, inspired by CEAR Hub's sea-level sensor network in Savannah, Georgia, and applied to the Houston and Galveston region. Built over one semester for the Georgia Tech VIP program, Coastal Georgia Climate Solutions.

**[Open the interactive weight explorer](https://cccansunny.github.io/floodgap/floodgap/interactive/floodgap_weight_explorer_satellite.html)** | **[Compare the five schemes](https://cccansunny.github.io/floodgap/floodgap/interactive/floodgap_five_schemes_satellite.html)**

---

## The question

Real-time water-level sensors tell communities when and where water is rising, but they are scarce and unevenly placed. CEAR Hub runs a network of them around Savannah, and seeing it raised an obvious question for my own city:

> If Houston had funding for 20 new flood sensors, where are the 20 most ideal places to put them?

FloodGap is an attempt to answer that with data rather than intuition: score every census tract on flood hazard and social vulnerability, rank them, and recommend where a limited number of sensors would do the most good.

## The two turns that shaped the project

The original plan was to validate the model against CEAR Hub's 48 existing sensors: if the model scores high where experts placed sensors, it captures expert judgment. It did not. The 48 sites averaged only the 42nd percentile of the model's score distribution, slightly below random. The reason turned out to matter more than the result: those sensors were installed 6 to 8 years ago based on **feasibility**, easy access, near a waterway, something to mount on, often a fence post, and the deployment was never completed. They are where it was possible to install, not where it is ideal to install.

So the mismatch is not a failed validation. It is a finding. The existing network is feasibility-limited and systematically misses high-risk, vulnerable places, and the model can surface exactly those. That gap is the second meaning of the project's name. With no ground truth of ideal placement, fitting weights is meaningless. So instead of guessing weights and calling them correct, this project treats weights as **explicit policy options**: five weighting schemes are scored, and the recommendation is the set of sites that rank high under every one of them.

## What the model finds

Applying the Savannah-style tidal-only logic to Houston put nearly every priority on the coast: Galveston tracts entered the top 50 at roughly **30 times** the per-tract rate of Harris County, while Harvey's actual disaster was inland. Adding a compound flood factor (tidal, stormwater, riverine) moved Harris County from 3 of the top 50 sites to 45.

Across the five weighting schemes, only **2 of 1,211 tracts** are chosen by all five (both in Texas City and La Marque); 28 more appear in four of five. The robust sites organize into four clusters:

| Cluster | Why it ranks | Sensor type |
|---|---|---|
| Texas City / La Marque | Highest surge exposure (tidal 0.92) with high vulnerability (SVI 0.69); the only region in all five schemes | Water-level |
| Gulfton / Sharpstown | Most impervious cluster (0.65), very high vulnerability (0.74), Brays Bayou watershed; invisible under tidal-only weighting | Street-light |
| East End / Ship Channel | Closest to water of any cluster (0.6 mi), SVI 0.82, industrial-adjacent environmental-justice concern | Mixed |
| Kashmere / Fifth Ward | Highest social vulnerability in the study area (SVI 0.91), chronic flooding on Hunting Bayou | Street-light |

Each recommended site carries a sensor-type tag derived from its dominant flood component: tidal and riverine sites need water-level sensors mounted over water, while inland stormwater sites have no waterway and call for pole-mounted ultrasonic street-light sensors instead.

## How the score works

```
score          = W_flood x compound_flood + W_vuln x social_vulnerability
compound_flood = w_tidal x tidal + w_storm x stormwater + w_river x riverine
```

Every factor is normalized 0 to 1 before weighting, and weights are normalized to sum to 1.

| Factor | Source |
|---|---|
| tidal | NOAA high-tide flood days combined with USGS elevation |
| stormwater | NOAA C-CAP impervious surface fraction, sampled from the 30 m raster |
| riverine | Distance to the nearest of 11 major bayous, inverted so closer means higher |
| social vulnerability | CDC/ATSDR Social Vulnerability Index, tract level |

The five schemes are Tidal-only (100/0/0, flood 60), Equal (33/33/33, flood 60), Harvey-informed (25/50/25, flood 60), Flood-dominant (33/33/33, flood 80) and Equity-forward (33/33/33, flood 50).

## Running the code

Requires Python 3.10 or newer with pandas, numpy, rasterio, pyproj and matplotlib.

```
pip install pandas numpy rasterio pyproj matplotlib
cd floodgap
python code/floodgap_compound.py     # writes output/houston_robust.csv and top-20
python code/render_maps.py           # writes all maps to output/
```

Set `REGION = "houston"` or `"georgia"` at the top of `floodgap_compound.py`. The elevation caches in `data/` mean no API calls are needed; delete them to re-query Open-Elevation from scratch.

## Data sources

- NOAA CO-OPS high-tide flooding: https://tidesandcurrents.noaa.gov/
- USGS elevation via Open-Elevation: https://open-elevation.com/
- CDC/ATSDR Social Vulnerability Index: https://www.atsdr.cdc.gov/placeandhealth/svi/
- NOAA C-CAP high-resolution land cover: https://coast.noaa.gov/digitalcoast/data/ccaphighres.html
- Census TIGER gazetteer: https://www.census.gov/geographies/reference-files.html
- Reference paper: Tien, Lozano and Chavan (2023), Communications Earth and Environment 4:96, https://www.nature.com/articles/s43247-023-00761-1
- Street-light sensor model, NYC FloodNet: https://www.floodnet.nyc/

## Limitations

The compound-flood internal weights cannot be learned from Savannah, which has no stormwater or riverine variation to learn from, so they are policy choices rather than fitted values; this is why the five-scheme structure exists. Vulnerability is tract-level SVI, which under-represents marginalized populations, with block-group SVI the known upgrade. The riverine factor is straight-line distance to simplified bayou centerlines, not hydraulic modeling. The stormwater factor is impervious fraction only, without drainage capacity or observed ponding records. Scores rank tracts rather than exact installation points, which is what the field-survey phase is for.

## What did not work

The semester report documents these in full, including the NOAA resolution problem (all 174 Georgia tracts shared one flood-day value), the invalid validation described above, an OpenStreetMap dependency abandoned after chronic API timeouts, a county-name regex that silently pulled in Harrison County, and a coverage factor that was removed after an audit showed it failed to prevent clustering while correlating negatively with the risk signals it shared a score with.

## Next steps

Fine-grain the scoring to block-group level inside each cluster and rank streets using 311 flooding complaints and HCFCD flood history; desktop pre-screen candidate poles and bridges via Street View and ownership records; field reconnaissance with a structured site checklist; and test the model's picks against the areas the team wanted to monitor but could not install in.
