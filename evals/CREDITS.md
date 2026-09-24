# Credits for the eval corpora

Every image the evals run on was fetched from Wikimedia Commons and is the work of the
person named below, under the licence named below. The images themselves are **not** in this
repository — `evals/_corpus/` is gitignored, and each fetch script re-downloads them — but
the **sheets and masks committed under `evals/segmentation_models/` and
`evals/object_completeness/` are derivative works of them**, so the attribution belongs here
where it ships. The `evals/out_*/` sheets each eval writes are not committed.

None of these images are covered by this project's LICENSE. Each keeps its own licence,
and several are share-alike (CC BY-SA), which propagates to any derivative you make of the
committed sheets.

---

## Photographs, charts and screenshots

Fetched by `evals/real_corpus.py` into `evals/_corpus/`.  
Used by: evals/grounding.py, colour_spaces.py, colour_space_defaults.py, object_completeness.py, object_verdicts.py, real_world.py, space_aware_checks.py, text_path.py, chart_remap.py

| file | artist | licence | source |
|---|---|---|---|
| `food_pizza` | Shisma | CC BY 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Margherita_pizza_on_plate.jpg) |
| `product_shoe` | Ally.J. | CC BY 2.0 | [Commons](https://commons.wikimedia.org/wiki/File:Product_RED_Converse_%284034111954%29.jpg) |
| `product_white` | ConverseChucks | CC BY-SA 2.0 | [Commons](https://commons.wikimedia.org/wiki/File:Superga_White_new_05.jpg) |
| `portrait_studio` | Dmitry Makeev | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Russia%2C_Moscow_Oblast._Young_woman_with_umbrella%2C_studio_portrait.jpg) |
| `flower_macro` | dion gillard | CC BY 2.0 | [Commons](https://commons.wikimedia.org/wiki/File:Gazania_petals_%28macro%29_-_Ashbury%2C_2007_1.jpg) |
| `night_city` | Wilfredor | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:%C3%89difice_Price_at_night%2C_Quebec_city%2C_Canada.jpg) |
| `ui_screenshot` | Yair Aichenbaum | MIT | [Commons](https://commons.wikimedia.org/wiki/File:Files_%28software%29_screenshot.png) |
| `chart_bar` | — | CC BY 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Fish-catch-sector-bar_%28OWID_0438%29.png) |
| `chart_health` | H Regitze K | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Health_indicators_Bar_chart.png) |
| `city_wide` | King of Hearts | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Midtown_Manhattan_from_Jersey_City_September_2020_HDR.jpg) |
| `car_red` | Kai3952 | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Red_car_parked_in_front_of_Wupaochun_Bakery_Taichung_on_29_January_2021.jpg) |
| `street_people` | Tyler A. McNeil | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Pedestrians_on_Caroline_Street_%28daytime%29%2C_Saratoga_Springs%2C_New_York.jpg) |

## Several-of-one-thing photographs

Fetched by `evals/instance_phrases.py` into `evals/_corpus/instances`.  
Used by: evals/instance_phrases.py, object_verdicts.py

| file | artist | licence | source |
|---|---|---|---|
| `bicycles` | Shixart1985 | CC BY 2.0 | — |
| `houses` | Jaggery | CC BY-SA 2.0 | — |
| `horses` | Oliver Dixon | CC BY-SA 2.0 | — |

## Eight-box segmentation set

Fetched by `evals/segmentation_models/eight_boxes.py` into `evals/_corpus/eight_boxes`.  
Used by: evals/segmentation_models/eight_boxes.py, degenerate_masks.py, object_verdicts.py, instance_phrases.py

| file | artist | licence | source |
|---|---|---|---|
| `dog_shepherd` | Jakub Hałun | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:20110425_German_Shepherd_Dog_8505.jpg) |
| `kingfisher` | Alexis LOURS | CC BY 2.0 | [Commons](https://commons.wikimedia.org/wiki/File:Common_kingfisher_on_a_branch_opening_its_wings.jpg) |
| `backpack_red` | Matti Blume | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Fjallraven%2C_OutDoor_2018%2C_Friedrichshafen_%281X7A0438%29.jpg) |
| `motorcycle` | Chris Woodrich | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Honda_CMX500_Rebel_parked_at_Hogs_for_Hospice%2C_Leamington%2C_Ontario%2C_2025-08-02.jpg) |
| `boat_sunset` | Liridon | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Sailing_boat_at_sunset%2C_Ionian_Sea%2C_Albania.jpg) |
| `guitarist` | Harald Krichel | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Sam_Amidon-1160199.jpg) |
| `chairs_blue` | Dietmar Rabich | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Santorin_%28GR%29%2C_Fira_--_2017_--_2598.jpg) |
| `horse_white` | fir0002 flagstaffotos [at] gmail.com | GFDL 1.2 | [Commons](https://commons.wikimedia.org/wiki/File:White_horse_in_field.jpg) |

The `horse_white` photograph is GFDL 1.2 only, which would require the full licence text to
travel with any derivative, so its sheets are not committed; the evals regenerate them.

## Agent-loop request set

Fetched by `evals/agent_loop/cases.py` into `evals/_corpus/agent_loop`.  
Used by: evals/agent_loop/, space_aware_checks.py

| file | artist | licence | source |
|---|---|---|---|
| `door` | Marco Ober | CC BY-SA 4.0 | — |
| `taxi` | Matt Kieffer | CC BY-SA 2.0 | — |
| `cat` | Fariz Gunawan | CC BY-SA 4.0 | — |
| `kite` | Cathy Cox | CC BY-SA 2.0 | — |
| `roadworks` | Ingolfson | Public domain | — |
| `two_cars` | Fabian Musto | CC BY-SA 2.0 | — |
| `heatmap` | Lorepenoten | CC0 | — |
| `zipf` | Cosmia Nebula | CC BY-SA 4.0 | — |
| `bird` | Ssemmanda will | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Blue_Bird_on_a_Wire_%E2%80%93_Minimalist_Portrait_from_Busaabala.jpg) |
| `powerlines` | "Free for Commercial Use" (Flickr, via Commons) | CC BY-SA 2.0 | [Commons](https://commons.wikimedia.org/wiki/File:Silhouette_of_power_lines_over_landscape_(8570002788).jpg) |
| `lamppost` | TeWeBs | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Street_Light_Blue_Sky.JPG) |
| `lighthouse` | Dietmar Rabich | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Cape_Otway_(AU),_Cape_Otway_Lighthouse_--_2019_--_1197.jpg) |
| `white_car` | Pambelle12 | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:A_white_parked_car.jpg) |
| `two_dogs` | Sanskritidubey | CC BY 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Two_dogs_sitting_outdoors.jpg) |
| `two_bicycles` | Christopher Harris (chrisrossharris), via Unsplash | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Two_bicycles_near_a_house_(Unsplash).jpg) |
| `chart_bar`, `chart_scatter`, `legend_bleed`, `annotation_box` | synthetic, generated for this eval | — | built by a matplotlib script written for this run, no external source |

## Phrase-grounding audit set

Fetched by `evals/audits/audit_phrase_labels.py` into `evals/_corpus/audit_phrase`.  
Used by: evals/occlusion_phrases.py, object_verdicts.py, instance_phrases.py

| file | artist | licence | source |
|---|---|---|---|
| `bicycle` | Yellow.Cat from Roma, Italy | CC BY 2.0 | [Commons](https://commons.wikimedia.org/wiki/File:My_parked_bicycle_%287872308036%29.jpg) |
| `cat_grass` | <a href="https://pixnio.com/media/grey-domestic-cat-nose-mouth-whiskers">Photo</a> by <a href="https://pixnio.com/author/milim84">Marko Milivojevic</a> on <a href="https://pixnio.com/">Pixnio</a> | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Domestic_shorthair_cat_portrait_in_grass.jpg) |
| `sheep_fence` | Kenneth Allen | CC BY-SA 2.0 | [Commons](https://commons.wikimedia.org/wiki/File:Sheep_behind_a_wire_fence%2C_Envagh_-_geograph.org.uk_-_7289230.jpg) |
| `winter_branches` | Fons Heijnsbroek | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Urban_winter_trees_with_bare_branches_at_the_Marineterrein_-_free_photo_Amsterdam_by_Fons_Heijnsbroek_01-2022.jpg) |

## Colormap-remap audit set

Fetched by `evals/audits/audit_remap_fetch.py` into `evals/_corpus/audit_remap`.  
Used by: evals/audits/ (REMAP.md)

| file | artist | licence | source |
|---|---|---|---|
| `gcd_heatmap` | Lorepenoten | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Heatmap_of_GCD_Matrix.png) |
| `ackley_contour` | Balluwun-enjoyer | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Ackley_2d.png) |
| `photon_jet` | Jarekt | CC BY-SA 3.0 | [Commons](https://commons.wikimedia.org/wiki/File:Photon_Cross_Sections.png) |
| `photon_mfp_jet` | Jarekt | CC BY-SA 3.0 | [Commons](https://commons.wikimedia.org/wiki/File:Photon_Mean_Free_Path.png) |
| `photon_mac_jet` | Jarekt | CC BY-SA 3.0 | [Commons](https://commons.wikimedia.org/wiki/File:Photon_Mass_Attenuation_Coefficients.png) |
| `cvd_strips` | Curran919 | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:CVD-friendly_sequential_colormaps.png) |
| `collatz_fractal` | Hugo Spinelli | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Collatz_Fractal.png) |
| `wiki_depth_scatter` | GPT-5 prompted by me using the data at Tim Ocean and slightly modified python code | Public domain | [Commons](https://commons.wikimedia.org/wiki/File:Wikipedias%27_article_depth_vs_number_of_articles.png) |
| `rho_oph_scatter` | Merikanto | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Rho_ophiuchi_region_L1688_star_mass_vs_dust_disk_mass.png) |
| `hr_diagram` | Merikanto | CC BY 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:HR_diagram_2msun_feh0_1.png) |
| `gliese_turbo` | Merikanto | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Temperature_of_gliese_12b_as_locked_desert_planet_1.png) |
| `aquaplanet_turbo` | Merikanto | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Aquaplanet_temperature_distribution_1_1_1_1.png) |
| `tidal_desert_turbo` | Merikanto | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Tidally_locked_desert_planet_surface_temperature_1_1_1_1.png) |
| `cloudless_desert_turbo` | Merikanto | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Cloudless_desert_planet_surface_temperature_1_1_1_1.png) |
| `kangerlussuaq_jet` | Merikanto | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:T_surface_july_kangerlussuaq_area_2000-2021_2.png) |
| `cretaceous_jet` | Merikanto | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Cretaceous_90ma_co2_900_annual_temperature_2.png) |
| `fertile_crescent_viridis` | Merikanto | CC BY 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Fertile_crescent_precipitation_1.png) |
| `gliese_profile_viridis` | Merikanto | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Gliese_12_b_temperature_profile_if_rotating_ocean_planet_1.png) |
| `jan_rain_viridis` | Merikanto | CC BY 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:January_Rain_Middle_East_1.png) |
| `rossmo_custom` | Merikanto | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Kyllikki_saari_murder_site_rossmo_2_1_1_1.png) |
| `galactic_corr` | Merikanto | CC BY 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Galactic_disk_stellar_elements_correlation_matrix_xfe_1.png) |
| `radar_reflectivity_custom` | NWS | Public domain | [Commons](https://commons.wikimedia.org/wiki/File:Composite_Reflectivity_San_Juan_radar_92017.png) |
| `hunter_gatherer_pop` | ChatGPT 4o | Public domain | [Commons](https://commons.wikimedia.org/wiki/File:Simple_estimation_of_hunter_gatherer_population_of_ancient_near_east_from_rainfall_and_land_wetness_estimation_17000_bp_1.png) |
| `gaussian_surface` | Kopak999 | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Gaussian_2d_surface.png) |

## Published charts

Fetched by `evals/chart_remap.py` into `evals/_corpus/charts`.  
Used by: evals/chart_remap.py

| file | artist | licence | source |
|---|---|---|---|
| `gcd_heatmap` | Lorepenoten | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Heatmap_of_GCD_Matrix.png) |
| `ackley_contour` | Balluwun-enjoyer | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Ackley_2d.png) |
| `photon_jet` | Jarekt | CC BY-SA 3.0 | [Commons](https://commons.wikimedia.org/wiki/File:Photon_Cross_Sections.png) |
| `gcd` | Lorepenoten | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Heatmap_of_GCD_Matrix.png) |
| `ackley` | Balluwun-enjoyer | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/wiki/File:Ackley_2d.png) |
| `exo_scatter` | Merikanto | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Exoplanet_distance_mass_relation_by_planet_pairs_inner_to_outer_1.png) |

---

## Not included: the occlusion set

`evals/occlusion_phrases.py` reads seven images from `evals/_corpus/occlusion/` (the sheep,
cow, bicycle and bollard cases) whose sources were not recorded, so the eval needs your own
copies of the images to run.

Images credited above: 65.
