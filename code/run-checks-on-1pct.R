library(zoneR)
library(here)

result_2_unit <- zr_run_zoning_checks(here("example-data",
                          "bldg",
                          "2_unit.bldg"),
                     here("example-data",
                          "parcel",
                          "sampled-parcels"),
                     here("example-data",
                          "zoning",
                          "all"),
                     save_to = here("example-data",
                                    "results",
                                    "2_unit.geojson"))

result_4_wide <- zr_run_zoning_checks(here("example-data",
                          "bldg",
                          "4_unit_wide.bldg"),
                     here("example-data",
                          "parcel",
                          "sampled-parcels"),
                     here("example-data",
                          "zoning",
                          "all"),
                     save_to = here("example-data",
                                    "results",
                                    "4-unit-wide.geojson"))

result_4_tall <- zr_run_zoning_checks(here("example-data",
                                           "bldg",
                                           "4_unit_tall.bldg"),
                                      here("example-data",
                                           "parcel",
                                           "sampled-parcels"),
                                      here("example-data",
                                           "zoning",
                                           "all"),
                                      save_to = here("example-data",
                                                     "results",
                                                     "4-unit-tall.geojson"))

result_12_unit <- zr_run_zoning_checks(here("example-data",
                                           "bldg",
                                           "12_unit.bldg"),
                                      here("example-data",
                                           "parcel",
                                           "sampled-parcels"),
                                      here("example-data",
                                           "zoning",
                                           "all"),
                                      save_to = here("example-data",
                                                     "results",
                                                     "12_unit.geojson"))

muni_summ_12 <- zr_summary_by_muni(result_12_unit)

muni_summ_2 <- zr_summary_by_muni(result_2_unit)
