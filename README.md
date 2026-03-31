# Open Zoning Feed Specification

The Open Zoning Feed Specification (OZFS) is offered as a scalable,
extensible data schema for encoding residential zoning regulations at
the parcel level in a way that is both machine-readable and
jurisdiction-agnostic. OZFS is implemented in JSON and GeoJSON, a geographic
extension of JSON that retains JSON’s nested flexibility. OZFS departs
from legacy tabular formats by storing only those variables actually
defined in a district and permitting new key–value pairs without
changing the underlying schema.

OZFS makes six specific contributions:

1.  Non-tabular structure. A hierarchical JSON model represents
    districts and optional constraints, eliminating
    column inflation and redundant data entry.

2.  Jurisdiction-specific definitions. Each feed contains an explicit
    glossary for locally defined terms—e.g., townhouse, multifamily,
    building height—so that uniform semantics are not imposed.

3.  Python-syntax mathematical expressions. Controls may be stored as
    formulas (`MaxHeight` = 0.5 \* `LotDepth`), allowing rules that
    reference parcel or building attributes.

4.  Variable-referencing mechanism. Constraints can point to other
    variables in the same feed, supporting compound rules and one-time
    entry of shared values.

5.  Embedded conditional logic. The schema accommodates
    context-dependent provisions
    (`if BuildingSetback ≥ 15 then MaxHeight = 35 else 30`) that
    spreadsheets cannot capture consistently.

6.  Standards for describing parcels and buildings. While previous
    efforts have focused on encoding characteristics of zoning
    regulations, we argue that the usefulness of that data depends on
    the interactions among characteristics of individual parcels and
    proposed buildings. OZFS includes rules for consistently encoding
    information about regulations, buildings, and parcels.

# OZFS Data Structures

The OZFS data standard includes three files types:

-   a file (following a GeoJSON format) with a \*.zoning extension to describe the zoning regulations
    for a particular municipality;

-   a file (following a GeoJSON format) with a \*.parcel extension to describe the geometry of all
    parcels (or all parcels of interest) within a municipality; and

-   a file (following a JSON format) with a \*.building extension to describe the geometry of a
    proposed building.

## Zoning regulations

Zoning data are encoded in a separate \*.zoning file for each
municipality. The \*.zoning file is formatted as a geojson file, which
offers the advantages of a nested data structure. The specific structure
of the \*.zoning file is illustrated is illustrated in outline form
below.

-   `[filename.zoning]`
    -   `Type: "FeatureCollection"`
    -   `version: "0.5.0"`
    -   `muni_name:` (required)
    -   `date:` (required)
    -   `source:` (required)
    -   `definitions:` (required) *(structured array, see further detail
        below)*
    -   `features:` (required) *(structured array, see further detail
        below)*

The top level of the file is an array with six key-value pairs:

-   `Type`: As for geojson files, the value for this key should be
    “FeatureCollection.”
-   `version`: The version of the OZFS data standard used in this file.
    The current version of the standard is 0.5.0.
-   `muni_name`: The name of the municipality this zoning code refers
    to. This is a required value.
-   `date`: The most recent date on which the zoning regulations are
    known to have been in effect. This is a required value.
-   `source`: This is a url or other note indicating where the information in this
    file comes from. In general, this will be an online municipal code.
-   `definitions` is an array of definitions of terms (in the current
    version of the standard, height and residential building types) that
    may very from one municipality to the next and are defined in the
    text of the zoning code. These are defined in a structured array
    described in greater detail below.
-   `features` contains information on each zoning district. As for
    definitions, these are defined in a structured array described in
    greater detail below.

The features array, which describes each zoning district, and the
definitions array, which contains municipality-specific definitions of
terms, are described in greater detail below.

### District-level information in the features array

The features array represents each district by an array of three
elements: `type`, `properties`, and `geometry`. As for geojson files,
the value for the `type` key will be “Feature” and the geometry key
takes an array of coordinates describing the geometry of the feature (in
this case, the district boundary). The structure of the features array
is illustrated in outline form below.

-   `features`
    -   `[[1]]`
        -   `type: "Feature"`
        -   `properties`
            -   `dist_name:` (optional)
            -   `dist_abbr:` (required)
            -   `planned_dev:` (optional)
            -   `overlay:` (optional)
            -   `res_types_allowed:` (conditionally required)
            -   `constraints`
        -   `geometry`
    -   `...`
    -   `[[n]]`
        -   `type: "Feature"`
        -   `properties`
        -   `geometry`

The value for the `properties` key is a list of key-value pairs that may
include the following:

-   `dist_name` is the name of district. This value is optional.

-   `dist_abbr` is the abbreviated name of district. This value is
    required.

-   `planned_dev` is a binary value indicating whether this is a planned
    development district (where the entire district will be developed by
    a single developer who negotiates constraints directly with the
    municipality). This value is optional and is assumed to have a value
    of false if it is missing.

-   `overlay` indicates whether this district is an overlay district (a
    district that modifies the requirements of any base districts it
    overlaps) and (optionally) how the overlay district regulations
    relate to those of the overlapping base district(s). This value is
    optional and should only be present for overlay districts. In
    overlay districts, the value of the overlay key should be one of the
    following:

    -   `"TRUE"` indicates that it is an overlay district and that no
        other information is available (the data standard does not
        require complete information on overlay districts).

    -   `"restrict"` indicates that the overlay district further
        restricts the requirements of the base district. In other words,
        when there is a conflict between the requirements of the base
        district and the requirements of the overlay district, the more
        restrictive of the two requirements applies.

    -   `"relax"` indicates that the overlay district relaxes the
        requirements of the base district: when there is a conflict
        between the requirements of the base district and the
        requirements of the overlay district, the less restrictive of
        the two requirements applies.

    -   `"replace"` indicates that the requirements of the overlay
        district replace those of the base districts. In other words,
        when there is a conflict between the requirements of the overlay
        district and the base district, the requirements of other
        overlay district applies (regardless of whether they relax or
        restrict the requirements of the base district).

    -   `"no_residential_effect"` indicates that the overlay district
        would have no effect on residential developments. As an example,
        Dallas has overlay districts that prohibit the sale of alcohol,
        but are not relevant to the question of siting multifamily
        housing.

    -   `"demolition_only"` indicates that the overlay district places
        restrictions on what can be demolished, but not on what can be
        built. Historic preservation districts may fall into this
        category.

    -   `"none-by-right"` indicates that any development within the
        overlay district requires discretionary approval. These are
        often (but not always) planned development overlay districts.

-   `res_types_allowed` is a list of residential land uses that are
    allowed in the district. All values in the list must also appear in
    the `definitions` array. If this list of values is missing for a
    base district, it is assumed that no residential uses are allowed by
    right in the district.

-   `non_res_allowed` is a binary variable indicating that non-residential uses
    are allowed in the district.
-   `constraints` is an array of constraints that define allowable
    building characteristics. The `constraints` is not necessary for
    planned development districts or for overlay districts that have a
    value other than “relax”, “restrict”, or “replace” for the overlay
    key.

Constraints are described by numeric values or expressions (in Python
syntax) and are stored in the structure illustrated below in outline
form.

-   `constraints`
    -   `[[constraint name 1]]`
        -   `min_val (conditionally required)`
            -   `[[1]]`
                -   `expression: (required)`
                -   `condition: (conditionally required)`
                -   `min_max: (conditionally required)`
            -   `...`
            -   `[[n]]`
        -   `max_val (conditionally required)`
            -   `[[1]]`
                -   `expression: (required)`
                -   `condition: (conditionally required)`
                -   `min_max: (conditionally required)`
            -   `...`
            -   `[[n]]`
    -   `...`
    -   `[[constraint name n]]`

Possible constraint names include:

-   `setback_front`: The front setback, in units of feet
-   `far`: The floor area ratio
-   `setback_side_int`: The interior side setback, in units of feet
-   `setback_side_ext`: The exterior side setback (for corner lots) in
    units of feet.
-   `lot_cov_bldg`: the percentage of the lot area covered by buildings,
    expressed as whole-number percentage points.

`setback_front` (the front setback), `far` (the floor area ratio), and
`lot_cov_bldg` (the lot coverage). [Appendix
A](https://github.com/vibe-lab-gsd/ozfs-standard/blob/main/appendices/appendix-a.md)
includes a complete list of constraints that have been defined for the
\*.zoning file, together with their descriptions. For each constraint
that is included in the constraints array, a minimum value `min_val`
and/or a maximum value `max_val` must be given.

Minimum and maximum values for constraints are stored as arrays
including the following key-value pairs:

-   `condition`: The condition under which the minimum or maximum value
    applies. This key is required if there is more than one element in
    the min_value (or max_value) array. The condition can be a logical
    expression (in Python syntax) defining the condition under which the
    minimum (or maximum) value applies. Some of the variables that may
    be use in constraint and condition expressions are:

    -   `lot_width`: The width of the parcel, in feet.
    -   `lot_area`: The area of the parcel, in acres.
    -   `height`: The building height, in feet.

[Appendix B](https://github.com/vibe-lab-gsd/ozfs-standard/blob/main/appendices/appendix-b.md)
includes a full list of the of the variables that can be used in
constraint and condition expressions, along with a description of
each variable. If the condition under which the value applies cannot
be described as a logical expression (one that evaluates to True or
False) with one or more of those variables, it may be described in a
text string (which will limit machine-readability).

-   `expression`: These can either be constant numeric values or
    equations (in Python syntax) referring to variables listed in
    [Appendix B](https://github.com/vibe-lab-gsd/ozfs-standard/blob/main/appendices/appendix-b.md).
    This can be a list of multiple values or expressions, in which case
    the `min_max` key should be used to specify whether the minimum or
    maximum value in the list should be used. If the value of the
    `condition` key is a text string that is not a logical expression,
    the value of the `expression` key may be a list of numbers (where
    the text string will describe the circumstances in which each number
    applies).

-   `min_max`: This key is required if the list of expressions has a
    more than one element in it and `condition` is a logical expression
    (rather than just a free-form text string). It is a character string
    that can take one of two values: `min` or `max`. A value of `min`
    indicates that the governing constraint is the minimum of the
    possible values listed in the `expression` key. A value of `maximum`
    indicates that the governing constraint is the maximum of the
    possible values listed in the `expression` key.

The four examples below illustrate how zoning code text can be stored in
the \*.zoning file.

**Example 1: A single constraint value.** In Dallas, the minimum is side
setback is specified for agricultural districts as follows:

> *Minimum side yard is 20 feet.* (City of Dallas 2024)

This requirement could be added to the constraints array in the
\*.zoning file as follows:

-   `constraints`
    -   `setback_side_int`
        -   `min_val`
            -   `[[1]]`
                -   `expression: "20"`

**Example 2: Using min_max field.** For the Cockrell Hill Single-Family
District, the minimum side setback depends on the length of the front
footage.

> *No structure shall be closer to a side or rear lot line than five
> feet or a distance equal to 10% of the front footage of the lot,
> whichever distance shall be greater* (City of Cockrell Hill 2010, 13)

This requires a list of expressions for the minimum side and rear
setbacks where the selected value should be the greater of the result of
the two expressions. This requirement could be added to the constraints
array in the \*.zoning file as follows:

-   `constraints`
    -   `setback_side_int`
        -   `min_val`
            -   `[[1]]`
                -   `expression: [ "5", "0.1 * lot_width" ]`
                -   `min_max: "max"`

**Example 3: Multiple conditions.** **?@fig-const-ex-3** from the Fort
Worth Zoning Ordinance (City of Fort Worth, Texas 2007) shows how a
district’s setback requirements are recorded when the value depends on
the building height.

> *The height of a building in the “A” through “F” districts may be
> increased when the front, side and rear yard dimensions are each
> increased above the minimum requirements by one foot for each foot
> such building exceeds the height limit of the district in which it is
> located.*

This requirement could be added to the constraints array in the
\*.zoning file as follows:

-   `constraints`
    -   `setback_side_int`
        -   `min_val`
            -   `[[1]]`
                -   `condition: "height <= 35"`
                -   `expression: "25"`
            -   `[[2]]`
                -   `condition: "height > 35"`
                -   `expression: "25 + (height - 35)"`

**Example 4: Unencodable conditions** 

To be added later.

### Definitions

There may be terms that are used in many different zoning codes, but
with definitions that vary across municipalities. The current version of
the OZFS standard (version 0.5.0) requires definitions for height and
for types of residential buildings. Other definitions may be added to
future extensions of the standard.

-   `definitions`
    -   `height`
        -   `[[1]]`
            -   `condition:`
            -   `expression:`
        -   `...`
        -   `[[n]]`
    -   `res_type`
        -   `[[1]]`
            -   `condition`
            -   `expression`
        -   `...`
        -   `[[n]]`

For each definition, one or more arrays comprising conditions and
expressions can be defined. The value of the `condition` key defines the
circumstance under which the value of the `expression` key applies and
should be a logical statement (one that returns a value of true or
false) in Python syntax, referencing any of the variable names listed in
[Appendix
B](https://github.com/vibe-lab-gsd/ozfs-standard/blob/main/appendices/appendix-b.md).
The value of the `expression` key should be an equation (in Python
syntax) referencing any of the variable names listed in [Appendix
B](https://github.com/vibe-lab-gsd/ozfs-standard/blob/main/appendices/appendix-b.md). 
Conditions do not necessarily need to be mutually exclusive. When they
are not, they are applied in the order in which they appear.

If the text of the zoning code does not include an explicit definition for 
building height, the following default definition should be used:

-   `height`
    -   `[[1]]`
        -   `condition: "roof_type == 'flat'"`
        -   `expression: "height_plate"`
    -   `[[2]]`
        -   `condition: "roof_type != 'flat'"`
        -   `expression: "(height_top + heigth_eave) / 2"`
        
In this default definition, if the height of a building is defined as the top of the
highest wall plate for buildings with a flat roof type and the mid-point
between the top of the roof and the eave for all roof types except a
flat roof (see [Appendix
C](https://github.com/vibe-lab-gsd/ozfs-standard/blob/main/appendices/appendix-c.md)
for an illustration of various roof types).

Definitions should not be included for residential types that are not referenced
in any district. All residential types that are referenced in a district should
be defined in the definitions array.

If the text of the zoning code does not include explicit definitions for any
residential types that are included as allowable residential uses in the features 
array, a set of default definitions should be included for each of those 
residential types, based on commonly-used understandings of those terms. 
For example, the following definitions would be appropriate for a municipality
with districts that allow single-family, duplex, townhouse, and multifamily
residential types.

For example, if the residential building type (`res_type`) of a building
with three or more units is defined as `multifamily` building unless all
units have outside entrances on the ground level, in which case it is
defined as a `townhouse`, this could be encoded as shown below. 

-   `res_type`
    -   `[[1]]`
        -   `condition: "total_units ==1"`
        -   `expression: "single-family`
    -   `[[2]]`
        -   `condition: "total_units == 2"`
        -   `expression: "duplex"`
    -   `[[3]]`
        -   `condition: "n_outside_entry == total_units and n_ground_entry == total_units and total_units > 2"`
        -   `expression: "townhouse`
    -   `[[4]]`
        -   `condition: "total_units > 2"`
        -   `expression: "multifamily`



## Parcel geometry

Parcel geometry data representing parcel boundaries as polygons are
commonly available in GIS files from state, county, or municipal
open-data portals. These require pre-processing for zoning analysis
because applying required setbacks to determine the buildable area of a
parcel requires information not only about the shape and location of the
parcel, but also about its orientation with respect to the street, since
zoning codes may specify difference setbacks for the front, sides, and
rear of a parcel, respectively.

In the OZFS data standard, parcels must be represented in a geojson file
that includes, for each parcel, line strings representing each parcel
edge (front, back, and side(s) and a point representing the parcel
centroid. All features have a `parcel_id` key with a value that uniquely
identifies which parcel each edge or centroid is associated with.

Each feature in the parcel dataset will also have a key, `side` , that
can take one of six values:

-   `front` indicates that this is a line string representing the front
    of a parcel.
-   `rear` indicates that this is line string representing the rear of a
    parcel.
-   `interior side` indicates that this is a line string representing
    the interior side of a parcel (the side adjacent to another parcel).
-   `exterior side` indicates that this is a line string representing
    the exterior side of a parcel. Only a corner lot can have an
    exterior side. This is the side of a corner lot that is adjacent
    (and approximately parallel) to the street that is not indicated in
    the parcel’s address.
-   `unknown` indicates that this is a line string representing the side
    of a parcel that has not been classified into one of the above
    categories (for irregular parcel geometries and/or parcels where the
    relationship to the adjacent street network is unclear).
-   `centroid` indicates that this is a point representing the parcel’s
    centroid.

Parcel centroids have four additional key/value pairs:

-   `lot_width` indicates the width of the parcel in feet.
-   `lot_depth` indicates the depth of the parcel in feet.
-   `lot_area` indicates the area of the parcel in acres.
-   `vacant` takes a value of True if the lot is vacant (if the lot is not vacant, 
    this key can be omitted or it can take a value of False).

In addition to parcel geometry features, the \*.parcel file must also
include a `version` key indicating what version of the OZFS standard the
file is consistent with. The version described here is 0.5.0.

## Building characteristics

Building characteristics for a single building are stored in json file
with the structure illustrated below.

-   `[filename].bldg`
    -   `bldg_info`
        -   `height_top:` (required)
        -   `height_plate:` (required)
        -   `height_eave:` (conditionally required)
        -   `height_deck:` (conditionally required)
        -   `height_parapet:` (optional)
        -   `height_tower:` (optional)
        -   `roof_type:` (required)
        -   `roof pitch:` (conditionally required)
        -   `width:` (required)
        -   `depth:` (required)
        -   `sep_platting:` (conditionally required)
        -   `unit_separation:` (conditionally required)
        -   `sep_wall_length:` (conditionally required)
        -   `parking:` (optional)
    -   `level_info`
        - `[[1]]`
            -   `level:` (required)
            -   `gross_fl_area:` (required)
        -   `...`
        -   `[[n]]`
            -   `level:` (required)
            -   `gross_fl_area:` (required)
    -   `unit_info`
        -   `[[1]]`
            -   `fl_area:` (required)
            -   `bedrooms:` (required)
            -   `entry_level:` (required)
            -   `outside_entry:` (required)
            -   `qty:` (required)

The file includes three arrays:

-   `bldg_info` includes information on the characteristics of the
    overall building (building dimensions and number of parking spaces
    within the structure).
-   `unit_info` includes information in each type of unit within the
    building, and
-   `level_info` contains information on each level within the building.


### Building dimensions

All building dimensions are in feet. The building information array
includes the height from the ground to the top of the building
(`height_top`), from the ground to the highest wall plate
(`height_plate`), as well as the building `width`, the building `depth`,
and the building’s roof type (`roof_type`). Refer to [Appendix
C](https://github.com/vibe-lab-gsd/ozfs-standard/blob/main/appendices/appendix-c.md)
for an illustration of roof types that are defined for use in OZFS.

As noted in the section on zoning constraints, there are differences
among zoning codes with regards to how a building’s height is defined
for various roof types. For roof types other than a flat roof, the eave
height must be specified in the `height_eave` key, and the roof pitch
must be specified in the `roof_pitch` key. For Mansard roofs,
`height_deck` must be specified as well. For roofs with a parapet, the
`height_parapet`, should also be added.

If the building includes towers, chimneys, antennas, or mechanical
structures, the (maximum) height of these (from the roof) can optionally
be specified with the `height_tower` key.

The `sep_platting` key is used to indicate whether each until in the
building would be on a separately platted parcel and the
`unit_separation` key describes how units are separated (by a
“party_wall”, a “fire_wall”, or an “open_area”). If units are separated
by a party wall or fire wall, the `sep_wall_length` key stores the
length of the separation wall. These keys may be used to determine
whether the building meet’s a municipality’s definition of a townhome.
There is also an optional `parking` key to indicate the number of
parking spaces contained within the building’s structure (i.e. in a
garage).

### Level information

The level array includes, for each level of the building, a two-element
array with the level number and the gross floor area (in square feet) of
that level. Above-ground levels are numbered with positive sequential
numbers beginning at one (for the lowest above-ground level), and
below-ground level are numbered with numbers decreasing from negative
one (for the level closest to the ground).

### Unit information

The unit array includes an array specifying the following
characteristics for each unit type, where units are classified as being
of the same type if they have the same values for each characteristic
below:

-   `fl_area`: The floor area of the unit in square feet.
-   `bedrooms`: The number of bedrooms in the unit, expressed as a whole
    number with a minimum value of zero (for a studio unit).
-   `entry_level`: The level number the entrance to the unit is on.
-   `outside_entry`: A binary value indicating whether the entrance to
    the unit is directly from the outside of the building.

In addition to the four characteristics above, there is also a `qty` key
to indicate how many units of each type are in the building.

# Sample Dataset

We'll fill this in later.

