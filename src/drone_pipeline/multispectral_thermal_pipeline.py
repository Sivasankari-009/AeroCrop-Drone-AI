import os
import json
import numpy as np
import pandas as pd
import cv2

# ============================================================
# AEROCROP AI
# Multispectral + Thermal Drone Processing Pipeline
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "multispectral"
)

SEGMENTATION_DIR = os.path.join(
    BASE_DIR,
    "data",
    "segmentation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEGMENTATION_DIR, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 256

BANDS = [
    "blue",
    "green",
    "red",
    "red_edge",
    "nir",
    "thermal"
]


# ============================================================
# Utility
# ============================================================

def normalize_band(band):
    band = band.astype(np.float32)

    minimum = band.min()
    maximum = band.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(band)

    return (band - minimum) / (
        maximum - minimum
    )


def create_synthetic_multispectral_scene():
    """
    Creates a prototype multispectral + thermal
    drone scene.

    This is synthetic data for prototype validation.
    """

    rng = np.random.default_rng(42)

    h = IMAGE_SIZE
    w = IMAGE_SIZE

    y, x = np.mgrid[0:h, 0:w]

    # --------------------------------------------------------
    # Base agricultural field pattern
    # --------------------------------------------------------

    field_pattern = (
        0.5
        + 0.25 * np.sin(x / 18)
        + 0.20 * np.cos(y / 23)
    )

    field_pattern = np.clip(
        field_pattern,
        0,
        1
    )

    # --------------------------------------------------------
    # Crop stress region
    # --------------------------------------------------------

    stress_region = (
        ((x - 75) ** 2) / (42 ** 2)
        + ((y - 170) ** 2) / (30 ** 2)
        < 1
    )

    # --------------------------------------------------------
    # Pest region
    # --------------------------------------------------------

    pest_region = (
        ((x - 190) ** 2) / (35 ** 2)
        + ((y - 80) ** 2) / (40 ** 2)
        < 1
    )

    # --------------------------------------------------------
    # Water / irrigation region
    # --------------------------------------------------------

    water_region = (
        ((x - 135) ** 2) / (50 ** 2)
        + ((y - 215) ** 2) / (25 ** 2)
        < 1
    )

    # --------------------------------------------------------
    # Spectral bands
    # --------------------------------------------------------

    blue = (
        0.30
        + 0.15 * field_pattern
        + rng.normal(0, 0.025, (h, w))
    )

    green = (
        0.45
        + 0.20 * field_pattern
        + rng.normal(0, 0.025, (h, w))
    )

    red = (
        0.42
        + 0.16 * field_pattern
        + rng.normal(0, 0.025, (h, w))
    )

    red_edge = (
        0.52
        + 0.25 * field_pattern
        + rng.normal(0, 0.025, (h, w))
    )

    nir = (
        0.68
        + 0.24 * field_pattern
        + rng.normal(0, 0.025, (h, w))
    )

    # --------------------------------------------------------
    # Stress / pest spectral changes
    # --------------------------------------------------------

    red[stress_region] += 0.12
    nir[stress_region] -= 0.18
    red_edge[stress_region] -= 0.12

    red[pest_region] += 0.18
    nir[pest_region] -= 0.22
    red_edge[pest_region] -= 0.16

    # Healthy/wet area
    nir[water_region] += 0.08
    red[water_region] -= 0.04

    # --------------------------------------------------------
    # Thermal band
    # --------------------------------------------------------

    thermal = (
        29.0
        + 2.5 * stress_region.astype(float)
        + 3.5 * pest_region.astype(float)
        - 2.0 * water_region.astype(float)
        + rng.normal(0, 0.35, (h, w))
    )

    bands = {
        "blue": normalize_band(blue),
        "green": normalize_band(green),
        "red": normalize_band(red),
        "red_edge": normalize_band(red_edge),
        "nir": normalize_band(nir),
        "thermal": thermal.astype(np.float32)
    }

    return bands, stress_region, pest_region, water_region


# ============================================================
# Vegetation Indices
# ============================================================

def calculate_indices(bands):

    red = bands["red"]
    red_edge = bands["red_edge"]
    nir = bands["nir"]

    ndvi = (
        (nir - red)
        / (nir + red + 1e-8)
    )

    ndre = (
        (nir - red_edge)
        / (nir + red_edge + 1e-8)
    )

    return ndvi, ndre


# ============================================================
# Field Zone Segmentation
# ============================================================

def segment_field_zones(ndvi, thermal):

    # Healthy vegetation generally has stronger NDVI.
    vegetation_mask = (
        ndvi > np.percentile(
            ndvi,
            35
        )
    ).astype(np.uint8)

    # Smooth noise
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    vegetation_mask = cv2.morphologyEx(
        vegetation_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    vegetation_mask = cv2.morphologyEx(
        vegetation_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Connected field regions
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            vegetation_mask,
            connectivity=8
        )
    )

    # Keep meaningful regions
    zone_mask = np.zeros_like(
        labels,
        dtype=np.int32
    )

    zone_id = 0

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < 100:
            continue

        zone_id += 1

        zone_mask[
            labels == label
        ] = zone_id

    return zone_mask


# ============================================================
# Zone Feature Extraction
# ============================================================

def extract_zone_features(
    zone_mask,
    ndvi,
    ndre,
    thermal
):

    records = []

    zone_ids = np.unique(
        zone_mask
    )

    for zone_id in zone_ids:

        if zone_id == 0:
            continue

        mask = (
            zone_mask == zone_id
        )

        pixel_count = int(
            mask.sum()
        )

        mean_ndvi = float(
            ndvi[mask].mean()
        )

        mean_ndre = float(
            ndre[mask].mean()
        )

        mean_thermal = float(
            thermal[mask].mean()
        )

        if mean_ndvi < 0.25:
            status = "Severe Stress"

        elif mean_ndvi < 0.45:
            status = "Crop Stress"

        else:
            status = "Healthy"

        records.append(
            {
                "zone_id": f"DRONE-ZONE-{zone_id:03d}",
                "pixel_count": pixel_count,
                "mean_ndvi": round(
                    mean_ndvi,
                    4
                ),
                "mean_ndre": round(
                    mean_ndre,
                    4
                ),
                "mean_thermal_temperature": round(
                    mean_thermal,
                    2
                ),
                "vegetation_status": status
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# Save Visualization
# ============================================================

def save_visualizations(
    bands,
    ndvi,
    ndre,
    zone_mask
):

    # --------------------------------------------------------
    # RGB composite
    # --------------------------------------------------------

    rgb = np.stack(
        [
            bands["red"],
            bands["green"],
            bands["blue"]
        ],
        axis=2
    )

    rgb = np.clip(
        rgb * 255,
        0,
        255
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "drone_rgb_composite.png"
        ),
        cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )
    )

    # --------------------------------------------------------
    # NIR visualization
    # --------------------------------------------------------

    nir_img = (
        normalize_band(
            bands["nir"]
        ) * 255
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "nir_band.png"
        ),
        nir_img
    )

    # --------------------------------------------------------
    # Thermal visualization
    # --------------------------------------------------------

    thermal_img = (
        normalize_band(
            bands["thermal"]
        ) * 255
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "thermal_map.png"
        ),
        thermal_img
    )

    # --------------------------------------------------------
    # NDVI visualization
    # --------------------------------------------------------

    ndvi_img = (
        normalize_band(
            ndvi
        ) * 255
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "ndvi_map.png"
        ),
        ndvi_img
    )

    # --------------------------------------------------------
    # NDRE visualization
    # --------------------------------------------------------

    ndre_img = (
        normalize_band(
            ndre
        ) * 255
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "ndre_map.png"
        ),
        ndre_img
    )

    # --------------------------------------------------------
    # Segmentation mask
    # --------------------------------------------------------

    segmentation_img = (
        zone_mask.astype(np.float32)
        / max(
            int(zone_mask.max()),
            1
        )
        * 255
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            SEGMENTATION_DIR,
            "field_zone_segmentation.png"
        ),
        segmentation_img
    )


# ============================================================
# Main Pipeline
# ============================================================

def main():

    print("=" * 60)
    print(
        "AeroCrop AI - Multispectral + Thermal Drone Pipeline"
    )
    print("=" * 60)

    print("\n[1/6] Generating prototype drone scene...")

    bands, stress_mask, pest_mask, water_mask = (
        create_synthetic_multispectral_scene()
    )

    print(
        "Bands:",
        ", ".join(BANDS)
    )

    print("\n[2/6] Calculating vegetation indices...")

    ndvi, ndre = calculate_indices(
        bands
    )

    print(
        f"NDVI range: "
        f"{ndvi.min():.3f} → {ndvi.max():.3f}"
    )

    print(
        f"NDRE range: "
        f"{ndre.min():.3f} → {ndre.max():.3f}"
    )

    print("\n[3/6] Segmenting field zones...")

    zone_mask = segment_field_zones(
        ndvi,
        bands["thermal"]
    )

    zone_count = int(
        zone_mask.max()
    )

    print(
        f"Field zones detected: {zone_count}"
    )

    print("\n[4/6] Extracting zone-level features...")

    zone_df = extract_zone_features(
        zone_mask,
        ndvi,
        ndre,
        bands["thermal"]
    )

    zone_csv = os.path.join(
        SEGMENTATION_DIR,
        "drone_field_zone_features.csv"
    )

    zone_df.to_csv(
        zone_csv,
        index=False
    )

    print(
        f"Zone feature file: {zone_csv}"
    )

    print("\n[5/6] Saving drone products...")

    np.savez_compressed(
        os.path.join(
            OUTPUT_DIR,
            "aerocrop_multispectral_thermal_scene.npz"
        ),
        blue=bands["blue"],
        green=bands["green"],
        red=bands["red"],
        red_edge=bands["red_edge"],
        nir=bands["nir"],
        thermal=bands["thermal"],
        ndvi=ndvi,
        ndre=ndre
    )

    save_visualizations(
        bands,
        ndvi,
        ndre,
        zone_mask
    )

    print("\n[6/6] Saving metadata...")

    metadata = {
        "project": "AeroCrop Drone AI",
        "pipeline": (
            "Multispectral + Thermal "
            "Drone Processing"
        ),
        "bands": BANDS,
        "image_size": [
            IMAGE_SIZE,
            IMAGE_SIZE
        ],
        "vegetation_indices": [
            "NDVI",
            "NDRE"
        ],
        "segmentation": (
            "Vegetation-based connected-component "
            "field-zone segmentation"
        ),
        "zone_count": zone_count,
        "data_type": (
            "Synthetic prototype data "
            "for end-to-end validation"
        )
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "pipeline_metadata.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nMultispectral bands processed: "
        f"{len(BANDS) - 1}"
    )

    print(
        "Thermal layer processed: YES"
    )

    print(
        f"Field zones segmented: {zone_count}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    print(
        f"Segmentation directory:\n"
        f"{SEGMENTATION_DIR}"
    )


if __name__ == "__main__":
    main()