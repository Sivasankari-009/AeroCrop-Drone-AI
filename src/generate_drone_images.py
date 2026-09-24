import os
import cv2
import numpy as np

# ============================================================
# AeroCrop Drone AI - Synthetic Drone Image Generator
# ============================================================

OUTPUT_DIR = "data/drone_image_dataset"

IMAGE_SIZE = 128
IMAGES_PER_CLASS = 300

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

print("=" * 70)
print("AeroCrop Drone AI - Synthetic Drone Image Dataset")
print("=" * 70)

classes = [
    "healthy",
    "crop_stress",
    "pest_zone"
]

# ------------------------------------------------------------
# Create directories
# ------------------------------------------------------------

for class_name in classes:
    os.makedirs(
        os.path.join(OUTPUT_DIR, class_name),
        exist_ok=True
    )

print("\nDataset directories created.")

# ------------------------------------------------------------
# Generate base agricultural aerial image
# ------------------------------------------------------------

def create_base_image():

    image = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8
    )

    # Soil background
    soil_color = np.random.randint(
        70,
        120
    )

    image[:, :] = (
        soil_color,
        soil_color - 10,
        soil_color - 20
    )

    # Crop rows
    row_spacing = np.random.randint(
        12,
        18
    )

    for x in range(
        5,
        IMAGE_SIZE,
        row_spacing
    ):

        for y in range(
            5,
            IMAGE_SIZE,
            8
        ):

            center_x = x + np.random.randint(
                -3,
                4
            )

            center_y = y + np.random.randint(
                -3,
                4
            )

            radius = np.random.randint(
                3,
                7
            )

            green_value = np.random.randint(
                120,
                210
            )

            cv2.circle(
                image,
                (
                    center_x,
                    center_y
                ),
                radius,
                (
                    40,
                    green_value,
                    40
                ),
                -1
            )

    return image


# ------------------------------------------------------------
# Healthy crop image
# ------------------------------------------------------------

def generate_healthy():

    image = create_base_image()

    # Add healthy green canopy
    overlay = np.zeros_like(image)

    for _ in range(
        np.random.randint(25, 45)
    ):

        x = np.random.randint(
            0,
            IMAGE_SIZE
        )

        y = np.random.randint(
            0,
            IMAGE_SIZE
        )

        radius = np.random.randint(
            5,
            12
        )

        cv2.circle(
            overlay,
            (x, y),
            radius,
            (
                np.random.randint(30, 70),
                np.random.randint(150, 220),
                np.random.randint(30, 80)
            ),
            -1
        )

    image = cv2.addWeighted(
        image,
        0.65,
        overlay,
        0.35,
        0
    )

    return image


# ------------------------------------------------------------
# Crop stress image
# ------------------------------------------------------------

def generate_crop_stress():

    image = create_base_image()

    # Yellow / brown stressed regions
    for _ in range(
        np.random.randint(8, 18)
    ):

        x = np.random.randint(
            0,
            IMAGE_SIZE
        )

        y = np.random.randint(
            0,
            IMAGE_SIZE
        )

        radius = np.random.randint(
            6,
            15
        )

        color = (
            np.random.randint(30, 90),
            np.random.randint(80, 150),
            np.random.randint(120, 190)
        )

        cv2.circle(
            image,
            (x, y),
            radius,
            color,
            -1
        )

    # Add brown patches
    for _ in range(5):

        x1 = np.random.randint(
            0,
            IMAGE_SIZE - 20
        )

        y1 = np.random.randint(
            0,
            IMAGE_SIZE - 20
        )

        x2 = x1 + np.random.randint(
            8,
            25
        )

        y2 = y1 + np.random.randint(
            8,
            25
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (
                60,
                90,
                130
            ),
            -1
        )

    return image


# ------------------------------------------------------------
# Pest zone image
# ------------------------------------------------------------

def generate_pest_zone():

    image = create_base_image()

    # Create concentrated pest hotspot
    hotspot_x = np.random.randint(
        35,
        95
    )

    hotspot_y = np.random.randint(
        35,
        95
    )

    hotspot_radius = np.random.randint(
        18,
        32
    )

    # Dark damaged vegetation
    for _ in range(
        np.random.randint(20, 35)
    ):

        angle = np.random.uniform(
            0,
            2 * np.pi
        )

        distance = np.random.uniform(
            0,
            hotspot_radius
        )

        x = int(
            hotspot_x
            + distance * np.cos(angle)
        )

        y = int(
            hotspot_y
            + distance * np.sin(angle)
        )

        radius = np.random.randint(
            3,
            8
        )

        cv2.circle(
            image,
            (x, y),
            radius,
            (
                np.random.randint(30, 80),
                np.random.randint(50, 100),
                np.random.randint(60, 110)
            ),
            -1
        )

    # Pest hotspot boundary
    cv2.circle(
        image,
        (
            hotspot_x,
            hotspot_y
        ),
        hotspot_radius,
        (
            40,
            70,
            100
        ),
        2
    )

    return image


# ------------------------------------------------------------
# Generate dataset
# ------------------------------------------------------------

generators = {
    "healthy": generate_healthy,
    "crop_stress": generate_crop_stress,
    "pest_zone": generate_pest_zone
}

total_images = 0

print("\nGenerating images...")

for class_name in classes:

    class_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    generator = generators[class_name]

    for i in range(
        IMAGES_PER_CLASS
    ):

        image = generator()

        # Add small realistic noise
        noise = np.random.normal(
            0,
            5,
            image.shape
        )

        image = np.clip(
            image.astype(np.float32)
            + noise,
            0,
            255
        ).astype(np.uint8)

        filename = os.path.join(
            class_dir,
            f"{class_name}_{i + 1:04d}.jpg"
        )

        cv2.imwrite(
            filename,
            image
        )

        total_images += 1

    print(
        f"{class_name}: "
        f"{IMAGES_PER_CLASS} images generated"
    )

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DRONE IMAGE DATASET READY")
print("=" * 70)

print(f"\nTotal images: {total_images}")
print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Classes: {len(classes)}")

print("\nClass distribution:")

for class_name in classes:

    print(
        f"- {class_name}: "
        f"{IMAGES_PER_CLASS}"
    )

print("\nDataset location:")
print(OUTPUT_DIR)

print("\nPhase 6 dataset generation completed successfully.")