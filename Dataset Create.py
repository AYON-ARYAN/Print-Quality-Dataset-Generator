import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from random import randint, choice

# Settings
output_dir = 'print_quality_dataset'
qualities = ['good', 'average', 'poor']
font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
image_count = 100  # Per category

# Create folders
for quality in qualities:
    os.makedirs(os.path.join(output_dir, quality), exist_ok=True)

# Generate images
for quality in qualities:
    for i in range(image_count):
        # Create white image
        img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size=randint(18, 30))

        # Add random text
        text = f"Sample Text {i}"
        d.text((10, 100), text, fill=(0, 0, 0), font=font)

        # Convert to OpenCV format
        img = np.array(img)

        # Apply degradation based on quality
        if quality == 'average':
            img = cv2.GaussianBlur(img, (5, 5), 1)
        elif quality == 'poor':
            img = cv2.GaussianBlur(img, (9, 9), 3)
            noise = np.random.randint(0, 50, (224, 224, 3), dtype='uint8')
            img = cv2.add(img, noise)
            img = cv2.addWeighted(img, 0.6, np.zeros_like(img), 0.4, 0)

        # Save image
        path = os.path.join(output_dir, quality, f"{quality}_{i}.jpg")
        cv2.imwrite(path, img)

print("✅ Dataset generated in 'print_quality_dataset/'")
