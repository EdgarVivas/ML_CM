import os
import random
from PIL import Image, ImageDraw
import math

# Set random seed for reproducibility
random.seed(42)

# Configuration
IMAGE_SIZE = (64, 64)  # Frame size is 64x64
NUM_IMAGES_PER_CLASS = 1000  # Number of images per class
OUTPUT_DIR = 'dataset_bw'  # Output directory for images

# Define classes
CLASSES = ['circle_damage', 'line_damage', 'star_damage', 'no_damage', 'multiple_damages']

# Configuration for number of shapes per damage type
SHAPES_PER_DAMAGE = {
    'circle_damage': (1, 3),  # 1 to 3 circles per image
    'line_damage': (1, 3),    # 1 to 3 lines per image
    'star_damage': (1, 3),    # 1 to 3 stars per image
}

# Create output directories
def create_directories(base_dir, classes):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

# Draw filled circle damage
def draw_circle(draw):
    radius = random.randint(5, 15)  # Adjusted for 64x64 frame size
    x = random.randint(radius, IMAGE_SIZE[0] - radius)
    y = random.randint(radius, IMAGE_SIZE[1] - radius)
    bounding_box = [x - radius, y - radius, x + radius, y + radius]
    color = 'white'
    draw.ellipse(bounding_box, outline=color, fill=color, width=2)

# Draw line damage
def draw_line(draw):
    x1 = random.randint(0, IMAGE_SIZE[0])
    y1 = random.randint(0, IMAGE_SIZE[1])
    x2 = random.randint(0, IMAGE_SIZE[0])
    y2 = random.randint(0, IMAGE_SIZE[1])
    color = 'white'
    width = random.randint(1, 2)  # Adjusted line width
    draw.line([x1, y1, x2, y2], fill=color, width=width)

# Draw star damage
def draw_star(draw):
    num_points = 5
    outer_radius = random.randint(8, 20)  # Adjusted for 64x64 frame size
    x_center = random.randint(outer_radius, IMAGE_SIZE[0] - outer_radius)
    y_center = random.randint(outer_radius, IMAGE_SIZE[1] - outer_radius)
    
    rotation_angle = random.uniform(0, 2 * math.pi)
    points = []
    angle_offset = rotation_angle - math.pi / 2
    for i in range(num_points):
        angle = angle_offset + (2 * math.pi * i) / num_points
        x = x_center + outer_radius * math.cos(angle)
        y = y_center + outer_radius * math.sin(angle)
        points.append((x, y))
    
    connect_order = [0, 2, 4, 1, 3, 0]
    color = 'white'
    width = 2
    for i in range(num_points):
        start_point = points[connect_order[i]]
        end_point = points[connect_order[i + 1]]
        draw.line([start_point, end_point], fill=color, width=width)

# Draw multiple damages
def draw_multiple(draw):
    damage_types = ['circle', 'line', 'star']
    num_damages = random.randint(2, 4)
    for _ in range(num_damages):
        damage = random.choice(damage_types)
        if damage == 'circle':
            draw_circle(draw)
        elif damage == 'line':
            draw_line(draw)
        elif damage == 'star':
            draw_star(draw)

# Generate no_damage image
def generate_no_damage():
    return Image.new('RGB', IMAGE_SIZE, color='black')

# Generate damage image with random number of shapes
def generate_damage_image(damage_type):
    image = Image.new('RGB', IMAGE_SIZE, color='black')
    draw = ImageDraw.Draw(image)
    
    if damage_type in SHAPES_PER_DAMAGE:
        min_shapes, max_shapes = SHAPES_PER_DAMAGE[damage_type]
        num_shapes = random.randint(min_shapes, max_shapes)
        for _ in range(num_shapes):
            if damage_type == 'circle_damage':
                draw_circle(draw)
            elif damage_type == 'line_damage':
                draw_line(draw)
            elif damage_type == 'star_damage':
                draw_star(draw)
    elif damage_type == 'multiple_damages':
        draw_multiple(draw)
    
    return image

# Save images to respective directories
def save_images():
    create_directories(OUTPUT_DIR, CLASSES)
    for cls in CLASSES:
        print(f"Generating images for class: {cls}")
        for i in range(NUM_IMAGES_PER_CLASS):
            if cls == 'no_damage':
                img = generate_no_damage()
            else:
                img = generate_damage_image(cls)
            img_filename = f"image_{i+1:04d}.png"
            img_path = os.path.join(OUTPUT_DIR, cls, img_filename)
            img.save(img_path)
            if (i+1) % 100 == 0:
                print(f"  Saved {i+1} images")
    print("Dataset generation complete.")

if __name__ == "__main__":
    save_images()
