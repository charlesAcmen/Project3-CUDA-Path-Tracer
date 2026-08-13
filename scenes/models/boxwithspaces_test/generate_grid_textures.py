from PIL import Image, ImageDraw

# Texture resolution
SIZE = 256
GRID_COUNT = 4  # 4x4 grid of material combinations

# Create images
basecolor_img = Image.new("RGB", (SIZE, SIZE))
orm_img = Image.new("RGB", (SIZE, SIZE))

draw_base = ImageDraw.Draw(basecolor_img)
draw_orm = ImageDraw.Draw(orm_img)

cell_size = SIZE // GRID_COUNT
border_width = 2

# Discrete values for 4x4 grid:
# Roughness (X-axis, left to right): 0.0, 0.33, 0.67, 1.0
roughness_vals = [0.0, 0.333, 0.667, 1.0]

# Metallic (Y-axis, top to bottom): 1.0, 0.67, 0.33, 0.0
metallic_vals = [1.0, 0.667, 0.333, 0.0]

# Base colors: warm gold/copper for metals, rich cyan/orange for dielectrics
# We use a 2x2 checkerboard inside each 4x4 cell plus dark borders
color_a = (240, 190, 60)   # Gold / Warm Orange
color_b = (60, 140, 240)   # Cool Blue / Cyan
border_color = (20, 20, 20)

for row in range(GRID_COUNT):
    for col in range(GRID_COUNT):
        x0 = col * cell_size
        y0 = row * cell_size
        x1 = (col + 1) * cell_size
        y1 = (row + 1) * cell_size

        r_val = roughness_vals[col]
        m_val = metallic_vals[row]

        # --- ORM Texture ---
        # R = Occlusion (1.0 = 255)
        # G = Roughness (r_val * 255)
        # B = Metallic (m_val * 255)
        g_byte = int(round(r_val * 255))
        b_byte = int(round(m_val * 255))
        draw_orm.rectangle([x0, y0, x1, y1], fill=(255, g_byte, b_byte))

        # --- Base Color Texture ---
        # Draw 2x2 micro-checkerboard inside the cell
        half = cell_size // 2
        draw_base.rectangle([x0, y0, x0 + half, y0 + half], fill=color_a)
        draw_base.rectangle([x0 + half, y0, x1, y0 + half], fill=color_b)
        draw_base.rectangle([x0, y0 + half, x0 + half, y1], fill=color_b)
        draw_base.rectangle([x0 + half, y0 + half, x1, y1], fill=color_a)

        # Draw cell border on both textures for ultra-clear grid separation
        draw_base.rectangle([x0, y0, x1 - 1, y1 - 1], outline=border_color, width=border_width)
        draw_orm.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, g_byte, b_byte), width=border_width)

# Save images to target directory
target_dir = r"d:\coding\repo\cis5650\Project3-CUDA-Path-Tracer\scenes\models\boxwithspaces_test"
basecolor_img.save(f"{target_dir}\\test_basecolor.png")
orm_img.save(f"{target_dir}\\test_orm.png")

print("Generated 4x4 discrete block test textures successfully!")
