from PIL import Image, ImageDraw

# Texture resolution
SIZE = 256
GRID_COUNT = 3  # 3x3 grid: 9 clear tiles total

basecolor_img = Image.new("RGB", (SIZE, SIZE))
orm_img = Image.new("RGB", (SIZE, SIZE))

draw_base = ImageDraw.Draw(basecolor_img)
draw_orm = ImageDraw.Draw(orm_img)

cell_size = SIZE // GRID_COUNT
border_width = 3

# 3 discrete steps:
# Roughness (X-axis, left to right): 0.0 (Smooth), 0.5 (Medium), 1.0 (Rough)
roughness_vals = [0.0, 0.5, 1.0]

# Metallic (Y-axis, top to bottom): 1.0 (Metal), 0.5 (Semi-metal), 0.0 (Plastic/Dielectric)
metallic_vals = [1.0, 0.5, 0.0]

# Neutral light gray baseColor — NO confusing patterns or colors!
NEUTRAL_GRAY = (200, 200, 200)
BORDER_COLOR = (30, 30, 30)

for row in range(GRID_COUNT):
    for col in range(GRID_COUNT):
        x0 = col * cell_size
        y0 = row * cell_size
        x1 = (col + 1) * cell_size
        y1 = (row + 1) * cell_size

        r_val = roughness_vals[col]
        m_val = metallic_vals[row]

        # --- ORM Texture ---
        g_byte = int(round(r_val * 255))
        b_byte = int(round(m_val * 255))
        draw_orm.rectangle([x0, y0, x1, y1], fill=(255, g_byte, b_byte))

        # --- Base Color Texture (Solid Neutral Gray) ---
        draw_base.rectangle([x0, y0, x1, y1], fill=NEUTRAL_GRAY)

        # Draw grid borders
        draw_base.rectangle([x0, y0, x1 - 1, y1 - 1], outline=BORDER_COLOR, width=border_width)

# Save images to target directory
target_dir = r"d:\coding\repo\cis5650\Project3-CUDA-Path-Tracer\scenes\models\boxwithspaces_test"
basecolor_img.save(f"{target_dir}\\test_basecolor.png")
orm_img.save(f"{target_dir}\\test_orm.png")

print("Generated super simple 3x3 neutral gray test textures successfully!")
