import os
from PIL import Image

output_dir = os.path.dirname(os.path.abspath(__file__))

# 1. test_basecolor.png (64x64, RGB, sRGB)
# 8x8 grid of 8x8 squares: even squares = warm orange (255, 180, 50), odd squares = cool blue (50, 120, 255)
base_img = Image.new('RGB', (64, 64))
base_pixels = base_img.load()
orange = (255, 180, 50)
blue = (50, 120, 255)

for y in range(64):
    for x in range(64):
        grid_x = x // 8
        grid_y = y // 8
        if (grid_x + grid_y) % 2 == 0:
            base_pixels[x, y] = orange
        else:
            base_pixels[x, y] = blue

basecolor_path = os.path.join(output_dir, 'test_basecolor.png')
base_img.save(basecolor_path)

# 2. test_orm.png (64x64, RGB, linear/raw)
# R channel (occlusion): 255
# G channel (roughness): gradient 0 (left) to 255 (right)
# B channel (metallic): gradient 255 (top) to 0 (bottom)
orm_img = Image.new('RGB', (64, 64))
orm_pixels = orm_img.load()

for y in range(64):
    b_val = int(round((63 - y) * 255.0 / 63.0))
    for x in range(64):
        g_val = int(round(x * 255.0 / 63.0))
        r_val = 255
        orm_pixels[x, y] = (r_val, g_val, b_val)

orm_path = os.path.join(output_dir, 'test_orm.png')
orm_img.save(orm_path)

print('Generated files successfully:')
print('  -', basecolor_path)
print('  -', orm_path)
