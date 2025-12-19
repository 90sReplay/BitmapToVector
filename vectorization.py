import pygame
import numpy
from PIL import Image
from skimage import measure
from skimage.measure import approximate_polygon
from scipy.ndimage import binary_closing
from matplotlib.path import Path
import pygame.gfxdraw
import sys
import os

cpath = os.path.dirname(os.path.abspath(__file__))

# VARIABLES  (Adjust for each image, just mess with the values until it looks correct)
IMAGE_PATH = r"C:\Windows\System32\@WLOGO_96x96.png" # Image path obviously, Default is just a random image in system32
SCREEN_SIZE = 800                                    # Pygame window size
COLOR_BUCKET = 10                                    # Quantization bucket size (Lower colors)
ZOOM_STEP = 1.2                                      # Zoom factor per key press
MIN_REGION_PIXELS = 5.0                              # Reduce "noise" by ignoring small regions
CONTOUR_TOLERANCE = 5.0                              # Distance to round to when simplifying polygons
IMAGE_SHAPE_FILL = (5, 5)                            # Structure for binary closing (to close small gaps in shapes)

# CHECK ARGV FOR IMAGE
try:                                # If something fails, it just skips
    if len(sys.argv) == 0:              # Skips check if no arguments
        pass
    else:
        if sys.argv[1].endswith(".py"): # If called via “python script argv” it adjusts for that
            IMAGE_PATH = sys.argv[2]
        else:
            IMAGE_PATH = sys.argv[1]
except:
    pass

# STEP 1: load image
image = Image.open(IMAGE_PATH).convert("RGB")   # Use PIL to open image, and force into RGB
data = numpy.array(image)                       # Convert image to numpy array (width, height, 3) (3 because R, G, B)
height, width, color_channels = data.shape      # (data.shape is the dimensions and number of color channels)
print("Image size:", width, "x", height, "with", color_channels, "color channels")

# STEP 2: Round colors to nearest bucket
def round_color(color, bucket=COLOR_BUCKET):
    return tuple((numpy.array(color) // bucket) * bucket)
rounded_data = numpy.apply_along_axis(round_color, 2, data)   # Run function on image data with section 2 (which is color_channels)

# STEP 3: Find all unique colors in the image
unique_colors = numpy.unique(rounded_data.reshape(-1, 3), axis=0)   # What colors are left in the image after rounding?
print("Found", len(unique_colors), "unique colors after rounding:\n", str(unique_colors))

# STEP 4: Go through each color and find shapes
vector_data = []

for i, color in enumerate(unique_colors):
    print("Processing color:", color, "... Color", str(i+1), "of", str(len(unique_colors)))
    mask = numpy.all(rounded_data == color, axis=2)     # Make a mask of all pixels that match this color

    if numpy.sum(mask) < MIN_REGION_PIXELS:             # Skip small regions (less than MIN_REGION_PIXELS)
        print("  Skipped (Mask smaller than MIN_REGION_PIXELS)")
        continue

    mask = binary_closing(mask, structure=numpy.ones(IMAGE_SHAPE_FILL))  # Close small gaps in the shape
    print("  Binary closed gaps in shapes")

    contours = measure.find_contours(mask.astype(float), 0.5)  # Find countours (aka outlines) of the masked area
    print("  Got countours (or outlines) of masked area")

    for j, contour in enumerate(contours):
        print("  Processing contour", str(j+1), "of", str(len(contours)))
        polygon = approximate_polygon(contour, tolerance=CONTOUR_TOLERANCE)  # Simplify the contour (again, contour is the outline points)
        print("    Contours simplified to contour tolerance")
        polygon = [(col[1], col[0]) for col in polygon]                      # convert (row,col) to (x,y)
        print("    Converted simplified contours to different format")

        if len(polygon) < 3: # Skip polygons that are too small
            print("    Skipped (Small polygon)")
            continue

        # Check which pixels are inside this polygon
        poly_path = Path(polygon)
        xx, yy = numpy.meshgrid(numpy.arange(width), numpy.arange(height))
        points = numpy.vstack((xx.ravel(), yy.ravel())).T
        mask_inside = poly_path.contains_points(points).reshape((height, width))
        print("    Checked which pixels are inside polygon")

        # Get average color inside the polygon
        if numpy.any(mask_inside):
            avg_color = tuple(numpy.mean(data[mask_inside], axis=0).astype(int))
            print("    Got average color in polygon")
            vector_data.append((avg_color, polygon))
            print("    Appeneded data to list")

pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Science Fair Project")
clock = pygame.time.Clock()

scale = SCREEN_SIZE / max(height, width)
offset_x, offset_y = 0, 0
dragging = False
last_mouse = (0, 0)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Zooming
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                scale *= ZOOM_STEP
            elif event.key == pygame.K_DOWN:
                scale /= ZOOM_STEP

        # Panning
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                dragging = True
                last_mouse = event.pos
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False
        if event.type == pygame.MOUSEMOTION and dragging:
            dx = event.pos[0] - last_mouse[0]
            dy = event.pos[1] - last_mouse[1]
            offset_x += dx
            offset_y += dy
            last_mouse = event.pos

    # Clear screen
    screen.fill((255, 255, 255))

    # Draw all vector shapes
    for color, polygon in vector_data:
        poly_points = [(int(x*scale + offset_x), int(y*scale + offset_y)) for x, y in polygon]
        if len(poly_points) >= 3:
            pygame.gfxdraw.filled_polygon(screen, poly_points, color)
            pygame.gfxdraw.aapolygon(screen, poly_points, color)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
