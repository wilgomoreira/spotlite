import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rasterio
from rasterio.features import rasterize
import numpy as np
from PIL import Image
import tifffile as tiff
from tifffile import imsave
import os
import re
import geopandas as gpd
from shapely.geometry import box
import shutil
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
from collections import defaultdict
from rasterio.enums import ColorInterp

def show_img_jpg(img):
    img = mpimg.imread(img)
    plt.figure(figsize=(4, 3)) 
    plt.imshow(img)
    plt.axis('off')  
    plt.show()

def generate_tile_paths_auto(base_path, prefix, extension=".TIF"):
    pattern = re.compile(r"R(\d+)C(\d+)")
    tile_dict = {}

    for filename in os.listdir(base_path):
        if filename.endswith(extension) and filename.startswith(prefix):
            match = pattern.search(filename)
            if match:
                row = int(match.group(1))
                col = int(match.group(2))
                path = os.path.join(base_path, filename)
                tile_dict[(row, col)] = path

    if not tile_dict:
        raise ValueError("No tiles found matching the pattern.")

    max_row = max(r for r, _ in tile_dict)
    max_col = max(c for _, c in tile_dict)

    tile_paths = []
    for r in range(1, max_row + 1):
        for c in range(1, max_col + 1):
            tile_paths.append(tile_dict.get((r, c), "MISSING"))

    return tile_paths, max_row, max_col


def show_all_tiles_grid(tile_paths, rows, cols, scale=1/10):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2),
                             gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape((rows, cols))

    for idx, path in enumerate(tile_paths):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        if path == "MISSING":
            ax.set_facecolor('lightgrey')
            ax.text(0.5, 0.5, "Missing Tile",
                    fontsize=10, ha='center', va='center')
            ax.axis('off')
        else:
            try:
                img = show_img_tile(path, scale=scale)
                ax.imshow(img)
                ax.axis('off')
            except Exception as e:
                print(f"Error loading {path}: {e}")
                ax.set_facecolor('red')
                ax.text(0.5, 0.5, "Error",
                        fontsize=10, ha='center', va='center')
                ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_img_tile(filepath, scale=1.0):
   
    with rasterio.open(filepath) as src:
        interp = src.colorinterp  
        mapping = {ci: i + 1 for i, ci in enumerate(interp)}
        bands = [
            mapping[ColorInterp.red],
            mapping[ColorInterp.green],
            mapping[ColorInterp.blue],
        ]
        img = src.read(bands).astype(np.float32)

    img = np.transpose(img, (1, 2, 0))

    p2, p98 = np.percentile(img, (2, 98))
    if p98 - p2 > 1e-5:
        img_norm = np.clip((img - p2) / (p98 - p2), 0, 1)
    else:
        img_norm = np.zeros_like(img)

    if scale < 1.0:
        h, w, _ = img_norm.shape
        new_h = int(h * scale)
        new_w = int(w * scale)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8)
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        img_norm = np.asarray(pil).astype(np.float32) / 255.0

    return img_norm
    
def show_classes(label_path, name_path=None):
    # Load shapefile
    gdf = gpd.read_file(label_path)
    class_column = gdf.columns[0]
    class_ids = sorted(gdf[class_column].unique())

    print("🧩 Unique class IDs found:", class_ids)

    class_labels = {}

    # Optional: load and parse class names from text file
    if name_path and os.path.exists(name_path):
        with open(name_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    class_labels[int(parts[0])] = parts[1]

        print("🔠 Class names:")
        for i in class_ids:
            name = class_labels.get(i, "(not defined)")
            print(f"  {i}: {name}")
    elif name_path:
        print(f"⚠️ Class name file not found at: {name_path}")

    return gdf, class_labels

def same_CRS(base_path, gdf):
    tif_files = [f for f in os.listdir(base_path) if f.endswith(".TIF")]
    tile_path = os.path.join(base_path, next(f for f in tif_files if f.endswith(f"R1C1.TIF")))

    with rasterio.open(tile_path) as src:
        print('\n---------CRS-----------')
        print(f'The CRS are iguals: {gdf.crs == src.crs}')
        print("CRS Img:", src.crs)
        print("CRS shapefile:", gdf.crs)
        print('\n-----AFTER TRANSFORM------------')
        new_gdf = gdf.to_crs(src.crs)
        print(f'The CRS are iguals: {new_gdf.crs == src.crs}')
        print("CRS Img:", src.crs)
        print("CRS shapefile:", new_gdf.crs)
    
    return new_gdf

def plot_tiles_and_shapefile(tile_dir, gdf, class_column='classnum2', class_labels=None,
                              filter_classes=None, title='All Tiles + Shapefile'):
    
    gdf_tiles = get_tiles_bounds(tile_dir)

    if gdf.crs != gdf_tiles.crs:
        gdf = gdf.to_crs(gdf_tiles.crs)

    # Filter classes if requested
    if filter_classes is not None:
        gdf = gdf[gdf[class_column].isin(filter_classes)].copy()

    # Apply class labels if provided
    if class_labels is not None:
        gdf['class_name'] = gdf[class_column].map(class_labels)
        class_column = 'class_name'
    else:
        gdf[class_column] = gdf[class_column].astype(str)


    fig, ax = plt.subplots(figsize=(6, 4))
    gdf_tiles.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1)

    # Label tile IDs
    for _, row in gdf_tiles.iterrows():
        x, y = row.geometry.centroid.x, row.geometry.centroid.y
        ax.text(x, y, row.tile_id, fontsize=8, ha='center', va='center', color='blue')

    # Plot shapefile with colored classes
    gdf.plot(
        ax=ax,
        column=class_column,
        cmap='tab10',
        categorical=True,
        linewidth=0.5,
        edgecolor='black',
        legend=True,
        legend_kwds={
            'loc': 'upper left',
            'bbox_to_anchor': (1.0, 1.0),
            'title': 'Forest Types (Classes)',
            'fontsize': 10
        }
    )

    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def get_tiles_bounds(directory):
    data = []
    for f in os.listdir(directory):
        if f.endswith(".TIF"):
            path = os.path.join(directory, f)
            with rasterio.open(path) as src:
                bounds = src.bounds
                geometry = box(*bounds)
                tile_id = f.split("_")[-1].replace(".TIF", "")  # e.g., R2C3
                data.append({
                    "tile_name": f,
                    "tile_id": tile_id,
                    "geometry": geometry,
                    "crs": src.crs
                })
    gdf_tiles = gpd.GeoDataFrame(data)
    gdf_tiles.set_crs(src.crs, inplace=True)
    return gdf_tiles

def assign_labels_to_tiles(gdf_labels, gdf_tiles):
    if gdf_labels.crs != gdf_tiles.crs:
        gdf_labels = gdf_labels.to_crs(gdf_tiles.crs)

    gdf_labels['geometry'] = gdf_labels['geometry'].buffer(0)
    gdf_tiles['geometry'] = gdf_tiles['geometry'].buffer(0)

    gdf_intersection = gpd.overlay(gdf_labels, gdf_tiles, how='intersection')

    return gdf_intersection

def load_full_images_and_masks(tile_paths, labels_gdf, class_column='classnum2'):
    loaded_data = []

    for tile_path in tile_paths:
        if tile_path == "MISSING":
            continue

        tile_id = get_tile_id_from_path(tile_path)

        if tile_id not in labels_gdf['tile_id'].values:
            print(f"⚠️ No labels found for tile {tile_id}")
            continue

        with rasterio.open(tile_path) as src:
            image = src.read([1, 2, 3, 4])  # RGB + NIR
            transform = src.transform
            crs = src.crs
            height, width = src.height, src.width

        # Normalize image
        def normalize(band):
            min_val = band.min()
            max_val = band.max()
            if max_val == min_val:
                return np.zeros_like(band, dtype=np.uint8)
            return ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        image_norm = np.stack([normalize(image[i]) for i in range(image.shape[0])], axis=0)  # shape [C, H, W]

        # Rasterize mask
        labels_tile = labels_gdf[labels_gdf['tile_id'] == tile_id]
        if not labels_tile.empty:
            if labels_tile.crs != crs:
                labels_tile = labels_tile.to_crs(crs)

            shapes = [(geom, int(val)) for geom, val in zip(labels_tile.geometry, labels_tile[class_column])]
            mask = rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype='uint8'
            )
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        # Save to memory (not disk)
        loaded_data.append({
            "tile_id": tile_id,
            "image": image_norm,  # shape [C, H, W]
            "mask": mask,         # shape [H, W]
            "transform": transform,
            "crs": crs
        })

    print(f"✅ Loaded {len(loaded_data)} tiles into memory.")
    return loaded_data

def get_tile_id_from_path(path):
    filename = os.path.basename(path)
    match = re.search(r'R\d+C\d+', filename)
    return match.group(0) if match else None


def check_class_distribution_in_tiles_data(tiles_data):
    all_classes = set()
    class_pixel_counts = defaultdict(int)

    for tile in tiles_data:
        tile_id = tile["tile_id"]
        mask = tile["mask"]

        unique, counts = np.unique(mask, return_counts=True)
        class_map = dict(zip(unique, counts))

        all_classes.update(unique)
        for cls, cnt in class_map.items():
            class_pixel_counts[cls] += cnt

        print(f"{tile_id}: classes found -> {unique}")

    total_pixels = sum(class_pixel_counts.values())

    print("\n✅ Total unique classes across all masks:", sorted(all_classes))
    print("📊 Total pixel count and class percentage:")
    for cls in sorted(class_pixel_counts):
        count = class_pixel_counts[cls]
        percent = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"  Class {cls}: {count:,} pixels ({percent:.2f}%)")


def visualize_tiles_data(tiles_data, desired_classes=None, max_items=None, selected_indices=None):
    # Filter selected indices
    if selected_indices:
        tiles_data = [tiles_data[i] for i in selected_indices if i < len(tiles_data)]

    if max_items:
        tiles_data = tiles_data[:max_items]

    if len(tiles_data) == 0:
        print("❌ No tiles to display.")
        return

    # Define label info and colors
    full_class_labels = [
        "0: background",
        "1: no trees",
        "2: other vegetation",
        "3: pinus",
        "4: eucalyptus",
        "5: shadow"
    ]
    full_mask_colors = [
        (0, 0, 0, 1.0),
        (0.12, 0.47, 0.71, 0.6),
        (0.20, 0.63, 0.17, 0.6),
        (0.84, 0.15, 0.16, 0.6),
        (0.58, 0.40, 0.74, 0.6),
        (0.10, 0.74, 0.81, 0.6)
    ]
    full_overlay_colors = [
        (0, 0, 0, 0.0),
        (0.12, 0.47, 0.71, 0.6),
        (0.20, 0.63, 0.17, 0.6),
        (0.84, 0.15, 0.16, 0.6),
        (0.58, 0.40, 0.74, 0.6),
        (0.10, 0.74, 0.81, 0.6)
    ]

    # Get all classes in current masks
    present_classes = set()
    for tile in tiles_data:
        present_classes.update(np.unique(tile["mask"]))
    present_classes = sorted(present_classes)

    # Filter by desired classes
    if desired_classes is not None:
        used_classes = [c for c in present_classes if c in desired_classes]
    else:
        used_classes = present_classes

    if 0 in present_classes and 0 not in used_classes:
        used_classes = [0] + used_classes

    # Reindex classes and define colormaps
    class_remap = {original: new_idx for new_idx, original in enumerate(used_classes)}
    mask_cmap = ListedColormap([full_mask_colors[c] for c in used_classes])
    overlay_cmap = ListedColormap([full_overlay_colors[c] for c in used_classes])
    legend_handles = [Patch(color=full_overlay_colors[c], label=full_class_labels[c]) for c in used_classes]

    # Plot
    n = len(tiles_data)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for idx, tile in enumerate(tiles_data):
        image = tile["image"]
        mask = tile["mask"]
        tile_id = tile["tile_id"]

        # Reindex classes
        mask_reindexed = np.vectorize(class_remap.get)(mask)

        # Convert to RGB
        if image.ndim == 3 and image.shape[0] in [3, 4]:  # CHW
            rgb = image[:3].transpose(1, 2, 0)
        elif image.ndim == 3 and image.shape[2] in [3, 4]:  # HWC
            rgb = image[:, :, :3]
        else:
            rgb = np.repeat(image[None, :, :], 3, axis=0).transpose(1, 2, 0)

        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title(f"Image: {tile_id}", fontsize=10)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(mask_reindexed, cmap=mask_cmap, interpolation='nearest', vmin=0, vmax=len(used_classes)-1)
        axes[idx, 1].set_title("Mask", fontsize=10)
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(rgb)
        axes[idx, 2].imshow(mask_reindexed, cmap=overlay_cmap, interpolation='nearest', vmin=0, vmax=len(used_classes)-1)
        axes[idx, 2].set_title("Overlay", fontsize=10)
        axes[idx, 2].axis('off')

    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=9,
               title="Class Index", title_fontsize=10, frameon=True)

    plt.tight_layout()
    plt.show()

def generate_patches_from_tiles_data(tiles_data, output_dir, patch_size=512, overlap=0):
    # Clean output directories first
    clean_output_dirs(output_dir)

    img_out_dir = os.path.join(output_dir, 'images')
    msk_out_dir = os.path.join(output_dir, 'masks')

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(msk_out_dir, exist_ok=True)

    total_saved = 0

    for tile in tiles_data:
        tile_id = tile['tile_id']
        image = tile['image']  # [C, H, W]
        mask = tile['mask']    # [H, W]

        # Transpose image to [H, W, C]
        if image.ndim == 3 and image.shape[0] in [3, 4]:
            img = image.transpose(1, 2, 0)
        else:
            raise ValueError(f"Invalid image shape in tile {tile_id}")

        H, W = mask.shape
        stride = patch_size - overlap

        patch_count = 0
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                img_patch = img[y:y+patch_size, x:x+patch_size, :]
                msk_patch = mask[y:y+patch_size, x:x+patch_size]

                # Skip if mask contains only background
                if np.all(msk_patch == 0):
                    continue

                patch_name = f"{tile_id}_patch_{y}_{x}.tif"
                imsave(os.path.join(img_out_dir, patch_name), img_patch, photometric='rgb')
                imsave(os.path.join(msk_out_dir, patch_name.replace('.tif', '_mask.tif')), msk_patch)

                patch_count += 1
                total_saved += 1

        print(f"✅ {patch_count} patches saved from tile {tile_id}")

    print(f"\n🎉 Finished: {total_saved} total image/mask pairs saved in '{output_dir}'")

def clean_output_dirs(output_dir):
    img_dir = os.path.join(output_dir, 'images')
    msk_dir = os.path.join(output_dir, 'masks')

    for d in [img_dir, msk_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def visualize_image_mask_overlay(images_dir, masks_dir, desired_classes=None, max_items=None, selected_indices=None):
    # Get only .tif and _mask.tif files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('_mask.tif')])

    # Match by basename
    image_basenames = [f.replace('.tif', '') for f in image_files]
    mask_basenames = [f.replace('_mask.tif', '') for f in mask_files]
    common = sorted(list(set(image_basenames) & set(mask_basenames)))

    image_files = [f + '.tif' for f in common]
    mask_files = [f + '_mask.tif' for f in common]

    # Select specific indices if provided
    if selected_indices:
        # Ensure no index is out of bounds
        selected_indices = [i for i in selected_indices if i < len(image_files)]
        image_files = [image_files[i] for i in selected_indices]
        mask_files = [mask_files[i] for i in selected_indices]

    # Limit to max_items
    n_total = len(image_files)
    if max_items:
        image_files = image_files[:max_items]
        mask_files = mask_files[:max_items]

    # Full label info
    full_class_labels = [
        "0: background",
        "1: no trees",
        "2: other vegetation",
        "3: pinus",
        "4: eucalyptus",
        "5: shadow"
    ]
    full_mask_colors = [
        (0, 0, 0, 1.0),
        (0.12, 0.47, 0.71, 0.6),
        (0.20, 0.63, 0.17, 0.6),
        (0.84, 0.15, 0.16, 0.6),
        (0.58, 0.40, 0.74, 0.6),
        (0.10, 0.74, 0.81, 0.6)
    ]
    full_overlay_colors = [
        (0, 0, 0, 0.0),
        (0.12, 0.47, 0.71, 0.15),
        (0.20, 0.63, 0.17, 0.15),
        (0.84, 0.15, 0.16, 0.15),
        (0.58, 0.40, 0.74, 0.15),
        (0.10, 0.74, 0.81, 0.15)
    ]

    # Check classes present in selected masks
    present_classes = set()
    for mf in mask_files:
        mask = tiff.imread(os.path.join(masks_dir, mf))
        present_classes.update(np.unique(mask))
    present_classes = sorted(present_classes)

    # Filter by desired_classes
    if desired_classes is not None:
        used_classes = [c for c in present_classes if c in desired_classes]
    else:
        used_classes = present_classes

    if 0 in present_classes and 0 not in used_classes:
        used_classes = [0] + used_classes

    # Create remap dict
    class_remap = {original: new_idx for new_idx, original in enumerate(used_classes)}

    # Color maps
    mask_cmap = ListedColormap([full_mask_colors[c] for c in used_classes])
    overlay_cmap = ListedColormap([full_overlay_colors[c] for c in used_classes])
    legend_handles = [Patch(color=full_overlay_colors[c], label=full_class_labels[c]) for c in used_classes]

    # Plotting
    n = len(image_files)
    fig, axes = plt.subplots(n, 3, figsize=(8, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for idx, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, mask_file)

        image = tiff.imread(image_path)
        mask = tiff.imread(mask_path)

        # Reindex mask values
        mask_reindexed = np.vectorize(class_remap.get)(mask)

        # Convert to RGB
        if image.ndim == 3 and image.shape[0] in [3, 4]:
            rgb = image[:3].transpose(1, 2, 0)
        elif image.ndim == 3 and image.shape[2] in [3, 4]:
            rgb = image[:, :, :3]
        else:
            rgb = np.repeat(image[None, :, :], 3, axis=0).transpose(1, 2, 0)

        # Image
        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title("Image", fontsize=10)
        axes[idx, 0].axis('off')

        # Mask
        axes[idx, 1].imshow(mask_reindexed, cmap=mask_cmap, interpolation='nearest', vmin=0, vmax=len(used_classes)-1)
        axes[idx, 1].set_title("Mask", fontsize=10)
        axes[idx, 1].axis('off')

        # Overlay
        axes[idx, 2].imshow(rgb)
        axes[idx, 2].imshow(mask_reindexed, cmap=overlay_cmap, interpolation='nearest', vmin=0, vmax=len(used_classes)-1)
        axes[idx, 2].set_title("Overlay", fontsize=10)
        axes[idx, 2].axis('off')

    # Legend
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=9,
               title="Class Index", title_fontsize=10, frameon=True)

    plt.tight_layout()
    plt.show()




    