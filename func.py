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
            interp = src.colorinterp
            mapping = {ci: i+1 for i, ci in enumerate(interp)}
            bands = [
                mapping[ColorInterp.red],
                mapping[ColorInterp.green],
                mapping[ColorInterp.blue],
                mapping.get(ColorInterp.nir,                       
                            mapping.get(ColorInterp.undefined, 4))  
            ]
            image = src.read(bands).astype(np.float32)  
            transform = src.transform
            crs = src.crs
            height, width = src.height, src.width

        def normalize_percentile(band):
            p2, p98 = np.percentile(band, (2, 98))
            if p98 - p2 > 1e-5:
                return np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                return np.zeros_like(band, dtype=np.float32)

        image_norm = np.stack([normalize_percentile(image[i])
                               for i in range(image.shape[0])], axis=0)
        

       
        labels_tile = labels_gdf[labels_gdf['tile_id'] == tile_id]
        if not labels_tile.empty:
            if labels_tile.crs != crs:
                labels_tile = labels_tile.to_crs(crs)
            shapes = [
                (geom, int(val))
                for geom, val in zip(labels_tile.geometry,
                                     labels_tile[class_column])
            ]
            mask = rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype='uint8'
            )
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        loaded_data.append({
            "tile_id": tile_id,
            "image": image_norm,  # [4, H, W],
            "mask": mask,         # [H, W]
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

def visualize_tiles_data(
    tiles_data,
    desired_classes=None,
    max_items=None,
    selected_indices=None,
    scale=0.5,         # ↓ define aqui o fator de down‐sampling
):
    # 1) filtra índices e limite
    if selected_indices:
        tiles_data = [tiles_data[i] for i in selected_indices if i < len(tiles_data)]
    if max_items:
        tiles_data = tiles_data[:max_items]
    if not tiles_data:
        print("❌ No tiles to display."); return

    # 2) informações de classes e cores
    full_class_labels = [
        "0: background","1: no trees","2: other vegetation",
        "3: pinus","4: eucalyptus","5: shadow"
    ]
    full_mask_colors = [
        (0,0,0,1.0),
        (0.12,0.47,0.71,0.6),
        (0.20,0.63,0.17,0.6),
        (0.84,0.15,0.16,0.6),
        (0.58,0.40,0.74,0.6),
        (0.10,0.74,0.81,0.6),
    ]

    # 3) descobre classes presentes
    present = set()
    for tile in tiles_data:
        present.update(np.unique(tile["mask"]))
    present = sorted(present)

    # 4) filtra conforme desired_classes
    if desired_classes is not None:
        used = [c for c in present if c in desired_classes]
    else:
        used = present[:]
    if 0 in present and 0 not in used:
        used.insert(0, 0)

    # 5) mapeamento e colormaps
    class_remap = {orig: new for new, orig in enumerate(used)}
    mask_cmap = ListedColormap([full_mask_colors[c] for c in used])
    legend_handles = [Patch(color=full_mask_colors[c], label=full_class_labels[c])
                      for c in used]

    # overlay totalmente opaco
    overlay_colors = [
        (0,0,0,0.0),
        (0.12,0.47,0.71,1.0),
        (0.20,0.63,0.17,1.0),
        (0.84,0.15,0.16,1.0),
        (0.58,0.40,0.74,1.0),
        (0.10,0.74,0.81,1.0),
    ]
    overlay_cmap = ListedColormap([overlay_colors[c] for c in used])

    # 6) função de contraste 2–98%
    def stretch(img):
        out = np.zeros_like(img, dtype=np.float32)
        for k in range(3):
            p2, p98 = np.percentile(img[...,k], (2,98))
            if p98 > p2:
                out[...,k] = np.clip((img[...,k] - p2)/(p98-p2), 0, 1)
        return out

    # 7) helper de down‑sample
    def downsample(rgb, mask_idx):
        h, w = rgb.shape[:2]
        new_w, new_h = int(w*scale), int(h*scale)
        # RGB: converte pra uint8, redimensiona e volta para float
        pil_rgb = Image.fromarray((rgb*255).astype(np.uint8))
        rgb_ds  = np.asarray(pil_rgb.resize((new_w,new_h), Image.BILINEAR)) / 255.0
        # máscara: nearest
        pil_m  = Image.fromarray(mask_idx.astype(np.uint8))
        m_ds   = np.asarray(pil_m.resize((new_w,new_h), Image.NEAREST))
        return rgb_ds, m_ds

    # 8) plot em grid
    n = len(tiles_data)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    if n == 1:
        axes = axes[np.newaxis,:]

    for i, tile in enumerate(tiles_data):
        img = tile["image"]   # pode ser CHW ou HWC
        msk = tile["mask"]
        tid = tile["tile_id"]

        # 8a) remapeia máscara
        m_idx = np.zeros_like(msk, dtype=np.int32)
        for orig, new in class_remap.items():
            m_idx[msk == orig] = new

        # 8b) extrai RGB em HWC
        if img.ndim==3 and img.shape[0] in (3,4):
            rgb = img[:3].transpose(1,2,0)
        elif img.ndim==3 and img.shape[2] in (3,4):
            rgb = img[...,:3]
        else:
            rgb = np.repeat(img[np.newaxis],3,axis=0).transpose(1,2,0)

        # 8c) converte / clipa
        if np.issubdtype(rgb.dtype, np.integer):
            rgb = rgb.astype(np.float32)/255.0
        else:
            rgb = np.clip(rgb,0,1)

        # 8d) contraste
        rgb = stretch(rgb)

        # 8e) down‑sample se solicitado
        if scale < 1.0:
            rgb, m_idx = downsample(rgb, m_idx)

        # 8f1) imagem pura
        ax = axes[i,0]
        ax.imshow(rgb)
        ax.set_title(f"Image: {tid}", fontsize=10)
        ax.axis('off')

        # 8f2) máscara
        ax = axes[i,1]
        ax.imshow(m_idx, cmap=mask_cmap,
                  interpolation='nearest',
                  vmin=0, vmax=len(used)-1)
        ax.set_title("Mask", fontsize=10)
        ax.axis('off')

        # 8f3) overlay
        ax = axes[i,2]
        ax.imshow(rgb)
        ax.imshow(m_idx, cmap=overlay_cmap,
                  interpolation='nearest',
                  vmin=0, vmax=len(used)-1)
        # contorno grosso
        for new_idx in range(1, len(used)):
            binary = (m_idx == new_idx).astype(np.uint8)
            ax.contour(binary,
                       levels=[0.5],
                       colors=['white'],
                       linewidths=2.0)
        ax.set_title("Overlay", fontsize=10)
        ax.axis('off')

    # 9) legenda
    fig.legend(handles=legend_handles,
               loc='upper right',
               bbox_to_anchor=(1.1,1.0),
               fontsize=9,
               title="Class Index",
               title_fontsize=10,
               frameon=True)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def generate_patches_from_tiles_data(
    tiles_data,
    output_dir,
    patch_size=512,
    overlap=0,
    include_empty=True   # <<< now you control whether to save pure background patches
):
    # Clean output directories
    clean_output_dirs(output_dir)
    img_out = os.path.join(output_dir, 'images')
    msk_out = os.path.join(output_dir, 'masks')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    total_saved = 0
    stats = defaultdict(int)  # optional: count how many patches of each type

    for tile in tiles_data:
        tid   = tile['tile_id']
        img   = tile['image']   # [C, H, W]
        msk   = tile['mask']    # [H, W]

        # Transpose to H, W, C
        if img.ndim == 3 and img.shape[0] in (3,4):
            img = img.transpose(1, 2, 0)
        else:
            raise ValueError(f"Invalid format in tile {tid}: {img.shape}")

        H, W = msk.shape
        stride = patch_size - overlap

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                img_patch = img[y:y+patch_size, x:x+patch_size, :]
                msk_patch = msk[y:y+patch_size, x:x+patch_size]

                # Calculate percentage of background
                frac_bg = np.mean(msk_patch == 0)
                # if include_empty=False and patch is only background, skip
                if not include_empty and frac_bg == 1.0:
                    stats['skipped_empty'] += 1
                    continue

                # save
                name = f"{tid}_y{y}_x{x}.tif"
                imsave(os.path.join(img_out, name),
                       img_patch, photometric='rgb')
                msk_name = name.replace('.tif', '_mask.tif')
                imsave(os.path.join(msk_out, msk_name), msk_patch)

                total_saved += 1
                # categorize for quick statistics
                if frac_bg == 1.0:
                    stats['empty'] += 1
                else:
                    stats['non_empty'] += 1

        print(f"Tile {tid}: patches saved so far: {total_saved}")

    print(f"\n🎉 Total patches saved: {total_saved}")
    print("📊 Quick patch statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

def clean_output_dirs(output_dir):
    img_dir = os.path.join(output_dir, 'images')
    msk_dir = os.path.join(output_dir, 'masks')

    for d in [img_dir, msk_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def visualize_image_mask_overlay(
    images_dir,
    masks_dir,
    desired_classes=None,
    max_items=None,
    selected_indices=None
):
    # 1) lista e casa arquivos
    imgs = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
    msks = sorted([f for f in os.listdir(masks_dir)  if f.endswith('_mask.tif')])
    base_i = [f[:-4] for f in imgs]
    base_m = [f[:-9] for f in msks]
    common = sorted(set(base_i) & set(base_m))
    image_files = [f + '.tif'      for f in common]
    mask_files  = [f + '_mask.tif' for f in common]

    # 2) seleção por índice e limite
    if selected_indices:
        image_files = [image_files[i] for i in selected_indices if i < len(common)]
        mask_files  = [mask_files[i]  for i in selected_indices if i < len(common)]
    if max_items:
        image_files = image_files[:max_items]
        mask_files  = mask_files[:max_items]

    # 3) definição de classes e cores
    full_labels = [
        "0: background","1: no trees","2: other vegetation",
        "3: pinus","4: eucalyptus","5: shadow"
    ]
    full_mask_colors = [
        (0,0,0,1.0),
        (0.12,0.47,0.71,0.6),
        (0.20,0.63,0.17,0.6),
        (0.84,0.15,0.16,0.6),
        (0.58,0.40,0.74,0.6),
        (0.10,0.74,0.81,0.6),
    ]
    full_overlay_colors = [
        (0,0,0,0.0),
        (0.12,0.47,0.71,1.0),
        (0.20,0.63,0.17,1.0),
        (0.84,0.15,0.16,1.0),
        (0.58,0.40,0.74,1.0),
        (0.10,0.74,0.81,1.0),
    ]

    # 4) quais classes aparecem?
    present = set()
    for mf in mask_files:
        arr = tiff.imread(os.path.join(masks_dir, mf))
        present.update(np.unique(arr))
    present = sorted(present)

    # 5) filtra se necessário
    if desired_classes is not None:
        used = [c for c in present if c in desired_classes]
    else:
        used = present[:]
    if 0 in present and 0 not in used:
        used.insert(0, 0)

    # 6) remap, cmaps e legenda
    class_remap = {orig: new for new, orig in enumerate(used)}
    mask_cmap    = ListedColormap([full_mask_colors[c]    for c in used])
    overlay_cmap = ListedColormap([full_overlay_colors[c] for c in used])
    legend_handles = [
        Patch(color=full_overlay_colors[c], label=full_labels[c])
        for c in used
    ]

    # 7) stretch percentil 2–98%
    def stretch_rgb(rgb):
        out = np.zeros_like(rgb, dtype=np.float32)
        for k in range(3):
            p2, p98 = np.percentile(rgb[...,k], (2, 98))
            if p98 > p2:
                out[...,k] = np.clip((rgb[...,k] - p2)/(p98-p2), 0, 1)
        return out

    # 8) plota
    n = len(image_files)
    fig, axes = plt.subplots(n, 3, figsize=(8, 4*n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (ifile, mfile) in enumerate(zip(image_files, mask_files)):
        img = tiff.imread(os.path.join(images_dir, ifile))
        msk = tiff.imread(os.path.join(masks_dir,  mfile))

        # 8a) remap máscara
        m_idx = np.zeros_like(msk, dtype=np.int32)
        for orig, new in class_remap.items():
            m_idx[msk == orig] = new

        # 8b) extrai RGB em HWC
        if img.ndim == 3 and img.shape[0] in (3,4):
            rgb = img[:3].transpose(1,2,0)
        elif img.ndim == 3 and img.shape[2] in (3,4):
            rgb = img[..., :3]
        else:
            rgb = np.repeat(img[np.newaxis], 3, axis=0).transpose(1,2,0)

        # 8c) converte inteiros→float ou clipa floats
        if np.issubdtype(rgb.dtype, np.integer):
            rgb = rgb.astype(np.float32) / 255.0
        else:
            rgb = np.clip(rgb, 0, 1)

        # 8d) aplica stretch
        rgb = stretch_rgb(rgb)

        # 8e1) subplot RGB
        ax = axes[i,0]
        ax.imshow(rgb)
        ax.set_title("Image", fontsize=10)
        ax.axis("off")

        # 8e2) subplot máscara
        ax = axes[i,1]
        ax.imshow(m_idx,
                  cmap=mask_cmap,
                  interpolation="nearest",
                  vmin=0, vmax=len(used)-1)
        ax.set_title("Mask", fontsize=10)
        ax.axis("off")

        # 8e3) subplot overlay
        ax = axes[i,2]
        ax.imshow(rgb)
        ax.imshow(m_idx,
                  cmap=overlay_cmap,
                  interpolation="nearest",
                  vmin=0, vmax=len(used)-1)
        # contorno branco grosso
        for new_idx in range(1, len(used)):
            binar = (m_idx == new_idx).astype(np.uint8)
            ax.contour(binar,
                       levels=[0.5],
                       colors=["white"],
                       linewidths=2.0)
        ax.set_title("Overlay", fontsize=10)
        ax.axis("off")

    fig.legend(handles=legend_handles,
               loc="upper right",
               bbox_to_anchor=(1.1, 1.0),
               fontsize=9,
               title="Class Index",
               title_fontsize=10,
               frameon=True)

    plt.tight_layout()
    plt.show()




    