#!/usr/bin/env python3
"""
Country Border Poster Generator

This module generates beautiful, minimalist country border posters.
It fetches OpenStreetMap boundary data using OSMnx, applies customizable themes,
and creates high-quality poster-ready images with country outlines.
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import requests
from geopandas import GeoDataFrame
from lat_lon_parser import parse
from matplotlib.font_manager import FontProperties
from shapely.geometry import MultiPolygon

from font_management import load_fonts


class CacheError(Exception):
    """Raised when a cache operation fails."""


CACHE_DIR_PATH = os.environ.get("CACHE_DIR", "cache")
CACHE_DIR = Path(CACHE_DIR_PATH)
CACHE_DIR.mkdir(exist_ok=True)

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

FILE_ENCODING = "utf-8"

FONTS = load_fonts()


def _cache_path(key: str) -> str:
    """
    Generate a safe cache file path from a cache key.

    Args:
        key: Cache key identifier

    Returns:
        Path to cache file with .pkl extension
    """
    safe = key.replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str):
    """
    Retrieve a cached object by key.

    Args:
        key: Cache key identifier

    Returns:
        Cached object if found, None otherwise

    Raises:
        CacheError: If cache read operation fails
    """
    try:
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CacheError(f"Cache read failed: {e}") from e


def cache_set(key: str, value):
    """
    Store an object in the cache.

    Args:
        key: Cache key identifier
        value: Object to cache (must be picklable)

    Raises:
        CacheError: If cache write operation fails
    """
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        path = _cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise CacheError(f"Cache write failed: {e}") from e


# Font loading now handled by font_management.py module


def is_latin_script(text):
    """
    Check if text is primarily Latin script.
    Used to determine if letter-spacing should be applied to country names.

    :param text: Text to analyze
    :return: True if text is primarily Latin script, False otherwise
    """
    if not text:
        return True

    latin_count = 0
    total_alpha = 0

    for char in text:
        if char.isalpha():
            total_alpha += 1
            # Latin Unicode ranges:
            # - Basic Latin: U+0000 to U+007F
            # - Latin-1 Supplement: U+0080 to U+00FF
            # - Latin Extended-A: U+0100 to U+017F
            # - Latin Extended-B: U+0180 to U+024F
            if ord(char) < 0x250:
                latin_count += 1

    # If no alphabetic characters, default to Latin (numbers, symbols, etc.)
    if total_alpha == 0:
        return True

    # Consider it Latin if >80% of alphabetic characters are Latin
    return (latin_count / total_alpha) > 0.8


def generate_output_filename(country, theme_name, output_format):
    """
    Generate unique output filename with country, theme, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    country_slug = country.lower().replace(" ", "_")
    ext = output_format.lower()
    filename = f"{country_slug}_{theme_name}_{timestamp}.{ext}"
    return os.path.join(POSTERS_DIR, filename)


def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []

    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith(".json"):
            theme_name = file[:-5]  # Remove .json extension
            themes.append(theme_name)
    return themes


def load_theme(theme_name="terracotta"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")

    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default terracotta theme.")
        # Fallback to embedded terracotta theme
        return {
            "name": "Terracotta",
            "description": "Mediterranean warmth - burnt orange and clay tones on cream",
            "bg": "#F5EDE4",
            "text": "#8B4513",
            "gradient_color": "#F5EDE4",
            "water": "#A8C4C4",
            "parks": "#E8E0D0",
            "road_motorway": "#A0522D",
            "road_primary": "#B8653A",
            "road_secondary": "#C9846A",
            "road_tertiary": "#D9A08A",
            "road_residential": "#E5C4B0",
            "road_default": "#D9A08A",
            "border": "#8B4513",
            "subdivision": "#C9846A",
        }

    with open(theme_file, "r", encoding=FILE_ENCODING) as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if "description" in theme:
            print(f"  {theme['description']}")
        return theme


# Load theme (can be changed via command line or input)
THEME = dict[str, str]()  # Will be loaded later


def create_gradient_fade(ax, color, location="bottom", zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))

    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]

    if location == "bottom":
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end

    ax.imshow(
        gradient,
        extent=[xlim[0], xlim[1], y_bottom, y_top],
        aspect="auto",
        cmap=custom_cmap,
        zorder=zorder,
        origin="lower",
    )


def _download_naturalearth(dataset, cache_key_name) -> GeoDataFrame:
    """
    Download and cache a Natural Earth shapefile.
    """
    import io
    import zipfile

    cached = cache_get(cache_key_name)
    if cached is not None:
        print(f"  ✓ Using cached Natural Earth {dataset} data")
        return cast(GeoDataFrame, cached)

    url = f"https://naciscdn.org/naturalearth/10m/cultural/{dataset}.zip"
    print(f"  Downloading Natural Earth {dataset}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    import geopandas as gpd

    gdf = gpd.read_file(io.BytesIO(resp.content))

    try:
        cache_set(cache_key_name, gdf)
    except CacheError as e:
        print(e)
    return gdf


def _download_naturalearth_10m() -> GeoDataFrame:
    """Download Natural Earth 10m country boundaries."""
    return _download_naturalearth(
        "ne_10m_admin_0_countries", "naturalearth_10m_countries"
    )


def _download_naturalearth_10m_admin1() -> GeoDataFrame:
    """Download Natural Earth 10m admin-1 (states/provinces) boundaries."""
    return _download_naturalearth(
        "ne_10m_admin_1_states_provinces", "naturalearth_10m_admin1"
    )


def fetch_country_boundary(country) -> GeoDataFrame | None:
    """
    Fetch high-detail country boundary from Natural Earth 10m data.

    Uses Natural Earth's cartographic-quality 10m resolution boundaries.
    Matches by NAME, NAME_LONG, ADMIN, FORMAL_EN, or SOVEREIGNT fields.
    Falls back to OSMnx if no match is found.

    Args:
        country: Country name to search for

    Returns:
        GeoDataFrame with country boundary, or None if not found
    """
    cache_key = f"country_boundary_ne10m_{country.lower().replace(' ', '_')}"
    cached = cache_get(cache_key)
    if cached is not None:
        print("✓ Using cached country boundary")
        return cast(GeoDataFrame, cached)

    try:
        ne = _download_naturalearth_10m()

        # Match country name against multiple fields
        country_lower = country.lower()
        name_fields = ["NAME", "NAME_LONG", "ADMIN", "FORMAL_EN", "SOVEREIGNT"]
        match = None
        for field in name_fields:
            if field not in ne.columns:
                continue
            hits = ne[ne[field].str.lower() == country_lower]
            if not hits.empty:
                match = hits
                break

        # Fuzzy fallback: check if country is a substring
        if match is None:
            for field in name_fields:
                if field not in ne.columns:
                    continue
                hits = ne[ne[field].str.lower().str.contains(country_lower, na=False)]
                if not hits.empty:
                    match = hits.iloc[[0]]
                    break

        if match is None:
            print(f"  Country '{country}' not found in Natural Earth data")
            return _fetch_boundary_fallback(country, cache_key)

        gdf = match[["geometry"]].copy()
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

        total_verts = sum(
            len(p.exterior.coords)
            for geom in gdf.geometry
            for p in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom])
        )
        print(f"  ✓ Found boundary ({total_verts} vertices)")

        try:
            cache_set(cache_key, gdf)
        except CacheError as e:
            print(e)
        return gdf
    except Exception as e:
        print(f"Error fetching Natural Earth boundary: {e}")
        return _fetch_boundary_fallback(country, cache_key)


def fetch_subdivisions(country) -> GeoDataFrame | None:
    """
    Fetch admin-1 subdivision boundaries (states/provinces) from Natural Earth 10m.

    Args:
        country: Country name to match

    Returns:
        GeoDataFrame with subdivision geometries, or None
    """
    cache_key = f"subdivisions_ne10m_{country.lower().replace(' ', '_')}"
    cached = cache_get(cache_key)
    if cached is not None:
        print("✓ Using cached subdivision boundaries")
        return cast(GeoDataFrame, cached)

    try:
        ne = _download_naturalearth_10m_admin1()

        country_lower = country.lower()
        name_fields = ["admin", "name", "sovereignt"]
        match = None
        for field in name_fields:
            if field not in ne.columns:
                continue
            hits = ne[ne[field].str.lower() == country_lower]
            if not hits.empty:
                match = hits
                break

        if match is None:
            for field in name_fields:
                if field not in ne.columns:
                    continue
                hits = ne[ne[field].str.lower().str.contains(country_lower, na=False)]
                if not hits.empty:
                    match = hits
                    break

        if match is None:
            print(f"  No subdivisions found for '{country}'")
            return None

        gdf = match[["geometry"]].copy()
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        print(f"  ✓ Found {len(gdf)} subdivisions")

        try:
            cache_set(cache_key, gdf)
        except CacheError as e:
            print(e)
        return gdf
    except Exception as e:
        print(f"Error fetching subdivisions: {e}")
        return None


def _fetch_boundary_fallback(country, cache_key=None) -> GeoDataFrame | None:
    """Fallback boundary fetch using OSMnx (lower resolution)."""
    try:
        print("  Using fallback (OSMnx)...")
        gdf = ox.geocode_to_gdf(country)
        time.sleep(0.5)
        if cache_key:
            try:
                cache_set(cache_key, gdf)
            except CacheError as e:
                print(e)
        return gdf
    except Exception as e:
        print(f"Fallback also failed: {e}")
        return None


def _extract_rings(geom):
    """Extract all exterior and interior rings from a geometry."""
    if geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            yield poly.exterior
            yield from poly.interiors
    elif geom.geom_type == "Polygon":
        yield geom.exterior
        yield from geom.interiors
    elif hasattr(geom, "coords"):
        yield geom


def _extract_rings_from_gdf(gdf):
    """Extract all rings from all geometries in a GeoDataFrame."""
    for geom in gdf.geometry:
        yield from _extract_rings(geom)


def create_poster(
    country,
    point,
    boundary,
    output_file,
    output_format,
    width=12,
    height=16,
    display_country=None,
    fonts=None,
    distance=None,
    subdivisions=None,
):
    """
    Generate a country border poster with typography.

    Creates a high-quality poster by rendering the country boundary outline,
    applying the current theme, and adding text labels with coordinates.

    Args:
        country: Country name for display on poster
        point: (latitude, longitude) tuple for coordinates display
        boundary: GeoDataFrame with country boundary geometry
        output_file: Path where poster will be saved
        output_format: File format ('png', 'svg', or 'pdf')
        width: Poster width in inches (default: 12)
        height: Poster height in inches (default: 16)
        display_country: Optional override for country text on poster
        fonts: Optional custom font dict
    """
    display_country = display_country or country

    print(f"\nGenerating poster for {country}...")

    # Setup Plot
    print("Rendering map...")
    fig, ax = plt.subplots(figsize=(width, height), facecolor=THEME["bg"])
    ax.set_facecolor(THEME["bg"])
    ax.set_position((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")

    # Project boundary to a Lambert Azimuthal Equal-Area CRS centered on the country.
    # This avoids the distortion that UTM causes for countries spanning multiple zones.
    centroid = boundary.geometry.union_all().centroid
    laea_crs = (
        f"+proj=laea +lat_0={centroid.y} +lon_0={centroid.x} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m"
    )
    boundary_proj = boundary.to_crs(laea_crs)

    # Compute viewport (IQR-based to exclude distant overseas territories).
    overall_centroid = boundary_proj.geometry.union_all().centroid
    seg_centroids = boundary_proj.geometry.centroid
    seg_distances = seg_centroids.distance(overall_centroid)
    q25, q75 = seg_distances.quantile(0.25), seg_distances.quantile(0.75)
    threshold = q75 + 1.5 * (q75 - q25)
    core = boundary_proj[seg_distances <= threshold]
    minx, miny, maxx, maxy = core.total_bounds

    if distance is not None:
        # Use explicit distance (meters) as the viewport radius from centroid
        center_x = overall_centroid.x
        center_y = overall_centroid.y
        half_span = float(distance)
        fig_aspect = width / height
        if fig_aspect > 1:
            half_x = half_span
            half_y = half_span / fig_aspect
        else:
            half_y = half_span
            half_x = half_span * fig_aspect
    else:
        pad_x = (maxx - minx) * 0.15
        pad_y = (maxy - miny) * 0.15

        data_width = (maxx - minx) + 2 * pad_x
        data_height = (maxy - miny) + 2 * pad_y
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2

        fig_aspect = width / height
        data_aspect = data_width / data_height

        if data_aspect > fig_aspect:
            half_x = data_width / 2
            half_y = half_x / fig_aspect
        else:
            half_y = data_height / 2
            half_x = half_y * fig_aspect

    # --- Render subdivision borders (optional, behind country border) ---
    if subdivisions is not None and not subdivisions.empty:
        sub_proj = subdivisions.to_crs(laea_crs)
        for geom in sub_proj.geometry:
            for ring in _extract_rings(geom):
                x, y = ring.coords.xy
                (line,) = ax.plot(
                    x,
                    y,
                    color=THEME.get("subdivision", THEME["text"]),
                    linewidth=0.3,
                    alpha=0.4,
                    zorder=0.5,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                )
                line.set_antialiased(False)

    # --- Render country boundary ---
    for ring in _extract_rings_from_gdf(boundary_proj):
        x, y = ring.coords.xy
        (line,) = ax.plot(
            x,
            y,
            color=THEME.get("border", THEME["text"]),
            linewidth=0.8,
            zorder=1,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        line.set_antialiased(False)

    ax.set_xlim(center_x - half_x, center_x + half_x)
    ax.set_ylim(center_y - half_y, center_y + half_y)
    ax.set_aspect("equal", adjustable="box")

    # Gradients (Top and Bottom)
    create_gradient_fade(ax, THEME["gradient_color"], location="bottom", zorder=10)
    create_gradient_fade(ax, THEME["gradient_color"], location="top", zorder=10)

    # Calculate scale factor based on smaller dimension (reference 12 inches)
    scale_factor = min(height, width) / 12.0

    # Base font sizes (at 12 inches width)
    base_main = 60
    base_coords = 14
    base_attr = 8

    # Typography - use custom fonts if provided, otherwise use default FONTS
    active_fonts = fonts or FONTS
    if active_fonts:
        font_coords = FontProperties(
            fname=active_fonts["regular"], size=base_coords * scale_factor
        )
        font_attr = FontProperties(
            fname=active_fonts["light"], size=base_attr * scale_factor
        )
    else:
        font_coords = FontProperties(
            family="monospace", size=base_coords * scale_factor
        )
        font_attr = FontProperties(family="monospace", size=base_attr * scale_factor)

    # Format country name based on script type
    if is_latin_script(display_country):
        spaced_name = "  ".join(list(display_country.upper()))
    else:
        spaced_name = display_country

    # Dynamically adjust font size based on name length
    base_adjusted_main = base_main * scale_factor
    name_char_count = len(display_country)

    if name_char_count > 10:
        length_factor = 10 / name_char_count
        adjusted_font_size = max(base_adjusted_main * length_factor, 10 * scale_factor)
    else:
        adjusted_font_size = base_adjusted_main

    if active_fonts:
        font_main_adjusted = FontProperties(
            fname=active_fonts["bold"], size=adjusted_font_size
        )
    else:
        font_main_adjusted = FontProperties(
            family="monospace", weight="bold", size=adjusted_font_size
        )

    # --- BOTTOM TEXT ---
    ax.text(
        0.5,
        0.14,
        spaced_name,
        transform=ax.transAxes,
        color=THEME["text"],
        ha="center",
        fontproperties=font_main_adjusted,
        zorder=11,
    )

    lat, lon = point
    coords = (
        f"{lat:.4f}° N / {lon:.4f}° E"
        if lat >= 0
        else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    )
    if lon < 0:
        coords = coords.replace("E", "W")

    ax.text(
        0.5,
        0.10,
        coords,
        transform=ax.transAxes,
        color=THEME["text"],
        alpha=0.7,
        ha="center",
        fontproperties=font_coords,
        zorder=11,
    )

    ax.plot(
        [0.4, 0.6],
        [0.125, 0.125],
        transform=ax.transAxes,
        color=THEME["text"],
        linewidth=1 * scale_factor,
        zorder=11,
    )

    # --- ATTRIBUTION (bottom right) ---
    if FONTS:
        font_attr = FontProperties(fname=FONTS["light"], size=8)
    else:
        font_attr = FontProperties(family="monospace", size=8)

    ax.text(
        0.98,
        0.02,
        "© OpenStreetMap contributors",
        transform=ax.transAxes,
        color=THEME["text"],
        alpha=0.5,
        ha="right",
        va="bottom",
        fontproperties=font_attr,
        zorder=11,
    )

    # Save
    print(f"Saving to {output_file}...")

    fmt = output_format.lower()
    save_kwargs = dict(
        facecolor=THEME["bg"],
        bbox_inches="tight",
        pad_inches=0.05,
    )

    if fmt == "png":
        save_kwargs["dpi"] = 300

    plt.savefig(output_file, format=fmt, **save_kwargs)

    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    """Print usage examples."""
    print("""
Country Border Poster Generator
================================

Usage:
  python create_map_poster.py --country <country> [options]

Examples:
  # European countries
  python create_map_poster.py -C "France" -t noir
  python create_map_poster.py -C "Italy" -t terracotta
  python create_map_poster.py -C "Germany" -t blueprint

  # With subdivisions (states/provinces)
  python create_map_poster.py -C "Argentina" -t noir -s
  python create_map_poster.py -C "United States" -t midnight_blue --subdivisions

  # Control zoom level with distance (meters from centroid)
  python create_map_poster.py -C "France" -t noir -d 800000
  python create_map_poster.py -C "Japan" -t japanese_ink -d 1500000

  # All themes for one country
  python create_map_poster.py -C "Australia" --all-themes

  # List themes
  python create_map_poster.py --list-themes

Options:
  --country, -C       Country name (required)
  --distance, -d      Viewport radius in meters (controls zoom)
  --subdivisions, -s  Show internal borders (states/provinces)
  --display-country   Override country text displayed on poster
  --theme, -t         Theme name (default: terracotta)
  --all-themes        Generate posters for all themes
  --list-themes       List all available themes

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")


def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return

    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, "r", encoding=FILE_ENCODING) as f:
                theme_data = json.load(f)
                display_name = theme_data.get("name", theme_name)
                description = theme_data.get("description", "")
        except (OSError, json.JSONDecodeError):
            display_name = theme_name
            description = ""
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful country border posters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --country "France" --theme noir
  python create_map_poster.py --country "Japan" --theme japanese_ink
  python create_map_poster.py --country "Brazil" --theme emerald
  python create_map_poster.py --list-themes
        """,
    )

    parser.add_argument("--country", "-C", type=str, help="Country name")
    parser.add_argument(
        "--latitude",
        "-lat",
        dest="latitude",
        type=str,
        help="Override latitude for coordinates display",
    )
    parser.add_argument(
        "--longitude",
        "-long",
        dest="longitude",
        type=str,
        help="Override longitude for coordinates display",
    )
    parser.add_argument(
        "--theme",
        "-t",
        type=str,
        default="terracotta",
        help="Theme name (default: terracotta)",
    )
    parser.add_argument(
        "--all-themes",
        "--All-themes",
        dest="all_themes",
        action="store_true",
        help="Generate posters for all themes",
    )
    parser.add_argument(
        "--width",
        "-W",
        type=float,
        default=12,
        help="Image width in inches (default: 12, max: 20)",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=float,
        default=16,
        help="Image height in inches (default: 16, max: 20)",
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=int,
        default=None,
        help="Viewport radius in meters from country centroid (controls zoom level)",
    )
    parser.add_argument(
        "--subdivisions",
        "-s",
        action="store_true",
        help="Show internal subdivision borders (states/provinces)",
    )
    parser.add_argument(
        "--list-themes", action="store_true", help="List all available themes"
    )
    parser.add_argument(
        "--display-country",
        "-dC",
        type=str,
        help="Custom display name for country (for i18n support)",
    )
    parser.add_argument(
        "--font-family",
        type=str,
        help='Google Fonts family name (e.g., "Noto Sans JP", "Open Sans"). If not specified, uses local Roboto fonts.',
    )
    parser.add_argument(
        "--format",
        "-f",
        default="png",
        choices=["png", "svg", "pdf"],
        help="Output format for the poster (default: png)",
    )

    args = parser.parse_args()

    # If no arguments provided, show examples
    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)

    # List themes if requested
    if args.list_themes:
        list_themes()
        sys.exit(0)

    # Validate required arguments
    if not args.country:
        print("Error: --country is required.\n")
        print_examples()
        sys.exit(1)

    # Enforce maximum dimensions
    if args.width > 20:
        print(
            f"⚠ Width {args.width} exceeds the maximum allowed limit of 20. It's enforced as max limit 20."
        )
        args.width = 20.0
    if args.height > 20:
        print(
            f"⚠ Height {args.height} exceeds the maximum allowed limit of 20. It's enforced as max limit 20."
        )
        args.height = 20.0

    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        sys.exit(1)

    if args.all_themes:
        themes_to_generate = available_themes
    else:
        if args.theme not in available_themes:
            print(f"Error: Theme '{args.theme}' not found.")
            print(f"Available themes: {', '.join(available_themes)}")
            sys.exit(1)
        themes_to_generate = [args.theme]

    print("=" * 50)
    print("Country Border Poster Generator")
    print("=" * 50)

    # Load custom fonts if specified
    custom_fonts = None
    if args.font_family:
        custom_fonts = load_fonts(args.font_family)
        if not custom_fonts:
            print(f"⚠ Failed to load '{args.font_family}', falling back to Roboto")

    try:
        # Fetch country boundary
        print(f"Fetching boundary for {args.country}...")
        boundary = fetch_country_boundary(args.country)
        if boundary is None:
            print(f"Error: Could not find boundary for {args.country}")
            sys.exit(1)

        # Fetch subdivisions if requested
        subdiv_gdf = None
        if args.subdivisions:
            print(f"Fetching subdivisions for {args.country}...")
            subdiv_gdf = fetch_subdivisions(args.country)

        # Get coordinates for display
        if args.latitude and args.longitude:
            lat = parse(args.latitude)
            lon = parse(args.longitude)
            coords = (lat, lon)
            print(f"✓ Coordinates (override): {lat}, {lon}")
        else:
            centroid = boundary.geometry.union_all().centroid
            coords = (centroid.y, centroid.x)
            print(f"✓ Coordinates (centroid): {coords[0]:.4f}, {coords[1]:.4f}")

        for theme_name in themes_to_generate:
            THEME = load_theme(theme_name)
            output_file = generate_output_filename(
                args.country, theme_name, args.format
            )
            create_poster(
                args.country,
                coords,
                boundary,
                output_file,
                args.format,
                args.width,
                args.height,
                display_country=args.display_country,
                fonts=custom_fonts,
                distance=args.distance,
                subdivisions=subdiv_gdf,
            )

        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
