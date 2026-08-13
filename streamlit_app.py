import streamlit as st
from PIL import Image, ImageCms
import numpy as np
import io
import zipfile
import re
import os

st.set_page_config(page_title="Batch Image Cropper", layout="wide", page_icon="🖼️")
st.title("🖼️ Batch Image Processor — Crop & Frame to Reference")

st.markdown(
    """
    Upload a **reference image** (defines the horizontal/vertical margins — i.e. how
    much "air" the subject has, and how far it sits from the bottom edge / "floor"),
    then upload the **batch of JPG images** to process. Only files whose filename
    (before the extension) ends in a letter from **a to f** (upper or lower case)
    will be cropped and reframed. Ex: `product_1a.jpg`, `bag_B.JPG` ✅ —
    `product_1g.jpg` ❌ (kept as-is in the output).

    **Positioning rule:** every processed image is centered **horizontally** on the
    canvas, and placed at the **same distance from the floor** (bottom margin) as
    the reference — not vertically centered as a box. This keeps a consistent
    baseline across different angles/shapes of the same product.

    The download is a **complete folder ZIP**: matching files come out cropped,
    reframed and re-encoded to spec; everything else in the batch is included
    unchanged, so you get back the whole folder.
    """
)

OUTPUT_SIZE = 2000  # fixed output canvas: 2000x2000 px
OUTPUT_DPI = (300, 300)

# ----------------------------------------------------------------------------
# Sidebar — parameters
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
tolerance = st.sidebar.slider(
    "White tolerance (threshold)",
    min_value=200,
    max_value=255,
    value=245,
    help=(
        "A pixel is treated as WHITE BACKGROUND if all 3 RGB channels are greater than "
        "or equal to this value. Lower it to also strip soft shadows; raise it if the "
        "subject has very light areas that shouldn't be cropped."
    ),
)
jpg_quality = st.sidebar.slider(
    "Output JPG quality",
    min_value=70,
    max_value=100,
    value=95,
    help="95–100 corresponds to Photoshop's 'Maximum' (≈10-12) quality tier.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Color profile")
icc_file = st.sidebar.file_uploader(
    "Adobe RGB (1998) ICC profile (optional)",
    type=None,
    help=(
        "Upload an AdobeRGB1998.icc file to have images properly color-converted "
        "(not just re-tagged) from sRGB into Adobe RGB and embedded with that profile. "
        "This file usually ships with Photoshop/Bridge, or can be downloaded free from "
        "Adobe. Without it, images are saved as standard sRGB JPEGs."
    ),
)
if icc_file:
    st.sidebar.success("Adobe RGB profile loaded — outputs will be color-converted.")
else:
    st.sidebar.warning("No ICC profile uploaded — outputs will stay in sRGB.")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Output spec: JPG · {'Adobe RGB' if icc_file else 'sRGB'} · 8-bit · "
    f"{OUTPUT_SIZE}×{OUTPUT_SIZE}px · 300dpi · quality {jpg_quality}"
)

FILENAME_PATTERN = re.compile(r"^.*[a-fA-F]$")


# ----------------------------------------------------------------------------
# Image processing functions
# ----------------------------------------------------------------------------
def get_bbox(img: Image.Image, tolerance: int):
    """Bounding box (left, top, right, bottom) of the non-white content."""
    arr = np.array(img.convert("RGB"))
    mask = np.any(arr < tolerance, axis=2)
    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    return (int(left), int(top), int(right) + 1, int(bottom) + 1)


def trim_white(img: Image.Image, tolerance: int):
    """Crops away the excess white background, keeping only the subject's bbox."""
    bbox = get_bbox(img, tolerance)
    if bbox is None:
        return img
    return img.crop(bbox)


def analyze_reference(ref_img: Image.Image, tolerance: int):
    """Extracts margin fractions from the reference: left/right (for width fit)
    and bottom (the fixed 'floor' distance every output will match)."""
    ref_img = ref_img.convert("RGB")
    W, H = ref_img.size
    bbox = get_bbox(ref_img, tolerance)
    if bbox is None:
        return 0.0, 0.0, 0.0  # margin_left, margin_right, margin_bottom
    left, top, right, bottom = bbox
    margin_left = left / W
    margin_right = (W - right) / W
    margin_bottom = (H - bottom) / H
    return margin_left, margin_right, margin_bottom


def process_image(
    img: Image.Image,
    tolerance: int,
    margin_left: float,
    margin_right: float,
    margin_bottom: float,
):
    """Trims the white background off the subject and mounts it on a fixed
    OUTPUT_SIZE x OUTPUT_SIZE canvas: horizontally centered, with its bottom
    edge placed at the same floor distance as the reference."""
    img = img.convert("RGB")
    cropped = trim_white(img, tolerance)
    obj_w, obj_h = cropped.size

    avail_w = max(OUTPUT_SIZE * (1 - margin_left - margin_right), 1)
    floor_px = int(round(margin_bottom * OUTPUT_SIZE))
    avail_h = max(OUTPUT_SIZE - floor_px, 1)  # space from canvas top down to the floor line

    scale = min(avail_w / obj_w, avail_h / obj_h)
    new_w = max(1, int(round(obj_w * scale)))
    new_h = max(1, int(round(obj_h * scale)))
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE), (255, 255, 255))

    paste_x = (OUTPUT_SIZE - new_w) // 2  # exact horizontal center
    paste_y = OUTPUT_SIZE - floor_px - new_h  # same distance from the floor

    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def save_to_spec(img: Image.Image, quality: int, icc_bytes: bytes | None):
    """Encodes the final image per spec: JPG, 8-bit, 300dpi, optional Adobe RGB
    conversion (real color conversion, not just a tag) if an ICC profile was
    supplied; otherwise plain sRGB."""
    out_buffer = io.BytesIO()
    icc_out = None

    if icc_bytes:
        try:
            srgb_profile = ImageCms.createProfile("sRGB")
            adobergb_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
            img = ImageCms.profileToProfile(img, srgb_profile, adobergb_profile, outputMode="RGB")
            icc_out = adobergb_profile.tobytes()
        except Exception as e:
            st.warning(f"Could not apply the Adobe RGB profile ({e}) — saving as sRGB instead.")
            icc_out = None

    save_kwargs = dict(format="JPEG", quality=quality, dpi=OUTPUT_DPI, optimize=True)
    if icc_out:
        save_kwargs["icc_profile"] = icc_out
    img.save(out_buffer, **save_kwargs)
    return out_buffer.getvalue()


# ----------------------------------------------------------------------------
# File uploads
# ----------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    ref_file = st.file_uploader("📐 Reference image", type=["jpg", "jpeg", "png"])
with col2:
    batch_files = st.file_uploader(
        "📦 Batch of images to process (multi-select)",
        type=["jpg", "jpeg"],
        accept_multiple_files=True,
    )

ref_data = None
if ref_file:
    ref_img = Image.open(ref_file)
    ml, mr, mb = analyze_reference(ref_img, tolerance)
    ref_data = (ml, mr, mb)
    st.success(
        f"Reference margins → left {ml:.1%} · right {mr:.1%} · floor (bottom) {mb:.1%} "
        f"— every output will share this floor distance and be horizontally centered."
    )
    st.image(ref_img, caption="Reference image", width=220)

matching_files = []
other_files = []
if batch_files:
    matching_files = [
        f for f in batch_files if FILENAME_PATTERN.match(os.path.splitext(f.name)[0])
    ]
    other_files = [f for f in batch_files if f not in matching_files]
    st.info(
        f"✅ {len(matching_files)} of {len(batch_files)} file(s) match the naming filter "
        f"(end in a–f) and will be cropped and reframed. The remaining "
        f"{len(other_files)} will be copied into the ZIP unchanged."
    )
    if other_files:
        with st.expander(f"See {len(other_files)} unmatched file(s)"):
            st.write([f.name for f in other_files])

# ----------------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------------
if ref_data and batch_files:
    if st.button("🚀 Process batch", type="primary"):
        ml, mr, mb = ref_data
        icc_bytes = icc_file.read() if icc_file else None
        zip_buffer = io.BytesIO()
        progress = st.progress(0)
        status = st.empty()

        preview_cols = st.columns(4)
        preview_count = 0
        total = len(batch_files)

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1) Process the matching files
            for i, f in enumerate(matching_files):
                status.text(f"Processing: {f.name}")
                try:
                    img = Image.open(f)
                    result = process_image(img, tolerance, ml, mr, mb)
                    jpeg_bytes = save_to_spec(result, jpg_quality, icc_bytes)
                    zf.writestr(f.name, jpeg_bytes)

                    if preview_count < 4:
                        with preview_cols[preview_count]:
                            st.image(result, caption=f.name, use_container_width=True)
                        preview_count += 1
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")
                progress.progress((i + 1) / total)

            # 2) Copy the non-matching files through unchanged, to keep the folder complete
            for j, f in enumerate(other_files):
                status.text(f"Copying (unchanged): {f.name}")
                try:
                    f.seek(0)
                    zf.writestr(f.name, f.read())
                except Exception as e:
                    st.error(f"Error copying {f.name}: {e}")
                progress.progress((len(matching_files) + j + 1) / total)

        status.text("Done!")
        st.success(
            f"{len(matching_files)} image(s) cropped, reframed & re-encoded to spec, "
            f"{len(other_files)} copied unchanged — {total} total in the ZIP."
        )
        st.download_button(
            label="⬇️ Download complete folder (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="processed_folder.zip",
            mime="application/zip",
        )
elif ref_file and not batch_files:
    st.warning("Also upload the batch of images to process.")
elif batch_files and not ref_file:
    st.warning("Also upload the reference image.")
