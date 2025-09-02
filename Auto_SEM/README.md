# Phenom SEM Automated Grid & Zoom Acquisition

## Overview
Running the crest/main.py file which contains the **Python automation script** that drives a **Thermo‑Fisher Phenom desktop scanning electron microscope (SEM)** through the `PyPhenom` SDK.  
The script performs three major tasks:

1. **Fine Zoom Sweep** – captures a 360‑frame logarithmic zoom series (`zoom/zoom_###.tiff`), ranging roughly from 1 mm HFW down to 1 µm HFW.  
2. **Large‑Area Mosaic 1** – acquires a 30 × 30 raster of high‑resolution images (`pd1/row##_col##.tiff`) with 40 % overlap (0.6× HFW step) at **10 µm HFW**.  
3. **Large‑Area Mosaic 2** – acquires a second 30 × 30 raster (`pd2/row##_col##.tiff`) with 10 % overlap (0.9× HFW step) at **30 µm HFW**.

Each capture **auto‑focuses** and optionally **adds a data bar** before saving a 1600 × 1600 pixel TIFF image.

---

## Quick Start

1. **Clone or copy the script**

   ```bash
   git clone <repo-url>
   cd <repo>/  # contains 0501.py
   ```

2. **Set up a clean virtual environment**

   ```bash
   conda create -n phenom python=3.10
   conda activate phenom
   pip install numpy tqdm
   # Install PyPhenom wheel or copy binaries manually
   ```

3. **Edit connection credentials**  
   At the top of `0501.py`, replace the placeholders with your microscope’s **instrument name**, **serial number**, and **API key**:

   ```python
   phenom = ppi.Phenom(
       "xxx",    # instrument name
       "xxx",      # serial
       "xxx",         # API key
   )
   ```

4. **Create output folders**

   ```bash
   mkdir -p zoom pd1 pd2 pd3
   ```

5. **Run the script**

   ```bash
   python Auto_SEM/main.py
   ```

   Progress bars appear in the terminal. Three acquisition phases run back‑to‑back and save images to the corresponding folders.

---

## Script Logic

| Section | Function | Parameters |
|---------|----------|------------|
| **Zoom Sweep** | `index = np.linspace(-3, -5, 360)` → `phenom.SetHFW(10**idx)` | Steps HFW from 1 mm → 10 nm (log scale), `nFrames = 8` |
| **Mosaic 1** | `phenom.SetHFW(10e-6)` | Grid 30 × 30, **step = 0.6 × HFW** (≈ 40 % overlap) |
| **Mosaic 2** | `phenom.SetHFW(30e-6)` | Grid 30 × 30, **step = 0.9 × HFW** (≈ 10 % overlap) |
| **Zoom Sweep 2** | `index = np.linspace(-3, -6, 400)` | HFW 1 mm → 1 nm (finer), `nFrames = 16` |

All acquisitions call:

```python
ppo.SemAutoFocus()
ppi.AddDatabar(acq)      # adds scale bar & metadata text
ppi.Save(acqWithDatabar, "folder/filename.tiff")
```

---

## Customization

| Variable                | Purpose                                 |
|-------------------------|-----------------------------------------|
| `row`, `col`            | Grid dimensions for mosaics             |
| `scale factor` (0.6/0.9)| Overlap between adjacent tiles          |
| `SetHFW()` values       | Working magnification / field width     |
| `nFrames`               | Signal / noise trade‑off                |
| `detector`              | `ppi.DetectorMode.All` ↔ SE/BSE         |

Feel free to modify these before running or expose them as CLI arguments.

## Example

An example of stitched SEM image is given, with google drive link provided.

---

