# Phenom SEM Automated Grid & Zoom Acquisition

## Overview
`0501.py` is a **Python automation script** that drives a **Thermo‑Fisher Phenom desktop scanning electron microscope (SEM)** through the `PyPhenom` SDK.  
The script performs three major tasks:

1. **Fine Zoom Sweep** – captures a 360‑frame logarithmic zoom series (`zoom/zoom_###.tiff`), ranging roughly from 1 mm HFW down to 1 µm HFW.  
2. **Large‑Area Mosaic 1** – acquires a 30 × 30 raster of high‑resolution images (`pd1/row##_col##.tiff`) with 40 % overlap (0.6× HFW step) at **10 µm HFW**.  
3. **Large‑Area Mosaic 2** – acquires a second 30 × 30 raster (`pd2/row##_col##.tiff`) with 10 % overlap (0.9× HFW step) at **30 µm HFW**.

Each capture **auto‑focuses** and optionally **adds a data bar** before saving a 1600 × 1600 pixel TIFF image.

---

## Repository Layout
```
.
└── 0501.py            # main automation script
```

---

## Prerequisites

| Requirement            | Version / Notes                                  |
|------------------------|--------------------------------------------------|
| **Python**             | ≥ 3.8 (tested with 3.10)                         |
| **PyPhenom**           | Thermo‐Fisher SDK (≥ 2.5.0) with valid license   |
| **Phenom SEM**         | Connected via Ethernet & unlocked for remote API |
| **tqdm**               | `pip install tqdm` (progress bars)               |
| **NumPy**              | `pip install numpy`                              |

> **Important:** Copy `PyPhenom.pyd`, `PyPhenom.dll`, and `DetectorMode.dll` supplied by Thermo‑Fisher into your Python site‑packages (or add their folder to `PATH`) before `import PyPhenom` will work.

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
       "MVE092827-20102-F",    # instrument name
       "MVE09282720102F",      # serial
       "8RZ9C72BYK74",         # API key
   )
   ```

4. **Create output folders**

   ```bash
   mkdir -p zoom pd1 pd2 pd3
   ```

5. **Run the script**

   ```bash
   python 0501.py
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

## Customisation

| Variable                | Purpose                                 |
|-------------------------|-----------------------------------------|
| `row`, `col`            | Grid dimensions for mosaics             |
| `scale factor` (0.6/0.9)| Overlap between adjacent tiles          |
| `SetHFW()` values       | Working magnification / field width     |
| `nFrames`               | Signal / noise trade‑off                |
| `detector`              | `ppi.DetectorMode.All` ↔ SE/BSE         |

Feel free to modify these before running or expose them as CLI arguments.

---

## Safety & Tips

* **Stage limits** – Make sure your sample height does not exceed the Z‑clearance the grid scan will require.  
* **Vacuum stability** – Allow the chamber pump‑down to finish before starting automation.  
* **File naming** – TIFF filenames embed grid indices so you can reconstruct mosaics later.  
* **Large datasets** – The full run generates \~2.5 GB of TIFFs; ensure adequate disk space.

---

## Troubleshooting

| Symptom | Possible Cause | Remedy |
|---------|----------------|--------|
| `OSError: DLL load failed` | PyPhenom binaries not found | Add DLL folder to `PATH` or copy into site‑packages |
| Script hangs on `SemAcquireImage` | SEM not in SEM mode | Manually switch to SEM view, disable navigation camera |
| Images out of focus | Sample too rough / height uneven | Adjust `SemAutoFocus()` logic or pre‑tilt sample holder |
| Progress stalls at grid edge | Stage travel limit reached | Reduce `row`/`col` or scale factor, verify `MoveTo` coords |

---

## License
MIT License – see `LICENSE` (or adapt as needed).

## Author
**Zhen Zhang** – Electrochemical Automation Lab

---

## Citation
> Zhang, Z. *et al.* “Automated Grid and Zoom SEM Imaging with PyPhenom.” (2025).

---

## Acknowledgements
We thank Thermo‑Fisher Scientific for providing the PyPhenom SDK and academic licensing.
