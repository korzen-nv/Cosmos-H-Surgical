# Segmentation Class Mappings to Unified Open-H Format

This document defines the class mappings from four cholecystectomy segmentation datasets
to the unified **Open-H** 8-class format used by the Cosmos-H-Surgical-Transfer segmentation controlnet.

## Open-H Target Classes

| ID | Class              | RGB Color       |
|----|--------------------|-----------------|
| 0  | Background         | (0, 0, 0)      |
| 1  | Abdominal Wall     | (64, 64, 64)   |
| 2  | Liver              | (255, 73, 40)  |
| 3  | Gallbladder        | (255, 45, 255) |
| 4  | Fat                | (57, 4, 105)   |
| 5  | Connective Tissue  | (137, 126, 8)  |
| 6  | Instruments        | (100, 228, 3)  |
| 7  | Other Anatomy      | (74, 74, 138)  |

Background (0,0,0) is treated as "no segmentation guidance" by the controlnet — black pixels
mean the model generates freely in those regions.

---

## Atlas120k (47 classes)

**Source:** `/data/atlas120k/CLASS_MAPPING.txt`
**Mask format:** PNG indexed-color (mode=P), class IDs as pixel values
**Notes:** Some clips have RGB-mode masks (non-standard) — skip those.

### Source Classes

```python
color_palette = {
    # Abdomen IDs
    1: (255, 255, 255),  # Tools/camera - White
    2: (0, 0, 255),      # Vein (major) - Blue
    3: (255, 0, 0),      # Artery (major) - Red
    4: (255, 255, 0),    # Nerve (major) - Yellow
    5: (0, 255, 0),      # Small intestine - Green
    6: (0, 200, 100),    # Colon/rectum - Dark Green
    7: (200, 150, 100),  # Abdominal wall - Beige
    8: (250, 150, 100),  # Diaphragm - light Beige
    9: (255, 200, 100),  # Omentum - Light Orange
    10: (180, 0, 0),     # Aorta - Dark Red
    11: (0, 0, 180),     # Vena cava - Dark Blue
    12: (150, 100, 50),  # Liver - Brown
    13: (0, 255, 255),   # Cystic duct - Cyan
    14: (0, 200, 255),   # Gallbladder - Teal
    15: (0, 100, 255),   # Hepatic vein - Light Blue
    16: (255, 150, 50),  # Hepatic ligament - Orange
    17: (255, 220, 200), # Cystic plate - Light Pink
    18: (200, 100, 200), # Stomach - Light Purple
    19: (144, 238, 144), # Ductus choledochus - Light Green
    20: (247, 255, 0),   # Mesenterium
    21: (255, 206, 27),  # Ductus hepaticus - Red
    22: (200, 0, 200),   # Spleen - Purple
    23: (255, 0, 150),   # Uterus - Pink
    24: (255, 100, 200), # Ovary - Light Pink
    25: (200, 100, 255), # Oviduct - Lavender
    # RARP
    26: (150, 0, 100),   # Prostate - Dark Purple
    27: (255, 200, 255), # Urethra - Light Pink
    28: (150, 100, 75),  # Ligated plexus - Brown
    29: (200, 0, 150),   # Seminal vesicles - Magenta
    30: (100, 100, 100), # Catheter - Gray
    31: (255, 150, 255), # Bladder - Light Purple
    32: (100, 200, 255), # Kidney - Light Blue
    # Thorax IDs
    33: (150, 200, 255), # Lung - Light Blue
    34: (0, 150, 255),   # Airway (bronchus/trachea) - Sky Blue
    35: (255, 100, 100), # Esophagus - Salmon
    36: (200, 200, 255), # Pericardium - Pale Blue
    37: (100, 100, 255), # V azygos - Blue
    38: (0, 255, 150),   # Thoracic duct - Green Cyan
    39: (255, 255, 100), # Nerves - Yellow
    # Non-anatomical structures
    40: (150, 150, 150), # Ureter - Gray
    41: (50, 50, 50),    # Non anatomical structures - Dark Gray
    42: (0, 0, 0),       # Excluded frames - Black (Not Included in final version)
    # Additional structures
    43: (173, 216, 230), # Mesocolon
    44: (255, 140, 0),   # Adrenal Gland
    45: (223, 3, 252),   # Pancreas
    46: (0, 80, 100),    # Duodenum
}
```

### Mapping: Atlas120k → Open-H

| Atlas ID | Atlas Class            | → | Open-H ID | Open-H Class       | Rationale |
|----------|------------------------|---|-----------|--------------------|----|
| 0        | Background             | → | 0 | Background         | Direct |
| 1        | Tools/camera           | → | 6 | Instruments        | Direct |
| 2        | Vein (major)           | → | 7 | Other Anatomy      | Vascular structure |
| 3        | Artery (major)         | → | 7 | Other Anatomy      | Vascular structure |
| 4        | Nerve (major)          | → | 7 | Other Anatomy      | Neural structure |
| 5        | Small intestine        | → | 7 | Other Anatomy      | Organ |
| 6        | Colon/rectum           | → | 7 | Other Anatomy      | Organ |
| 7        | Abdominal wall         | → | 1 | Abdominal Wall     | Direct |
| 8        | Diaphragm              | → | 1 | Abdominal Wall     | Peritoneal boundary |
| 9        | Omentum                | → | 4 | Fat                | Omentum is primarily fatty tissue |
| 10       | Aorta                  | → | 7 | Other Anatomy      | Vascular structure |
| 11       | Vena cava              | → | 7 | Other Anatomy      | Vascular structure |
| 12       | Liver                  | → | 2 | Liver              | Direct |
| 13       | Cystic duct            | → | 5 | Connective Tissue  | Duct in Calot's triangle |
| 14       | Gallbladder            | → | 3 | Gallbladder        | Direct |
| 15       | Hepatic vein           | → | 7 | Other Anatomy      | Vascular structure |
| 16       | Hepatic ligament       | → | 5 | Connective Tissue  | Ligament |
| 17       | Cystic plate           | → | 5 | Connective Tissue  | Connective tissue layer |
| 18       | Stomach                | → | 7 | Other Anatomy      | Organ |
| 19       | Ductus choledochus     | → | 5 | Connective Tissue  | Bile duct |
| 20       | Mesenterium            | → | 5 | Connective Tissue  | Peritoneal fold |
| 21       | Ductus hepaticus       | → | 5 | Connective Tissue  | Hepatic duct |
| 22       | Spleen                 | → | 7 | Other Anatomy      | Organ |
| 41       | Non-anatomical         | → | 0 | Background         | Not anatomy |
| 43       | Mesocolon              | → | 7 | Other Anatomy      | Peritoneal fold |
| 44       | Adrenal Gland          | → | 7 | Other Anatomy      | Organ |
| 45       | Pancreas               | → | 7 | Other Anatomy      | Organ |
| 46       | Duodenum               | → | 7 | Other Anatomy      | Organ |

**Implementation:** `scripts/prepare_atlas120k_openh.py` — `ATLAS_TO_OPENH` dict + LUT-based remapping.

---

## CholecSeg8k (13 classes)

**Source:** CholecSeg8k dataset
**Mask format:** `*_endo_mask.png` — RGB images with identical channels. Pixel values are the
hex-code prefix digits read as decimal (NOT hex-to-decimal conversion).
**Notes:** The documented RGB hex codes (e.g., #505050) encode the pixel value: take the first
two hex characters and read them as a plain decimal number. So #505050 → pixel value 50,
#212121 → pixel value 21, etc. Pixel value 0 is unlabeled/outside-body.

### Source Classes

| Class ID | Class Name                | RGB Hex  | Pixel Value | Notes |
|----------|---------------------------|----------|-------------|-------|
| 0        | Black Background          | #505050  | **50**      | hex "50" → decimal 50 |
| 1        | Abdominal Wall            | #111111  | **11**      | hex "11" → decimal 11 |
| 2        | Liver                     | #212121  | **21**      | hex "21" → decimal 21 |
| 3        | Gastrointestinal Tract    | #131313  | **13**      | hex "13" → decimal 13 |
| 4        | Fat                       | #121212  | **12**      | hex "12" → decimal 12 |
| 5        | Grasper                   | #313131  | **31**      | hex "31" → decimal 31 |
| 6        | Connective Tissue         | #232323  | **23**      | hex "23" → decimal 23 |
| 7        | Blood                     | #242424  | **24**      | hex "24" → decimal 24 |
| 8        | Cystic Duct               | #252525  | **25**      | hex "25" → decimal 25 |
| 9        | L-hook Electrocautery     | #323232  | **32**      | hex "32" → decimal 32 |
| 10       | Gallbladder               | #222222  | **22**      | hex "22" → decimal 22 |
| 11       | Hepatic Vein              | #333333  | **33**      | hex "33" → decimal 33 |
| 12       | Liver Ligament            | #050505  | **5**       | hex "05" → decimal 5  |
| —        | Unlabeled                 | —        | **0**       | Outside body / not annotated |

All unique pixel values observed across dataset: `[0, 5, 11, 12, 13, 21, 22, 23, 24, 25, 31, 32, 33, 50]`

### Mapping: CholecSeg8k → Open-H

| Pixel Val | CS8k Class                | → | Open-H ID | Open-H Class       | Rationale |
|-----------|---------------------------|---|-----------|--------------------|----|
| 50        | Black Background          | → | 0 | Background         | Direct |
| 11        | Abdominal Wall            | → | 1 | Abdominal Wall     | Direct |
| 21        | Liver                     | → | 2 | Liver              | Direct |
| 13        | Gastrointestinal Tract    | → | 7 | Other Anatomy      | Organ (stomach/intestine visible in field) |
| 12        | Fat                       | → | 4 | Fat                | Direct |
| 31        | Grasper                   | → | 6 | Instruments        | Surgical instrument |
| 23        | Connective Tissue         | → | 5 | Connective Tissue  | Direct |
| 24        | Blood                     | → | 7 | Other Anatomy      | Transient fluid, not a structural class |
| 25        | Cystic Duct               | → | 5 | Connective Tissue  | Duct in Calot's triangle |
| 32        | L-hook Electrocautery     | → | 6 | Instruments        | Surgical instrument |
| 22        | Gallbladder               | → | 3 | Gallbladder        | Direct |
| 33        | Hepatic Vein              | → | 7 | Other Anatomy      | Vascular structure |
| 5         | Liver Ligament            | → | 5 | Connective Tissue  | Ligament |
| 0         | Unlabeled                 | → | 0 | Background         | Not annotated |

**Note:** CholecSeg8k annotations are boundary/contour-style (not filled regions). Most pixels
are unlabeled (val=0). Use `*_endo_mask.png` files — the `*_endo_color_mask.png` use different
RGB colors for visualization only.

---

## Endoscapes (7 classes)

**Source:** Endoscapes dataset
**Mask format:** Semantic segmentation masks (PNG)
**Notes:** Minimal class set focused on Calot's triangle critical view of safety (CVS) structures.

### Source Classes

| Class Name       | Notes |
|------------------|-------|
| background       | Non-annotated regions |
| cystic_plate     | Connective tissue layer between gallbladder and liver |
| calot_triangle   | Hepatocystic triangle region (connective tissue being dissected) |
| cystic_artery    | Arterial vessel in Calot's triangle |
| cystic_duct      | Ductal structure connecting gallbladder to bile duct |
| gallbladder      | Target organ for removal |
| tool             | Surgical instruments |

### Mapping: Endoscapes → Open-H

| Endoscapes Class | → | Open-H ID | Open-H Class       | Rationale |
|------------------|---|-----------|--------------------|----|
| background       | → | 0 | Background         | Direct |
| cystic_plate     | → | 5 | Connective Tissue  | Connective tissue layer in hepatocystic triangle |
| calot_triangle   | → | 5 | Connective Tissue  | Calot's triangle is the connective tissue region being dissected |
| cystic_artery    | → | 7 | Other Anatomy      | Vascular structure (critical to identify, not connective tissue) |
| cystic_duct      | → | 5 | Connective Tissue  | Duct in Calot's triangle, consistent with Atlas120k mapping |
| gallbladder      | → | 3 | Gallbladder        | Direct |
| tool             | → | 6 | Instruments        | Direct |

**Note:** Endoscapes has no liver, abdominal wall, or fat classes. These structures are
present in the images but not annotated — they will map to Background (0) by default, meaning
the controlnet generates freely in those regions.

---

## HeiSurf (22 classes)

**Source:** HeiSurf dataset (Heidelberg Surgical Finegrained)
**Mask format:** Semantic segmentation masks (PNG)
**Notes:** Richest annotation with non-anatomy classes (gauze, trocar, specimenbag, etc.).
"Uncertainties" and "Censored out of Body" should be excluded/mapped to background.

### Source Classes

| Label                                          | RGB Color       | Notes |
|------------------------------------------------|-----------------|-------|
| Abdominal Wall and Diaphragm                   | (253, 101, 14)  | — |
| Blood Pool                                     | (41, 10, 12)    | — |
| Censored out of Body                           | (62, 62, 62)    | — |
| Clip                                           | (255, 51, 0)    | Surgical clip, metal |
| Cystic Artery                                  | (51, 221, 255)  | — |
| Cystic Duct                                    | (153, 255, 51)  | — |
| Drainage                                       | (1, 68, 34)     | — |
| Fatty Tissue                                   | (255, 255, 0)   | — |
| Gallbladder                                    | (25, 176, 34)   | — |
| Gallbladder Resection Bed                      | (0, 55, 104)    | Exposed liver surface after GB removal |
| Gastrointestinal Tract                         | (74, 165, 255)  | — |
| Gauze                                          | (255, 255, 255) | — |
| Hilum of Liver / Hepatoduodenal Ligament       | (255, 168, 0)   | — |
| Inside of Trocar                               | (160, 56, 133)  | — |
| Instrument                                     | (101, 46, 151)  | — |
| Liver                                          | (136, 0, 0)     | — |
| Other                                          | (204, 204, 204) | Organic structures not annotated further |
| Out of Image                                   | (82, 82, 82)    | Black border of endoscope |
| Round and Falciform Ligament of the Liver      | (61, 245, 61)   | — |
| Specimenbag                                    | (79, 48, 31)    | — |
| Trocar                                         | (255, 51, 119)  | — |
| Uncertainties                                  | (240, 240, 240) | Ignore during training |

### Mapping: HeiSurf → Open-H

| HeiSurf Class                              | → | Open-H ID | Open-H Class       | Rationale |
|--------------------------------------------|---|-----------|--------------------|----|
| Abdominal Wall and Diaphragm               | → | 1 | Abdominal Wall     | Direct (includes diaphragm, same as Atlas120k) |
| Blood Pool                                 | → | 7 | Other Anatomy      | Transient fluid in surgical field |
| Censored out of Body                       | → | 0 | Background         | Not part of surgical scene |
| Clip                                       | → | 6 | Instruments        | Surgical hardware placed by instruments |
| Cystic Artery                              | → | 7 | Other Anatomy      | Vascular structure (consistent with Endoscapes) |
| Cystic Duct                                | → | 5 | Connective Tissue  | Duct in Calot's triangle (consistent across all datasets) |
| Drainage                                   | → | 6 | Instruments        | Surgical device/hardware |
| Fatty Tissue                               | → | 4 | Fat                | Direct |
| Gallbladder                                | → | 3 | Gallbladder        | Direct |
| Gallbladder Resection Bed                  | → | 5 | Connective Tissue  | Exposed connective tissue surface after GB removal |
| Gastrointestinal Tract                     | → | 7 | Other Anatomy      | Organ (consistent with CholecSeg8k, Atlas120k) |
| Gauze                                      | → | 0 | Background         | Non-anatomy, temporary surgical material |
| Hilum of Liver / Hepatoduodenal Ligament   | → | 5 | Connective Tissue  | Ligamentous structure at liver hilum |
| Inside of Trocar                           | → | 0 | Background         | Not part of surgical anatomy |
| Instrument                                 | → | 6 | Instruments        | Direct |
| Liver                                      | → | 2 | Liver              | Direct |
| Other                                      | → | 7 | Other Anatomy      | Unspecified organic structures |
| Out of Image                               | → | 0 | Background         | Endoscope border |
| Round and Falciform Ligament of the Liver  | → | 5 | Connective Tissue  | Ligament (consistent with Atlas120k liver ligament mapping) |
| Specimenbag                                | → | 0 | Background         | Non-anatomy, temporary surgical material |
| Trocar                                     | → | 0 | Background         | External hardware, not part of anatomy |
| Uncertainties                              | → | 0 | Background         | Ignore during training (per dataset docs) |

---

## Cross-Dataset Consistency Notes

The mappings above maintain consistency across all four datasets for the same anatomical structures:

| Anatomical Structure    | Atlas120k      | CholecSeg8k      | Endoscapes       | HeiSurf                | → Open-H |
|-------------------------|----------------|------------------|------------------|------------------------|-----------|
| **Abdominal wall**      | Abdominal wall, Diaphragm | Abdominal Wall | — | Abdominal Wall and Diaphragm | 1: Abdominal Wall |
| **Liver**               | Liver          | Liver            | —                | Liver                  | 2: Liver |
| **Gallbladder**         | Gallbladder    | Gallbladder      | gallbladder      | Gallbladder            | 3: Gallbladder |
| **Fat/omentum**         | Omentum        | Fat              | —                | Fatty Tissue           | 4: Fat |
| **Cystic duct**         | Cystic duct    | Cystic Duct      | cystic_duct      | Cystic Duct            | 5: Connective Tissue |
| **Cystic plate**        | Cystic plate   | —                | cystic_plate     | —                      | 5: Connective Tissue |
| **Calot's triangle**    | —              | —                | calot_triangle   | —                      | 5: Connective Tissue |
| **Hepatic ligament**    | Hepatic ligament | Liver Ligament | —                | Hilum/Hepatoduodenal Lig. | 5: Connective Tissue |
| **Liver ligament**      | —              | —                | —                | Round/Falciform Lig.   | 5: Connective Tissue |
| **GB resection bed**    | —              | —                | —                | Gallbladder Resection Bed | 5: Connective Tissue |
| **Instruments**         | Tools/camera   | Grasper, L-hook  | tool             | Instrument             | 6: Instruments |
| **Clips/drainage**      | —              | —                | —                | Clip, Drainage         | 6: Instruments |
| **Cystic artery**       | Artery (major) | —                | cystic_artery    | Cystic Artery          | 7: Other Anatomy |
| **GI tract**            | Small intestine, Stomach | GI Tract | —             | GI Tract               | 7: Other Anatomy |
| **Blood**               | —              | Blood            | —                | Blood Pool             | 7: Other Anatomy |
| **Hepatic vein**        | Hepatic vein   | Hepatic Vein     | —                | —                      | 7: Other Anatomy |

### Design Decisions

1. **Cystic duct → Connective Tissue (not Other Anatomy):** In the context of cholecystectomy,
   the cystic duct is a structure being dissected through connective tissue in Calot's triangle.
   Grouping it with connective tissue reflects the surgical workflow.

2. **Cystic artery → Other Anatomy (not Connective Tissue):** The artery is a vascular structure
   that must be identified and clipped. Keeping it as "Other Anatomy" distinguishes it from the
   connective tissue being dissected, consistent with how Atlas120k maps all vessels.

3. **Blood/Blood Pool → Other Anatomy:** Blood is transient and not a structural class. Mapping
   to Background would lose information; Other Anatomy signals "something is here" to the controlnet.

4. **Non-anatomy items (gauze, trocar, specimenbag, etc.) → Background:** These are not part
   of the surgical anatomy. Mapping to Background (black) tells the controlnet to generate freely
   in those regions, which is the correct behavior.

5. **Clips and drainage → Instruments:** Surgical clips and drainage tubes are placed by instruments
   and represent surgical hardware in the field, similar to instruments.
