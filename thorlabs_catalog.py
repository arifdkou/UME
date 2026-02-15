# thorlabs_catalog.py
# Minimal embedded “catalog” so GUI works offline.
# You can extend this list later (or load from a CSV you maintain).

LENSES = [
    {"name": "LA1131-A", "f_mm": 50.0,  "T": 0.995, "note": "Ø1\", AR 350-700"},
    {"name": "LA1433-A", "f_mm": 100.0, "T": 0.995, "note": "Ø1\", AR 350-700"},
    {"name": "LA1608-A", "f_mm": 160.0, "T": 0.995, "note": "Ø1\", AR 350-700"},
    {"name": "LA1509-A", "f_mm": 200.0, "T": 0.995, "note": "Ø1\", AR 350-700"},
]

FIBERS = [
    {"name": "SMF-ish (NA 0.12, 9um)", "na": 0.12, "core_um": 9.0, "eta0": 0.75},
    {"name": "MMF (NA 0.22, 50um)",    "na": 0.22, "core_um": 50.0, "eta0": 0.80},
]

BEAMSPLITTERS = [
    {"name": "BS 50/50 (ideal)", "split": 0.50, "abs": 0.01},
    {"name": "BS 45/55 (realistic)", "split": 0.45, "abs": 0.02},
]

AOMS = [
    {"name": "AOM off (bypass)", "eta_1st": 1.0},
    {"name": "AOM typical", "eta_1st": 0.80},
]
