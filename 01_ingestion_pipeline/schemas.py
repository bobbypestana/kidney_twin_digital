from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date

class BronzeSlicerRecord(BaseModel):
    """Pydantic model for a single row in the bronze slicer table."""
    record_id: int
    phase: str
    structure: str
    volume_cm3: Optional[float] = None
    hu_mean: Optional[float] = None
    hu_std: Optional[float] = None
    hu_median: Optional[float] = None
    source_folder: str

class SilverSegmentationRecord(BaseModel):
    """Pydantic model for the silver segmentations table (wide format)."""
    record_id: str
    scan_date: Optional[date] = None
    current_age: Optional[float] = None
    sex: Optional[str] = None
    # Dynamic columns for HU/Volume statistics will be validated partially
    # or left as a flexible dict if needed, but for now we define the core.

class GoldFeatureRecord(BaseModel):
    """Pydantic model for the final ML features in the gold layer."""
    record_id: str
    egfrc: float
    current_age: float
    sex: float  # 1.0 for M, 0.0 for F
    
    # Core physiological ratios/interactions
    E_arterial_mean: Optional[float] = None
    art_flow_efficiency: Optional[float] = None
    age_x_E_arterial: Optional[float] = None
    
    @validator('sex')
    def validate_sex(cls, v):
        if v not in [0.0, 1.0]:
            raise ValueError('Sex must be 0.0 or 1.0')
        return v
