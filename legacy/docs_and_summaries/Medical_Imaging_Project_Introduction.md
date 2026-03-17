# Medical Imaging Project: Kidney Analysis with Multi-Phase CT Scans

## Project Overview

This project focuses on advanced kidney analysis using multi-phase CT (Computed Tomography) imaging data. The study involves comprehensive segmentation, measurement, and functional assessment of kidney function through contrast-enhanced CT scans and eGFR (estimated Glomerular Filtration Rate) calculations.

## 1. CT Scan Data with Different Contrast Phases

### Multi-Phase CT Imaging Protocol

Multi-phase CT imaging is a sophisticated technique that captures images at different time points after contrast agent injection, allowing for comprehensive assessment of kidney function and anatomy.

#### **Arterial Phase (Early Phase)**
- **Timing**: 15-25 seconds after contrast injection
- **Purpose**: Captures the arterial blood supply to the kidneys
- **Key Features**:
  - High contrast in renal arteries
  - Clear visualization of arterial anatomy
  - Assessment of arterial stenosis or abnormalities
  - Early enhancement of kidney cortex

#### **Venous Phase (Portal Phase)**
- **Timing**: 60-90 seconds after contrast injection
- **Purpose**: Captures venous drainage and parenchymal enhancement
- **Key Features**:
  - Peak enhancement of kidney parenchyma
  - Visualization of renal veins
  - Assessment of venous drainage
  - Optimal for parenchymal volume measurements

#### **Late Phase (Delayed Phase)**
- **Timing**: 3-5 minutes after contrast injection
- **Purpose**: Captures contrast excretion and collecting system
- **Key Features**:
  - Contrast in renal collecting system
  - Assessment of kidney function
  - Visualization of ureters and bladder
  - Evaluation of contrast excretion

### Technical Specifications
- **Image Format**: DICOM (Digital Imaging and Communications in Medicine)
- **File Structure**: Organized by case numbers (1-25)
- **Data Organization**:
  - `IOD_arterial/`: Arterial phase DICOM files
  - `IOD_venous/`: Venous phase DICOM files  
  - `IOD_late/`: Late phase DICOM files

## 2. Kidney Segmentation and Analysis

### Segmentation Process

The kidney segmentation process involves precise delineation of kidney structures from CT images using advanced image processing techniques.

#### **Segmented Structures**
1. **Kidney Parenchyma**: The functional tissue of the kidney
2. **Renal Arteries**: Blood vessels supplying the kidney
3. **Renal Veins**: Blood vessels draining the kidney
4. **Cortex**: Outer layer of kidney tissue
5. **Medulla**: Inner region of kidney tissue

#### **Segmentation Output Files**
- **`.seg.nrrd`**: Segmentation files in NRRD format
- **`.nrrd`**: Volume data files
- **`.csv`**: Measurement tables with quantitative data
- **`.mrml`**: 3D Slicer scene files for visualization

#### **Quantitative Measurements**
The segmentation provides detailed measurements including:
- **Volume measurements** (mm³, cm³)
- **Voxel counts** for each segmented structure
- **Intensity statistics** (mean, median, standard deviation)
- **Percentile values** (5th, 95th percentiles)
- **Minimum and maximum intensity values**

### Example Measurement Data
```
Segment: arterial_threshold_1
- Volume: 1,639.82 cm³
- Mean intensity: 194.116 HU
- Standard deviation: 98.376 HU
- Voxel count: 2,944,174
```

## 3. eGFR (Estimated Glomerular Filtration Rate) Calculations

### What is eGFR?

eGFR is a crucial measure of kidney function that estimates how well the kidneys are filtering waste from the blood. It's calculated using serum creatinine levels and other factors.

### Clinical Significance
- **Normal eGFR**: >90 mL/min/1.73m²
- **Mild reduction**: 60-89 mL/min/1.73m²
- **Moderate reduction**: 30-59 mL/min/1.73m²
- **Severe reduction**: 15-29 mL/min/1.73m²
- **Kidney failure**: <15 mL/min/1.73m²

### Data Structure in This Project

The eGFR data is stored in CSV files with the following structure:
- **Patient ID**: Unique identifier for each case
- **Date**: Date of eGFR measurement
- **eGFR Value**: Calculated eGFR in mL/min/1.73m²
- **Serum Creatinine**: Laboratory value used in calculation
- **Measurement Type**: Method used for eGFR calculation

### Longitudinal Analysis
The project includes longitudinal eGFR data showing:
- **Multiple measurements** over time for each patient
- **Trend analysis** of kidney function
- **Correlation** between imaging findings and functional decline

## 4. Project Data Structure

### Case Organization
```
Cases/
├── 1/                    # Case number
│   ├── DICOM/           # Original DICOM files
│   │   ├── IOD_arterial/ # Arterial phase images
│   │   ├── IOD_venous/   # Venous phase images
│   │   └── IOD_late/     # Late phase images
│   ├── eGFR_1.csv       # eGFR measurements
│   └── Segmenteringer/  # Segmentation results
│       ├── *.seg.nrrd   # Segmentation files
│       ├── *.nrrd       # Volume files
│       ├── *.csv        # Measurement tables
│       └── *.mrml       # 3D Slicer scenes
```

### Key File Types
- **DICOM**: Medical imaging standard format
- **NRRD**: Nearly Raw Raster Data format for volumes
- **SEG.NRRD**: Segmentation data format
- **CSV**: Tabular data for measurements
- **MRML**: 3D Slicer scene files

## 5. Clinical Applications

### Diagnostic Capabilities
1. **Kidney Function Assessment**: Quantitative evaluation of kidney performance
2. **Disease Progression**: Longitudinal monitoring of kidney function
3. **Treatment Planning**: Data-driven decisions for patient care
4. **Research Applications**: Large-scale analysis of kidney health

### Research Value
- **Population Studies**: Analysis across multiple patients
- **Method Validation**: Comparison of different measurement techniques
- **Clinical Outcomes**: Correlation between imaging and functional data
- **Biomarker Development**: Identification of imaging biomarkers

## 6. Technical Tools and Software

### Primary Software
- **3D Slicer**: Open-source medical imaging platform
- **Python**: Data analysis and processing
- **Jupyter Notebooks**: Interactive analysis environment

### Data Analysis Capabilities
- **Volume Calculations**: Precise kidney volume measurements
- **Intensity Analysis**: Quantitative assessment of contrast enhancement
- **Statistical Analysis**: Comprehensive data processing
- **Visualization**: 3D rendering and multi-planar views

## 7. Future Directions

### Potential Enhancements
1. **Machine Learning**: Automated segmentation and analysis
2. **Predictive Modeling**: Risk stratification based on imaging data
3. **Standardization**: Development of standardized protocols
4. **Integration**: Connection with electronic health records

### Research Opportunities
- **Biomarker Discovery**: Identification of new imaging biomarkers
- **Population Health**: Large-scale epidemiological studies
- **Precision Medicine**: Personalized treatment approaches
- **Clinical Trials**: Support for interventional studies

---

*This project represents a comprehensive approach to kidney analysis using modern medical imaging techniques, combining structural assessment through CT segmentation with functional evaluation through eGFR calculations.*
