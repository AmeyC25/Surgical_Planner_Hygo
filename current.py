import torch
import torch.nn.functional as F
from monai.networks.nets import UNet  # type: ignore
import nibabel as nib  # type: ignore
import numpy as np
import streamlit as st
from openai import OpenAI
import os
import tempfile
import shutil
import zipfile
from dotenv import load_dotenv
from scipy.stats import ttest_ind
import pydicom  # type: ignore
import dicom2nifti  # type: ignore
from pathlib import Path
import io

# ============================
# 🩹 PATCH: Fix torch + Streamlit bug with __path__._path
# ============================
import sys
import types
import streamlit.watcher.local_sources_watcher as st_watcher

def safe_get_module_paths(module):
    """Fixed version that accepts module parameter"""
    module_paths = set()
    
    # Handle the specific module passed in
    if isinstance(module, types.ModuleType):
        try:
            if hasattr(module, "__file__") and module.__file__:
                module_paths.add(module.__file__)
            elif hasattr(module, "__path__") and isinstance(module.__path__, (list, tuple)):
                module_paths.update(module.__path__)
        except Exception:
            pass
    
    return list(module_paths)

# Apply the safe patch
st_watcher.get_module_paths = safe_get_module_paths

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1,
    ).to(device)
    model.load_state_dict(torch.load('segmentation_model.pth', map_location=device))
    model.eval()
    return model

def extract_and_organize_dicoms(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    dicom_series = {}
    try:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        dicom_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path)
                    dicom_files.append((file_path, ds))
                except:
                    continue
        
        for file_path, ds in dicom_files:
            try:
                series_uid = ds.SeriesInstanceUID
                series_desc = getattr(ds, 'SeriesDescription', 'Unknown Series')
                series_number = getattr(ds, 'SeriesNumber', 'Unknown')
                series_key = f"Series {series_number}: {series_desc} (UID: {series_uid[:8]}...)"
                
                if series_key not in dicom_series:
                    dicom_series[series_key] = {
                        'files': [],
                        'series_uid': series_uid,
                        'description': series_desc,
                        'number': series_number
                    }
                
                dicom_series[series_key]['files'].append(file_path)
            except Exception as e:
                continue
        
        return dicom_series, temp_dir
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e

def convert_dicom_to_nifti(dicom_files, output_path):
    try:
        series_temp_dir = tempfile.mkdtemp()
        for i, dicom_file in enumerate(dicom_files):
            shutil.copy2(dicom_file, series_temp_dir)
        
        dicom2nifti.dicom_series_to_nifti(series_temp_dir, output_path, reorient_nifti=True)
        shutil.rmtree(series_temp_dir)
        return True
    except Exception as e:
        st.error(f"Error converting DICOM to NIfTI: {e}")
        return False

def predict_segmentation(model, nifti_path):
    try:
        mri = nib.load(nifti_path)
        mri_image = mri.get_fdata()
        voxel_dims = mri.header.get_zooms()[:3]
        affine = mri.affine
        
        image_tensor = torch.tensor(mri_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        image_tensor = F.interpolate(image_tensor, size=(128, 128, 128), mode='trilinear', align_corners=False)
        
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1).cpu().numpy()
        
        return prediction.squeeze(), voxel_dims, affine
    except Exception as e:
        return f"Error in MRI processing: {e}", None, None

def calculate_p_value(segmentation_result):
    tumor_values = (segmentation_result.flatten() == 1).astype(int)
    ventricle_values = (segmentation_result.flatten() == 2).astype(int)
    
    if np.sum(tumor_values) > 1 and np.sum(ventricle_values) > 1:
        try:
            stat, p_value = ttest_ind(tumor_values, ventricle_values, equal_var=False)
            return f"p-value = {p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not significant'})"
        except Exception as e:
            return f"Error calculating p-value: {e}"
    else:
        return "Statistical comparison not possible (insufficient data)."

def summarize_findings(segmentation_result, voxel_dims, affine, series_info):
    if segmentation_result is None:
        return "Error: Invalid segmentation data."
    
    voxel_volume = np.prod(voxel_dims)
    tumor_voxels = np.sum(segmentation_result == 1)
    ventricles_voxels = np.sum(segmentation_result == 2)
    gray_matter_voxels = np.sum(segmentation_result == 0)
    
    tumor_volume = tumor_voxels * voxel_volume / 1000
    ventricles_volume = ventricles_voxels * voxel_volume / 1000
    gray_matter_volume = gray_matter_voxels * voxel_volume / 1000
    
    tumor_indices = np.argwhere(segmentation_result == 1)
    if tumor_indices.size > 0:
        centroid_voxel = np.mean(tumor_indices, axis=0)
        centroid_mm = nib.affines.apply_affine(affine, centroid_voxel)
        anatomical_location = f"Tumor centroid coordinates (mm): {', '.join(map(str, centroid_mm.round(2).tolist()))}"
    else:
        anatomical_location = "Tumor not clearly detected."
    
    p_value_text = calculate_p_value(segmentation_result)
    
    summary = f"""
    Series Analyzed: {series_info}
    Tumor Volume: {tumor_volume:.2f} cm³
    Ventricles Volume: {ventricles_volume:.2f} cm³
    Gray Matter Volume: {gray_matter_volume:.2f} cm³
    {anatomical_location}
    Statistical Comparison (Tumor vs Ventricles): {p_value_text}
    Data Augmentation Techniques Used: Affine transformations, Intensity shifts.
    Powered by HYGO AI Advanced Medical Imaging Technology
    """
    return summary

def create_download_link(file_path, filename):
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    
    st.download_button(
        label="📥 Download Converted NIfTI File",
        data=bytes_data,
        file_name=filename,
        mime="application/octet-stream"
    )

# Streamlit UI
st.title("🧠 HYGO AI - Advanced DICOM/MRI Clinical Analyzer & Report Generator")
st.markdown("*Powered by HYGO AI's cutting-edge medical imaging technology*")
st.markdown("**Now supports DICOM folder uploads with automatic NIfTI conversion!**")

upload_option = st.radio(
    "Choose upload method:",
    ["Upload DICOM Files (Multiple)", "Upload Single NIfTI File (.nii)"]
)

if upload_option == "Upload DICOM Files (Multiple)":
    st.markdown("### 📁 Upload DICOM Files")
    st.info("Upload multiple DICOM files from a patient folder.")
    
    uploaded_files = st.file_uploader(
        "Select all DICOM files from patient folder", 
        type=['dcm', 'dicom', 'ima'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"📋 Uploaded {len(uploaded_files)} files.")
        
        try:
            with st.spinner("🔍 Analyzing DICOM files..."):
                dicom_series, temp_dir = extract_and_organize_dicoms(uploaded_files)
            
            if not dicom_series:
                st.error("❌ No valid DICOM series found.")
            else:
                st.success(f"✅ Found {len(dicom_series)} DICOM series")
                
                series_info = {}
                for series_key, series_data in dicom_series.items():
                    st.write(f"**{series_key}** - {len(series_data['files'])} files")
                    series_info[series_key] = series_data
                
                selected_series = st.selectbox("🎯 Select series to analyze:", list(series_info.keys()))
                
                if st.button("🚀 Running Sequence of Steps...."):
                    if selected_series:
                        with st.spinner("🔄 Converting..."):
                            nifti_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.nii')
                            nifti_path = nifti_temp_file.name
                            nifti_temp_file.close()
                            
                            conversion_success = convert_dicom_to_nifti(series_info[selected_series]['files'], nifti_path)
                            
                            if conversion_success and os.path.exists(nifti_path):
                                st.success("✅ DICOM to NIfTI conversion successful!")
                                
                                series_desc = series_info[selected_series]['description'].replace(' ', '_')
                                create_download_link(nifti_path, f"converted_{series_desc}.nii")
                                
                                st.write("🔍 Performing segmentation...")
                                model = load_model()
                                segmentation_result, voxel_dims, affine = predict_segmentation(model, nifti_path)
                                
                                if isinstance(segmentation_result, str):
                                    st.error(segmentation_result)
                                else:
                                    findings = summarize_findings(segmentation_result, voxel_dims, affine, selected_series)
                                    st.write("**Detailed Findings:**", findings)
                                    
                                    st.write("✍️ Generating Clinical Report...")
                                    prompt = f"""
                                    You are an AI medical assistant specializing in advanced medical imaging and clinical analysis.

                                    MRI segmentation and statistical analysis results from HYGO AI's proprietary algorithms:
                                    {findings}

                                    Generate a structured clinical report explicitly including precise anatomical details, comprehensive morphological characteristics (borders, shape, necrosis, edema), statistical significance interpretation, detailed differential diagnoses (glioblastoma, meningioma, metastasis).

                                    Make the report as crisp and to the point as possible, using medical terminology and ensuring clarity for healthcare professionals.
                                    Ensure the report is concise, informative, and suitable for clinical use.   
                                    Note that this is just a report of the MRI analysis, and nothing else. This is to be passed along with history and other data to another LLM for diagnosis.
                                    """
                                    
                                    try:
                                        response = client.chat.completions.create(
                                            model="gpt-3.5-turbo",
                                            messages=[{"role": "user", "content": prompt}],
                                            temperature=0.3,
                                            max_tokens=4000
                                        )
                                        
                                        report = response.choices[0].message.content
                                        st.subheader("📜 Clinical Report")
                                        st.write(report)
                                        
                                    except Exception as e:
                                        st.error(f"Error generating report: {e}")
                                
                                try:
                                    os.unlink(nifti_path)
                                except:
                                    pass
                            else:
                                st.error("❌ Conversion failed.")
                
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                    
        except Exception as e:
            st.error(f"Error processing DICOM files: {e}")

else:
    st.markdown("### 📄 Upload Single NIfTI File")
    uploaded_file = st.file_uploader("Upload MRI/CT Scan (.nii)", type=['nii'])
    
    if uploaded_file:
        with open("uploaded_scan.nii", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("🔍 Performing segmentation...")
        model = load_model()
        segmentation_result, voxel_dims, affine = predict_segmentation(model, "uploaded_scan.nii")
        
        if isinstance(segmentation_result, str):
            st.error(segmentation_result)
        else:
            findings = summarize_findings(segmentation_result, voxel_dims, affine, "Single NIfTI Upload")
            st.write("**Detailed Findings:**", findings)
            
            st.write("✍️ Generating Clinical Report...")
            prompt = f"""
            You are an AI medical assistant specializing in advanced medical imaging and clinical analysis.

            MRI segmentation and statistical analysis results from HYGO AI's proprietary algorithms:
            {findings}

            Generate a structured clinical report explicitly including precise anatomical details, comprehensive morphological characteristics (borders, shape, necrosis, edema), statistical significance interpretation, detailed differential diagnoses (glioblastoma, meningioma, metastasis).

            Make the report as crisp and to the point as possible, using medical terminology and ensuring clarity for healthcare professionals.
            Ensure the report is concise, informative, and suitable for clinical use.   
            Note that this is just a report of the MRI analysis, and nothing else. This is to be passed along with history and other data to another LLM for diagnosis.
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=4000
                )
                
                report = response.choices[0].message.content
                st.subheader("📜 Clinical Report")
                st.write(report)
                
            except Exception as e:
                st.error(f"Error generating report: {e}")
        
        try:
            os.remove("uploaded_scan.nii")
        except:
            pass

st.markdown("---")
st.markdown("**🏥 Report generated by HYGO AI Medical Imaging Platform**")
st.markdown("*HYGO AI - Revolutionizing Medical Diagnostics through Advanced AI Technology*")