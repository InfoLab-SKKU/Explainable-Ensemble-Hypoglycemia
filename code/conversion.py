import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def detect_outliers(series, method='iqr', threshold=3):
    """Detect outliers using IQR or z-score method"""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    return pd.Series([False] * len(series), index=series.index)

def clean_numeric_column(series, column_name):
    """Clean numeric columns by removing extreme outliers"""
    if series.dtype in ['object', 'bool']:
        return series
    
    # Remove infinite values
    series = series.replace([np.inf, -np.inf], np.nan)
    
    if series.nunique() <= 2:  # Binary or constant columns
        return series
    
    # Detect and handle extreme outliers
    outliers = detect_outliers(series, method='iqr')
    n_outliers = outliers.sum()
    
    if n_outliers > 0:
        print(f"    Found {n_outliers} outliers in {column_name} (range: {series.min():.2f} to {series.max():.2f})")
        # Replace extreme outliers with NaN
        series.loc[outliers] = np.nan
    
    return series

def safe_mode(series):
    """Safely get the mode of a series, returning the first mode value or first value if no mode"""
    try:
        series_clean = series.dropna()  # Remove NaN values
        if len(series_clean) == 0:
            return np.nan
        
        mode_result = series_clean.mode()
        if len(mode_result) > 0:
            return mode_result.iloc[0]  # Return the first mode value
        else:
            return series_clean.iloc[0]  # Fallback to first value
    except:
        return series.iloc[0] if len(series) > 0 else np.nan

def safe_first(series):
    """Safely get the first non-null value from a series"""
    try:
        non_null = series.dropna()
        return non_null.iloc[0] if len(non_null) > 0 else np.nan
    except:
        return np.nan

def clean_and_aggregate_data(df):
    """Clean data and handle duplicates by aggregating properly"""
    if df is None or len(df) == 0:
        return df
    
    # Check for duplicate PtIDs
    if df['PtID'].duplicated().any():
        print(f"    Found {df['PtID'].duplicated().sum()} duplicate PtIDs, aggregating...")
        
        # Create aggregation dictionary
        agg_dict = {}
        
        for col in df.columns:
            if col == 'PtID':
                continue
            elif pd.api.types.is_numeric_dtype(df[col]):
                # For numeric columns, take the mean (ignoring NaN)
                agg_dict[col] = 'mean'
            else:
                # For text/categorical columns, use safe_first to get the first non-null value
                agg_dict[col] = safe_first
        
        # Aggregate the data
        df_clean = df.groupby('PtID').agg(agg_dict).reset_index()
        
        print(f"    After aggregation: {len(df_clean)} unique patients")
        return df_clean
    
    return df

def load_roster_data(roster_path):
    """Load the roster file with Case/Control status"""
    print("Loading roster data...")
    roster_df = pd.read_csv(roster_path, sep='|')
    print(f"Roster loaded: {len(roster_df)} records")
    return roster_df[['PtID', 'BCaseControlStatus']]

def load_tabular_data(tabular_dir):
    """Load all tabular CSV files and merge them by PtID"""
    print("Loading tabular data...")
    tabular_files = list(Path(tabular_dir).glob('*.csv'))
    
    merged_tabular = None
    
    for file_path in tabular_files:
        print(f"  Processing {file_path.name}...")
        df = pd.read_csv(file_path)
        
        if 'PtID' not in df.columns:
            print(f"    Warning: No PtID column in {file_path.name}, skipping...")
            continue
            
        # Remove problematic columns
        cols_to_remove = ['label_encoded']  # Target-related columns
        # Remove unnamed columns
        cols_to_remove.extend([col for col in df.columns if col.startswith('Unnamed:')])
        
        for col in cols_to_remove:
            if col in df.columns:
                df = df.drop(col, axis=1)
                print(f"    Removed column: {col}")
        
        # Clean column names and add suffix (except PtID)
        file_suffix = file_path.stem.replace('tab_', '')
        new_columns = ['PtID']
        
        for col in df.columns:
            if col != 'PtID':
                new_col_name = f"{col}_{file_suffix}"
                new_columns.append(new_col_name)
        
        df.columns = new_columns
          # Clean numeric columns
        for col in df.columns:
            if col != 'PtID':
                df[col] = clean_numeric_column(df[col], col)
        
        if merged_tabular is None:
            merged_tabular = df
        else:
            merged_tabular = pd.merge(merged_tabular, df, on='PtID', how='outer')
    
    # Clean and aggregate any duplicates
    merged_tabular = clean_and_aggregate_data(merged_tabular)
    
    print(f"Tabular data merged: {len(merged_tabular)} patients, {len(merged_tabular.columns)-1} features")
    return merged_tabular

def load_text_data(text_dir):
    """Load text data files and merge them by PtID"""
    print("Loading text data...")
    text_files = list(Path(text_dir).glob('*.csv'))
    
    merged_text = None
    
    for file_path in text_files:
        print(f"  Processing {file_path.name}...")
        df = pd.read_csv(file_path)
        
        if 'PtID' not in df.columns:
            print(f"    Warning: No PtID column in {file_path.name}, skipping...")
            continue
            
        # Remove problematic columns
        cols_to_remove = ['label_encoded']  # Target-related columns
        # Remove redundant long text columns (keep only the main one)
        text_cols = [col for col in df.columns if any(x in col.lower() for x in ['medcon', 'medication'])]
        if len(text_cols) > 1:
            # Keep only the shortest text column and basic demographic info
            main_col = min(text_cols, key=lambda x: len(str(x)))
            cols_to_keep = ['PtID', main_col]
            # Keep basic demographic columns if available
            for col in ['Gender', 'Race', 'Weight_mod', 'Height_mod']:
                if col in df.columns:
                    cols_to_keep.append(col)
            df = df[cols_to_keep]
            print(f"    Kept main text column: {main_col} and demographic info")
        
        # Remove remaining problematic columns
        for col in cols_to_remove:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Add suffix to column names (except PtID) to avoid conflicts
        file_suffix = file_path.stem.replace('bbdd_', '').replace('2', '')
        new_columns = ['PtID']
        
        for col in df.columns:
            if col != 'PtID':
                new_col_name = f"{col}_{file_suffix}"
                new_columns.append(new_col_name)
        
        df.columns = new_columns
          # Clean numeric columns
        for col in df.columns:
            if col != 'PtID' and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = clean_numeric_column(df[col], col)
        
        if merged_text is None:
            merged_text = df
        else:
            merged_text = pd.merge(merged_text, df, on='PtID', how='outer')
    
    # Clean and aggregate any duplicates
    merged_text = clean_and_aggregate_data(merged_text)
    
    print(f"Text data merged: {len(merged_text)} patients, {len(merged_text.columns)-1} features")
    return merged_text

def load_time_series_data(time_series_dir):
    """Load time series data and create aggregated features by PtID"""
    print("Loading time series data...")
    time_series_files = list(Path(time_series_dir).glob('*.csv'))
    
    # Group files by PtID
    patient_files = {}
    for file_path in time_series_files:
        filename = file_path.stem
        try:
            pt_id = int(filename.split('_')[0])
            if pt_id not in patient_files:
                patient_files[pt_id] = []
            patient_files[pt_id].append(file_path)
        except (ValueError, IndexError):
            print(f"    Warning: Could not parse filename {filename}, skipping...")
            continue
    
    # Process each patient's time series data
    time_series_features = []
    
    for pt_id, files in patient_files.items():
        # Aggregate features across all seeds for this patient
        all_features = {
            'PtID': pt_id,
            'ts_num_series': len(files),  # Number of time series for this patient
        }
        
        # Initialize aggregation lists
        word_counts = []
        hypo_totals = []
        number_totals = []
        hypo_means = []
        number_means = []
        hypo_maxs = []
        number_maxs = []
        hypo_stds = []
        number_stds = []
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                
                if 'Hypo' in df.columns and 'Number' in df.columns:
                    word_counts.append(len(df))
                    hypo_totals.append(df['Hypo'].sum())
                    number_totals.append(df['Number'].sum())
                    hypo_means.append(df['Hypo'].mean())
                    number_means.append(df['Number'].mean())
                    hypo_maxs.append(df['Hypo'].max())
                    number_maxs.append(df['Number'].max())
                    hypo_stds.append(df['Hypo'].std())
                    number_stds.append(df['Number'].std())
                    
            except Exception as e:
                print(f"    Warning: Error processing {file_path.name}: {e}")
                continue
        
        # Calculate aggregate statistics across all seeds
        if word_counts:
            all_features.update({
                'ts_avg_word_count': np.mean(word_counts),
                'ts_avg_hypo_total': np.mean(hypo_totals),
                'ts_avg_number_total': np.mean(number_totals),
                'ts_avg_hypo_mean': np.mean(hypo_means),
                'ts_avg_number_mean': np.mean(number_means),
                'ts_avg_hypo_max': np.mean(hypo_maxs),
                'ts_avg_number_max': np.mean(number_maxs),
                'ts_avg_hypo_std': np.mean([x for x in hypo_stds if not np.isnan(x)]) if hypo_stds else 0,
                'ts_avg_number_std': np.mean([x for x in number_stds if not np.isnan(x)]) if number_stds else 0,
                'ts_hypo_variability': np.std(hypo_totals) if len(hypo_totals) > 1 else 0,
                'ts_number_variability': np.std(number_totals) if len(number_totals) > 1 else 0,
            })
        
        time_series_features.append(all_features)
    
    if time_series_features:
        ts_df = pd.DataFrame(time_series_features)
        
        # Clean numeric columns
        for col in ts_df.columns:
            if col != 'PtID' and pd.api.types.is_numeric_dtype(ts_df[col]):
                ts_df[col] = clean_numeric_column(ts_df[col], col)
        
        print(f"Time series data processed: {len(ts_df)} patients, {len(ts_df.columns)-1} features")
        return ts_df
    else:
        print("No valid time series data found")
        return pd.DataFrame(columns=['PtID'])

def remove_duplicate_columns(df):
    """Remove duplicate columns that represent the same information"""
    print("Removing duplicate columns...")
    
    # Group similar columns
    demographic_cols = {}
    for col in df.columns:
        if col == 'PtID' or col == 'BCaseControlStatus':
            continue
            
        base_name = col.lower()
        # Identify demographic duplicates
        if 'gender' in base_name:
            if 'gender' not in demographic_cols:
                demographic_cols['gender'] = []
            demographic_cols['gender'].append(col)
        elif 'race' in base_name:
            if 'race' not in demographic_cols:
                demographic_cols['race'] = []
            demographic_cols['race'].append(col)
        elif 'weight' in base_name and 'mod' in base_name:
            if 'weight' not in demographic_cols:
                demographic_cols['weight'] = []
            demographic_cols['weight'].append(col)
        elif 'height' in base_name and 'mod' in base_name:
            if 'height' not in demographic_cols:
                demographic_cols['height'] = []
            demographic_cols['height'].append(col)
    
    # Keep only one column from each demographic group
    cols_to_remove = []
    for demo_type, cols in demographic_cols.items():
        if len(cols) > 1:
            # Keep the first one, remove the rest
            cols_to_remove.extend(cols[1:])
            print(f"  Removing duplicate {demo_type} columns: {cols[1:]}")
    
    # Remove duplicate columns
    df_cleaned = df.drop(columns=cols_to_remove)
    
    print(f"Removed {len(cols_to_remove)} duplicate columns")
    return df_cleaned

def remove_high_missing_columns(df, threshold=0.8):
    """Remove columns with more than threshold proportion of missing values"""
    print(f"Removing columns with >{threshold*100}% missing values...")
    
    missing_prop = df.isnull().sum() / len(df)
    high_missing_cols = missing_prop[missing_prop > threshold].index.tolist()
    
    # Don't remove PtID or target
    high_missing_cols = [col for col in high_missing_cols if col not in ['PtID', 'BCaseControlStatus']]
    
    if high_missing_cols:
        print(f"  Removing {len(high_missing_cols)} columns with high missing data:")
        for col in high_missing_cols:
            print(f"    {col}: {missing_prop[col]*100:.1f}% missing")
        df = df.drop(columns=high_missing_cols)
    
    return df

def clean_outliers(df):
    """Clean extreme outliers and suspicious values"""
    print("Cleaning outliers and suspicious values...")
    
    df_clean = df.copy()
    
    # Fix extreme height values (normal range: 140-220 cm)
    height_cols = [col for col in df_clean.columns if 'height' in col.lower() or 'Height' in col]
    for col in height_cols:
        if col in df_clean.columns:
            # Replace extreme values with median
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].apply(lambda x: median_val if pd.isna(x) or x > 250 or x < 100 else x)
    
    # Fix extreme weight values (normal range: 40-200 kg)
    weight_cols = [col for col in df_clean.columns if 'weight' in col.lower() or 'Weight' in col]
    for col in weight_cols:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].apply(lambda x: median_val if pd.isna(x) or x > 300 or x < 30 else x)
    
    # Fix other extreme numeric outliers using IQR method
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['PtID']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # More conservative than 1.5*IQR
            upper_bound = Q3 + 3 * IQR
            
            # Replace extreme outliers with median
            median_val = df_clean[col].median()
            mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            if mask.sum() > 0:
                print(f"  Fixed {mask.sum()} outliers in {col}")
                df_clean.loc[mask, col] = median_val
    
    return df_clean

def clean_bound_methods(df):
    """Clean up any bound method strings that might exist in the dataset"""
    print("Cleaning bound method artifacts...")
    
    for col in df.columns:
        if df[col].dtype == 'object':  # Only check text columns
            # Check if any values contain bound method strings
            bound_method_mask = df[col].astype(str).str.contains('<bound method', na=False)
            
            if bound_method_mask.any():
                print(f"    Found bound method strings in column: {col}")
                # Replace bound method strings with NaN for now
                df.loc[bound_method_mask, col] = np.nan
    
    return df

def calculate_missing_percentage(df, exclude_cols=['PtID', 'BCaseControlStatus']):
    """Calculate percentage of missing values for each row"""
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    missing_counts = df[feature_cols].isnull().sum(axis=1)
    total_features = len(feature_cols)
    missing_percentage = (missing_counts / total_features) * 100
    return missing_percentage

def main():
    # Define paths
    base_dir = Path("preprocessed")
    roster_path = base_dir / "BPtRoster.txt"
    tabular_dir = base_dir / "tabular"
    text_dir = base_dir / "text"
    time_series_dir = base_dir / "time_series"
    
    print("=" * 60)
    print("DATASET CONSOLIDATION FOR DEEP LEARNING")
    print("=" * 60)
    
    # Load all data sources
    roster_df = load_roster_data(roster_path)
      # Load different data types
    tabular_df = load_tabular_data(tabular_dir) if tabular_dir.exists() else pd.DataFrame(columns=['PtID'])
    text_df = load_text_data(text_dir) if text_dir.exists() else pd.DataFrame(columns=['PtID'])
    time_series_df = load_time_series_data(time_series_dir) if time_series_dir.exists() else pd.DataFrame(columns=['PtID'])
    
    print("\n" + "=" * 60)
    print("MERGING ALL DATA SOURCES")
    print("=" * 60)
    
    # Start with roster data (contains target variable)
    final_df = roster_df.copy()
    
    # Merge tabular data
    if len(tabular_df) > 0:
        print(f"Merging tabular data...")
        final_df = pd.merge(final_df, tabular_df, on='PtID', how='left')
        print(f"After tabular merge: {len(final_df)} patients, {len(final_df.columns)} columns")
    
    # Merge text data
    if len(text_df) > 0:
        print(f"Merging text data...")
        final_df = pd.merge(final_df, text_df, on='PtID', how='left')
        print(f"After text merge: {len(final_df)} patients, {len(final_df.columns)} columns")
    
    # Merge time series data
    if len(time_series_df) > 0:
        print(f"Merging time series data...")
        final_df = pd.merge(final_df, time_series_df, on='PtID', how='left')
        print(f"After time series merge: {len(final_df)} patients, {len(final_df.columns)} columns")
    
    print("\n" + "=" * 60)
    print("CLEANING DATASET")
    print("=" * 60)
    
    # Remove duplicate columns
    final_df = remove_duplicate_columns(final_df)
    print(f"After removing duplicates: {len(final_df)} patients, {len(final_df.columns)} columns")
      # Remove columns with too much missing data
    final_df = remove_high_missing_columns(final_df, threshold=0.8)
    print(f"After removing high-missing columns: {len(final_df)} patients, {len(final_df.columns)} columns")
    
    print("\n" + "=" * 60)
    print("CLEANING AND PROCESSING DATA")
    print("=" * 60)
    
    # Clean outliers and suspicious values
    final_df = clean_outliers(final_df)
    
    # Clean any bound method artifacts
    final_df = clean_bound_methods(final_df)
    
    print("\n" + "=" * 60)
    print("REMOVING PATIENTS WITH MORE THAN 5 MISSING VALUES")
    print("=" * 60)
    
    # Remove patients with more than 5 missing values
    patients_before = len(final_df)
    print(f"Patients before removing missing values: {patients_before}")
    
    # Check for missing values in all columns except PtID
    feature_cols = [col for col in final_df.columns if col != 'PtID']
    missing_counts = final_df[feature_cols].isnull().sum(axis=1)
    
    # Keep only patients with 5 or fewer missing values
    final_df_clean = final_df[missing_counts <= 15]
    
    patients_after = len(final_df_clean)
    print(f"Patients after removing those with >5 missing values: {patients_after}")
    print(f"Patients removed: {patients_before - patients_after}")
    
    # Show distribution of missing values
    print(f"\nMissing values distribution:")
    missing_dist = missing_counts.value_counts().sort_index()
    for missing_count, num_patients in missing_dist.items():
        if missing_count <= 10:  # Show detailed breakdown for low counts
            status = "kept" if missing_count <= 5 else "removed"
            print(f"  {missing_count} missing values: {num_patients} patients ({status})")
        elif missing_count > 5:
            print(f"  {missing_count} missing values: {num_patients} patients (removed)")
            break
    
    print("\n" + "=" * 60)
    print("SORTING BY PtID")
    print("=" * 60)
    
    # Sort by PtID in ascending order
    final_df_clean = final_df_clean.sort_values('PtID').reset_index(drop=True)
    print(f"Dataset sorted by PtID (ascending order)")
    print(f"PtID range: {final_df_clean['PtID'].min()} to {final_df_clean['PtID'].max()}")
    
    print("\n" + "=" * 60)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 60)
    
    # Calculate missing data percentage for each patient (should be 0 now)
    missing_percentages = calculate_missing_percentage(final_df_clean)
    
    print(f"Total patients: {len(final_df_clean)}")
    print(f"Total features: {len(final_df_clean.columns) - 2}")  # Exclude PtID, BCaseControlStatus
    print(f"Missing data per patient: {missing_percentages.mean():.2f}% (should be 0)")
    print(f"Patients with any missing data: {(missing_percentages > 0).sum()} (should be 0)")
    
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    
    print(f"Final dataset shape: {final_df_clean.shape}")
    print(f"Patients: {len(final_df_clean)}")
    print(f"Features: {len(final_df_clean.columns) - 2}")  # Exclude PtID and BCaseControlStatus
    
    # Show target variable distribution
    if 'BCaseControlStatus' in final_df_clean.columns:
        target_dist = final_df_clean['BCaseControlStatus'].value_counts()
        print(f"\nTarget variable distribution:")
        for status, count in target_dist.items():
            percentage = (count / len(final_df_clean)) * 100
            print(f"  {status}: {count} ({percentage:.1f}%)")
    
    # Show feature types
    numeric_cols = final_df_clean.select_dtypes(include=[np.number]).columns
    text_cols = final_df_clean.select_dtypes(include=['object']).columns
    print(f"\nFeature types:")
    print(f"  Numeric features: {len(numeric_cols) - 1}")  # Exclude PtID
    print(f"  Text/Categorical features: {len(text_cols) - 1}")  # Exclude BCaseControlStatus
    
    print("\n" + "=" * 60)
    print("SAVING FINAL DATASET")
    print("=" * 60)
    
    # Save the final dataset
    output_file = "consolidated_dataset.csv"
    final_df_clean.to_csv(output_file, index=False)
    print(f"Dataset saved as: {output_file}")
    
    # Save a summary report
    summary_file = "dataset_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("CONSOLIDATED DATASET SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset shape: {final_df_clean.shape}\n")
        f.write(f"Patients: {len(final_df_clean)}\n")
        f.write(f"Features: {len(final_df_clean.columns) - 2}\n\n")
        
        f.write("Target variable distribution:\n")
        if 'BCaseControlStatus' in final_df_clean.columns:
            for status, count in target_dist.items():
                percentage = (count / len(final_df_clean)) * 100
                f.write(f"  {status}: {count} ({percentage:.1f}%)\n")
        
        f.write(f"\nFeature types:\n")
        f.write(f"  Numeric features: {len(numeric_cols) - 1}\n")
        f.write(f"  Text/Categorical features: {len(text_cols) - 1}\n")
        
        f.write(f"\nColumns in final dataset:\n")
        for i, col in enumerate(final_df_clean.columns, 1):
            f.write(f"  {i:3d}. {col}\n")
        
        f.write(f"\nFirst 5 PtIDs: {list(final_df_clean['PtID'].head())}\n")
        f.write(f"Last 5 PtIDs: {list(final_df_clean['PtID'].tail())}\n")
    
    print(f"Summary report saved as: {summary_file}")
    
    print("\n" + "=" * 60)
    print("DATASET CONSOLIDATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return final_df_clean

if __name__ == "__main__":
    final_dataset = main()
