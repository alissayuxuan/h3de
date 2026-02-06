import pandas as pd
import json

def add_splits_to_master_df(csv_path, splits_json_path=None, output_path=None):
    """
    Fügt eine 'split' Spalte zum master_df hinzu basierend auf Subject-Zuordnungen.
    
    Args:
        csv_path: Pfad zur master_df.csv
        splits_json_path: Pfad zur JSON-Datei mit train/val/test splits (optional)
        output_path: Pfad zum Speichern der neuen CSV (optional, sonst überschreibt input)
    
    Returns:
        pd.DataFrame: DataFrame mit neuer 'split' Spalte
    """
    
    # Lade master_df
    df = pd.read_csv(csv_path)
    
    # Splits (entweder aus JSON laden oder hardcoded)
    if splits_json_path:
        with open(splits_json_path, 'r') as f:
            splits_data = json.load(f)
        train_subjects = splits_data['train_subjects']
        val_subjects = splits_data['val_subjects']
        test_subjects = splits_data['test_subjects']
    else:
        # Hardcoded splits
        train_subjects = [
            "WS-17", "WS-18", "WS-53", "WS-09", "WS-08", "WS-22", "WS-63", 
            "WS-52", "WS-36", "WS-47", "WS-34", "WS-23", "WS-13", "WS-38", 
            "WS-43", "WS-07", "WS-56", "WS-31", "WS-05", "WS-30", "WS-50", 
            "WS-19", "WS-40", "WS-29", "WS-16", "WS-00"
        ]
        val_subjects = ["WS-62", "WS-45", "WS-26"]
        test_subjects = ["WS-06", "WS-55", "WS-15", "WS-46", "WS-54", "WS-25", "WS-48"]
    
    # Erstelle Mapping Subject -> Split
    subject_to_split = {}
    for subj in train_subjects:
        subject_to_split[subj] = 'train'
    for subj in val_subjects:
        subject_to_split[subj] = 'val'
    for subj in test_subjects:
        subject_to_split[subj] = 'test'
    
    # Füge 'split' Spalte hinzu
    df['split'] = df['subject'].map(subject_to_split)
    
    # Check für unmapped subjects
    unmapped = df[df['split'].isna()]['subject'].unique()
    if len(unmapped) > 0:
        print(f"⚠️  Warning: {len(unmapped)} subjects ohne split gefunden:")
        print(f"   {unmapped.tolist()}")
    
    # Statistiken ausgeben
    print("\n✓ Split statistics:")
    print(df['split'].value_counts().sort_index())
    print(f"\nTotal samples: {len(df)}")
    
    # Speichern
    if output_path is None:
        output_path = csv_path
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    master_df = "datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv"

    add_splits_to_master_df(csv_path=master_df)