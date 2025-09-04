import pandas as pd

# Load the CSV file
df = pd.read_csv('speakingPredictionAVA.csv')

# Check for mismatches where Binary_Avg_Col5 == 1 and Max_Col4 == 0
df['Mismatch_Type1'] = (df['Binary_Avg_Col5'] == 1) & (df['Max_Col4'] == 0)

# Check for mismatches where Binary_Avg_Col5 == 0 and Max_Col4 == 1
df['Mismatch_Type2'] = (df['Binary_Avg_Col5'] == 0) & (df['Max_Col4'] == 1)

def extract_mismatches(df, mismatch_column):
    mismatch_groups = []
    current_mismatch_group = []

    for index, row in df.iterrows():
        if row[mismatch_column]:
            current_mismatch_group.append(row)
        else:
            if len(current_mismatch_group) >= 3:
                mismatch_groups.extend(current_mismatch_group)
            current_mismatch_group = []

    # Add the last group if it meets the length criteria
    if len(current_mismatch_group) >= 3:
        mismatch_groups.extend(current_mismatch_group)

    return pd.DataFrame(mismatch_groups).drop(columns=[mismatch_column])

# Extract mismatches for both types
mismatch_df_type1 = extract_mismatches(df, 'Mismatch_Type1')
mismatch_df_type2 = extract_mismatches(df, 'Mismatch_Type2')

# Save the mismatches to separate CSV files
mismatch_df_type1.to_csv('mismatch_type1_output.csv', index=False)
mismatch_df_type2.to_csv('mismatch_type2_output.csv', index=False)

print("Mismatches have been saved to 'mismatch_type1_outputAVAval.csv' and 'mismatch_type2_outputAVAval.csv'")
