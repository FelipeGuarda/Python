import pandas as pd # type: ignore
import glob
import os

folder_path = r"C:\Dev\Python\Estacion metereologica\Linea de tiempo"
dat_files = glob.glob(os.path.join(folder_path, "*.dat"))

df_list = []

for file in dat_files:
    try:
        # Detect the header line (the one containing "TIMESTAMP")
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        header_line_index = None
        for i, line in enumerate(lines):
            if '"TIMESTAMP"' in line or 'TIMESTAMP' in line:
                header_line_index = i
                break

        if header_line_index is None:
            print(f"⚠️ No TIMESTAMP header found in {os.path.basename(file)}")
            continue

        # Read from the detected header line
        df = pd.read_csv(
            file,
            skiprows=header_line_index,
            quotechar='"',
            sep=','
        )

        # Clean column names
        df.columns = df.columns.str.replace('"', '').str.strip()

        # Parse timestamps
        if 'TIMESTAMP' not in df.columns:
            print(f"⚠️ After parsing, still no TIMESTAMP in {os.path.basename(file)} — columns: {df.columns.tolist()}")
            continue

        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
        df['source_file'] = os.path.basename(file)
        df_list.append(df)
        print(f"✅ Loaded: {os.path.basename(file)} ({len(df)} rows)")

    except Exception as e:
        print(f"⚠️ Error reading {os.path.basename(file)}: {e}")

# Combine and sort all
if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df.sort_values(by='TIMESTAMP').drop_duplicates(subset='TIMESTAMP')

    output_file = os.path.join(folder_path, "merged_timeline.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"\n✅ Merged file created successfully: {output_file}")
else:
    print("\n❌ No valid data files found.")
