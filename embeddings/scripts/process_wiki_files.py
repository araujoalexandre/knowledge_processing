import os
import json
import argparse

def process_wiki_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all subfolders in the input folder
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Create a corresponding output file for each subfolder
            output_file_path = os.path.join(output_folder, f"{subfolder}.txt")
            
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                # Process all wiki files in the current subfolder
                for wiki_file in os.listdir(subfolder_path):
                    wiki_file_path = os.path.join(subfolder_path, wiki_file)
                    
                    with open(wiki_file_path, 'r', encoding='utf-8') as input_file:
                        for line in input_file.readlines():
                            try:
                                json_data = json.loads(line)
                                # Check if the 'text' attribute is not empty
                                if json_data.get('text'):
                                    output_file.write(line)
                            except json.JSONDecodeError:
                                print(f"Error decoding JSON in file: {wiki_file_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process wiki files.")
    parser.add_argument('--input', required=True, help="Path to the input 'wikipedia' folder")
    parser.add_argument('--output', required=True, help="Path to the output folder for processed files")

    # Parse arguments
    args = parser.parse_args()

    # Call the process_wiki_files function with the provided arguments
    process_wiki_files(args.input, args.output)
    print(f"Processing complete.")

if __name__ == "__main__":
    main()
