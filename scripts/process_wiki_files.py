import os
import json
import argparse
import tarfile
import natsort

def process_wiki_files(input_dir, output_dir):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all subfolders in the input folder
    subfolders = natsort.natsorted(os.listdir(input_dir))
    n_subfolders = len(subfolders)

    for subfolder_id, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(input_dir, subfolder)
        print(subfolder_path)

        if os.path.isdir(subfolder_path):
            # Create a corresponding output tar file for each subfolder
            output_file_path = os.path.join(output_dir, f"data-{subfolder_id+1:06d}-{n_subfolders:06d}.tar")
            with tarfile.open(output_file_path, "w") as tar:

                # Process all wiki files in the current subfolder
                article_id = 0
                for wiki_file in os.listdir(subfolder_path):
                    wiki_file_path = os.path.join(subfolder_path, wiki_file)
                    
                    with open(wiki_file_path, 'r', encoding='utf-8') as input_file:
                        for line in input_file.readlines():
                            try:
                                json_data = json.loads(line)
                                # Check if the 'text' attribute is not empty
                                if not json_data.get('text', False):
                                    continue

                                # Create a temporary file for each JSON line
                                temp_file = os.path.join(output_dir, f"{subfolder}_{article_id}.json")
                                with open(temp_file, 'w') as tf:
                                    tf.write(line)

                                # Add the temporary file to the tar archive
                                tar.add(temp_file, arcname=f"{subfolder}_{article_id}.txt")

                                # Remove the temporary file
                                os.remove(temp_file)

                                article_id += 1

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
