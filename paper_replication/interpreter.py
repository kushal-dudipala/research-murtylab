import os

# Mapping of categories to brain regions
brain_region_mapping = {
    "face": "FFA (Fusiform Face Area)",
    "building": "PPA (Parahippocampal Place Area)",
    "animal": "EBA (Extrastriate Body Area)",
    "body part": "EBA (Extrastriate Body Area)",
    "foot": "EBA (Extrastriate Body Area)",
    "hand": "EBA (Extrastriate Body Area)",
    "arm": "EBA (Extrastriate Body Area)",
    "leg": "EBA (Extrastriate Body Area)",
    "head": "EBA (Extrastriate Body Area)",
    "car": "LOC (Lateral Occipital Complex)",
    "landscape": "RSC (Retrosplenial Cortex)",
    "torso": "EBA (Extrastriate Body Area)",
}

# Function to read the classification results from a file
def read_classification_results(file_path):
    classifications = {}
    with open(file_path, 'r') as f:
        for line in f:
            image_name, category = line.strip().split(': ')
            if category not in classifications:
                classifications[category] = []
            classifications[category].append(image_name)
    return classifications

# Function to assign brain regions to each category and print the results
def assign_brain_regions(classifications):
    sorted_output = {}
    for category, images in classifications.items():
        brain_region = brain_region_mapping.get(category, "Unknown Region")
        if brain_region not in sorted_output:
            sorted_output[brain_region] = []
        sorted_output[brain_region].extend(images)
    return sorted_output

# Function to print and save the sorted output to a file
def print_and_save_sorted_output(sorted_output, output_file_path):
    with open(output_file_path, 'w') as f:
        for brain_region, images in sorted_output.items():
            images_list = ', '.join(images)
            output_str = f"{brain_region}: {images_list}\n"
            print(output_str)
            f.write(output_str)

# Main function
def main():
    # Correct the relative path to the classification results file
    input_file_path = "research-murtylab/paper_replication/clip_classification_results.txt"
    output_file_path = os.path.join(os.path.dirname(__file__), "sorted_brain_regions.txt")

    # Step 1: Read classification results
    classifications = read_classification_results(input_file_path)

    # Step 2: Assign brain regions
    sorted_output = assign_brain_regions(classifications)

    # Step 3: Print and save the sorted output
    print_and_save_sorted_output(sorted_output, output_file_path)

    print("Brain region assignment completed and saved.")

if __name__ == "__main__":
    main()
