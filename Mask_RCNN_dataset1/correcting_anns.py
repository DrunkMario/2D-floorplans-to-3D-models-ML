import json

# Load the existing annotations file
annotation_file = 'floor planner/Mask_RCNN_dataset1/coco_json_dir/annotations.json'

with open(annotation_file, 'r') as f:
    data = json.load(f)

# Iterate through annotations and add the polygon coordinates
for annotation in data['annotations']:
    x_min, y_min, width, height = annotation['bbox']
    
    # Calculate the polygon coordinates (4 points)
    polygon = [
        x_min, y_min,                 # Top-left
        x_min + width, y_min,         # Top-right
        x_min + width, y_min + height,# Bottom-right
        x_min, y_min + height         # Bottom-left
    ]
    
    # Add the segmentation field to the annotation
    annotation['segmentation'] = [polygon]

# Save the updated annotations file
with open(annotation_file, 'w') as f:
    json.dump(data, f, indent=4)

print("Annotations updated with polygon coordinates.")
