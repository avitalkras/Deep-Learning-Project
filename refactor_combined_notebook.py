import json
import re

# Read the combined notebook
with open('Part_A_and_B_Combined.ipynb', 'r') as f:
    nb = json.load(f)

# Track which cell is Part A vs Part B
part_a_end = 31  # Part A ends at cell 31 (0-indexed)
part_b_start = 33  # Part B starts at cell 33 (after separator)

print("Refactoring notebook for variable isolation...")

# 1. Part A: Change final_model to trained_classifier
print("\n1. Part A: Renaming final_model to trained_classifier...")
for i in range(part_a_end + 1):
    cell = nb['cells'][i]
    if 'source' in cell:
        source = cell['source']
        if isinstance(source, list):
            source_str = ''.join(source)
        else:
            source_str = source
        
        # Replace final_model with trained_classifier in Part A
        if 'final_model' in source_str:
            new_source = []
            for line in source:
                # Replace final_model with trained_classifier
                line = line.replace('final_model', 'trained_classifier')
                new_source.append(line)
            cell['source'] = new_source
            print(f"   Updated cell {i}")

# 2. Part B: Change final_model references to trained_classifier
print("\n2. Part B: Updating references from final_model to trained_classifier...")
for i in range(part_b_start, len(nb['cells'])):
    cell = nb['cells'][i]
    if 'source' in cell:
        source = cell['source']
        if isinstance(source, list):
            source_str = ''.join(source)
        else:
            source_str = source
        
        # Replace final_model references with trained_classifier
        if 'final_model' in source_str:
            new_source = []
            for line in source:
                # Replace references to final_model
                line = line.replace('final_model', 'trained_classifier')
                line = line.replace('"final_model"', '"trained_classifier"')
                line = line.replace("'final_model'", "'trained_classifier'")
                new_source.append(line)
            cell['source'] = new_source
            print(f"   Updated cell {i}")

# 3. Part B: Rename local 'model' variable in apply_and_evaluate_style_transfer to 'style_loss_network'
print("\n3. Part B: Renaming local 'model' variable in style transfer functions...")
for i in range(part_b_start, len(nb['cells'])):
    cell = nb['cells'][i]
    if 'source' in cell:
        source = cell['source']
        source_str = ''.join(source) if isinstance(source, list) else source
        
        # Find apply_and_evaluate_style_transfer function
        if 'def apply_and_evaluate_style_transfer' in source_str:
            new_source = []
            in_function = False
            for line in source:
                if 'def apply_and_evaluate_style_transfer' in line:
                    in_function = True
                elif in_function and line.strip().startswith('def '):
                    in_function = False
                
                # Inside the function, replace local 'model' variable assignments
                if in_function:
                    # Replace: model = vgg19_features -> style_loss_network = vgg19_features
                    line = re.sub(r'(\s+)model = (vgg19_features|alexnet_features)', r'\1style_loss_network = \2', line)
                    # Replace: model = model_name -> style_loss_network = model_name (but be careful)
                    # Replace uses of 'model' variable (but not in function parameter or other contexts)
                    # Only replace standalone 'model' that's assigned from vgg19_features or alexnet_features
                    if 'style_loss_network' in ''.join(new_source):
                        # After assignment, replace uses of 'model' with 'style_loss_network'
                        line = re.sub(r'\bmodel\b', 'style_loss_network', line)
                
                new_source.append(line)
            cell['source'] = new_source
            print(f"   Updated cell {i} (apply_and_evaluate_style_transfer)")

# Also update neural_style_transfer function parameter name for clarity
print("\n4. Part B: Updating neural_style_transfer function parameter...")
for i in range(part_b_start, len(nb['cells'])):
    cell = nb['cells'][i]
    if 'source' in cell:
        source = cell['source']
        source_str = ''.join(source) if isinstance(source, list) else source
        
        if 'def neural_style_transfer(model,' in source_str:
            new_source = []
            in_function = False
            for line in source:
                if 'def neural_style_transfer(model,' in line:
                    # Change parameter name from 'model' to 'style_loss_network'
                    line = line.replace('def neural_style_transfer(model,', 'def neural_style_transfer(style_loss_network,')
                    in_function = True
                elif in_function and line.strip().startswith('def '):
                    in_function = False
                
                # Inside function, replace 'model' with 'style_loss_network'
                if in_function:
                    line = line.replace('model.eval()', 'style_loss_network.eval()')
                    line = line.replace('model.parameters()', 'style_loss_network.parameters()')
                    line = line.replace('model._modules', 'style_loss_network._modules')
                
                new_source.append(line)
            cell['source'] = new_source
            print(f"   Updated cell {i} (neural_style_transfer)")

# Update all calls to neural_style_transfer to use style_loss_network parameter name
print("\n5. Part B: Updating calls to neural_style_transfer...")
for i in range(part_b_start, len(nb['cells'])):
    cell = nb['cells'][i]
    if 'source' in cell:
        source = cell['source']
        source_str = ''.join(source) if isinstance(source, list) else source
        
        if 'neural_style_transfer(' in source_str and 'style_loss_network' not in source_str:
            new_source = []
            for line in source:
                # Update calls: neural_style_transfer(model, ...) -> neural_style_transfer(style_loss_network, ...)
                # But be careful - we need to update the variable passed, not the parameter name
                # Actually, the calls pass vgg_features, vgg19_features, or alexnet_features, which is fine
                # The function parameter name change is internal
                new_source.append(line)
            cell['source'] = new_source

# 6. Add bridge cell at the end
print("\n6. Adding integration bridge cell at the end...")
bridge_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# ' + '='*60 + '\n',
        '# INTEGRATION: Classify Final Stylized Image with Trained Classifier\n',
        '# ' + '='*60 + '\n',
        '\n',
        '# Get the final stylized image from Part B (use the last result)\n',
        'if "all_results" in globals() and len(all_results) > 0:\n',
        '    # Get the last stylized image (or you can select a specific one)\n',
        '    final_stylized_image = all_results[-1][\'stylized_image\']\n',
        '    \n',
        '    print("Classifying final stylized image with trained_classifier from Part A...")\n',
        '    print("="*60)\n',
        '    \n',
        '    # Preprocess the stylized image for the classifier\n',
        '    # Resize to (224, 224) and normalize with ImageNet stats\n',
        '    classifier_transform = transforms.Compose([\n',
        '        transforms.Resize((224, 224)),\n',
        '        transforms.Normalize(\n',
        '            mean=[0.485, 0.456, 0.406],\n',
        '            std=[0.229, 0.224, 0.225]\n',
        '        )\n',
        '    ])\n',
        '    \n',
        '    # The image is already a tensor, so we need to handle it differently\n',
        '    # Convert from (1, 3, H, W) format, resize, then normalize\n',
        '    import torch.nn.functional as F\n',
        '    \n',
        '    # Resize to 224x224\n',
        '    preprocessed_image = F.interpolate(\n',
        '        final_stylized_image,\n',
        '        size=(224, 224),\n',
        '        mode=\'bilinear\',\n',
        '        align_corners=False\n',
        '    )\n',
        '    \n',
        '    # Normalize (the image should already be normalized from style transfer, but ensure it matches classifier input)\n',
        '    # ImageNet normalization\n',
        '    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)\n',
        '    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)\n',
        '    \n',
        '    # Denormalize first (in case it was normalized), then re-normalize\n',
        '    # Check if it needs normalization - style transfer output should be in [0,1] range\n',
        '    # We need to normalize it for the classifier\n',
        '    # First, ensure it is in [0,1] range, then apply ImageNet normalization\n',
        '    \n',
        '    # Denormalize from style transfer normalization (if applied)\n',
        '    # Style transfer uses ImageNet normalization, so we reverse it first\n',
        '    style_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)\n',
        '    style_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)\n',
        '    \n',
        '    # Reverse normalization: (x * std) + mean\n',
        '    denormalized = preprocessed_image * style_std + style_mean\n',
        '    \n',
        '    # Clamp to [0, 1] range\n',
        '    denormalized = torch.clamp(denormalized, 0, 1)\n',
        '    \n',
        '    # Now normalize for classifier: (x - mean) / std\n',
        '    classifier_input = (denormalized - IMAGENET_MEAN) / IMAGENET_STD\n',
        '    \n',
        '    # Set classifier to eval mode\n',
        '    trained_classifier.eval()\n',
        '    \n',
        '    # Classify the image\n',
        '    with torch.no_grad():\n',
        '        output = trained_classifier(classifier_input)\n',
        '        probabilities = F.softmax(output, dim=1)\n',
        '        van_gogh_prob = probabilities[0][1].item()  # Probability of being Van Gogh (class 1)\n',
        '        not_van_gogh_prob = probabilities[0][0].item()  # Probability of not being Van Gogh (class 0)\n',
        '        prediction = output.argmax(dim=1).item()\n',
        '    \n',
        '    print(f"\\nClassification Results:")\n',
        '    print(f"  Probability of being Van Gogh: {van_gogh_prob:.4f} ({van_gogh_prob*100:.2f}%)")\n',
        '    print(f"  Probability of NOT being Van Gogh: {not_van_gogh_prob:.4f} ({not_van_gogh_prob*100:.2f}%)")\n',
        '    print(f"  Prediction: {\'Van Gogh\' if prediction == 1 else \'Not Van Gogh\'}")\n',
        '    print("="*60)\n',
        'else:\n',
        '    print("No stylized images found. Please run Part B style transfer first.")\n'
    ]
}

nb['cells'].append(bridge_cell)

# Save the refactored notebook
with open('Part_A_and_B_Combined.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✓ Refactoring complete!")
print(f"  - Part A: final_model → trained_classifier")
print(f"  - Part B: Updated references to use trained_classifier")
print(f"  - Part B: Style transfer uses style_loss_network variable")
print(f"  - Added integration bridge cell at the end")
print(f"\nTotal cells: {len(nb['cells'])}")

