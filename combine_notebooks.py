import json

# Read Part A
with open('Part_A.ipynb', 'r') as f:
    part_a = json.load(f)

# Read Part B  
with open('Part_B.ipynb', 'r') as f:
    part_b = json.load(f)

# Combine: Part A cells + separator + Part B cells
combined_cells = part_a['cells'].copy()

# Add separator
separator = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '# ' + '='*60 + '\n',
        '# PART B - NEURAL STYLE TRANSFER\n',
        '# ' + '='*60 + '\n',
        '\n',
        'This section implements style transfer using the trained classifier from Part A.\n',
        'The `final_model` from Part A will be used as the judge for evaluating style transfer quality.'
    ]
}
combined_cells.append(separator)

# Add Part B cells (skip cells 0-6: intro, setup, imports since Part A already has them)
# Start from cell 7 which is "Load Part A Classifiers" but we'll replace it
part_b_cells_to_add = part_b['cells'][7:]  # Start from cell 7

# Modify the first cell (Load Part A Classifiers) to use final_model directly
if len(part_b_cells_to_add) > 0:
    part_b_cells_to_add[0]['source'] = [
        '# Use the trained model from Part A as judge\n',
        '# The final_model from Part A is already trained and ready to use\n',
        '\n',
        'if "final_model" not in globals():\n',
        '    raise NameError(\n',
        '        "final_model not found! Please run Part A first.\\n"\n',
        '        "The model needs to be trained in Part A before using it in Part B."\n',
        '    )\n',
        '\n',
        '# Use final_model as the judge\n',
        'judge_model = final_model\n',
        'judge_model.eval()  # Ensure it\'s in eval mode\n',
        'judge_model_name = best_params.get("model_name", "VGG19") if "best_params" in globals() else "VGG19"\n',
        '\n',
        'print(f"Using {judge_model_name} from Part A as judge for hyperparameter search")\n',
        'print(f"Model validation accuracy: {best_val_acc:.4f}")\n'
    ]

combined_cells.extend(part_b_cells_to_add)

# Create combined notebook
combined_notebook = {
    'cells': combined_cells,
    'metadata': part_a['metadata'],
    'nbformat': 4,
    'nbformat_minor': 2
}

# Save
with open('Part_A_and_B_Combined.ipynb', 'w') as f:
    json.dump(combined_notebook, f, indent=1)

print(f'Combined notebook created successfully!')
print(f'Part A: {len(part_a["cells"])} cells')
print(f'Part B added: {len(part_b_cells_to_add)} cells')
print(f'Total: {len(combined_cells)} cells')

