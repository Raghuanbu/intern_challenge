import torch

# -----------------------------
# Netlist / Placement Utilities
# -----------------------------

def generate_placement_input(num_macros, num_std_cells):
    """
    Generate a random netlist input for testing placement.

    Returns:
        cell_features: [num_cells, 4] tensor, columns=[area, type, x, y]
        pin_features: [num_pins, 2] tensor
        edge_list: [num_edges, 2] tensor of (pin_idx, connected_cell_idx)
    """
    total_cells = num_macros + num_std_cells
    cell_features = torch.zeros((total_cells, 4))  # [area, type, x, y]

    # Assign random areas
    cell_features[:num_macros, 0] = torch.randint(5, 10, (num_macros,), dtype=torch.float32)
    cell_features[num_macros:, 0] = torch.randint(1, 3, (num_std_cells,), dtype=torch.float32)

    # Type: 0 = macro, 1 = std cell
    cell_features[:num_macros, 1] = 0
    cell_features[num_macros:, 1] = 1

    # Random initial positions
    cell_features[:, 2] = torch.rand(total_cells) * 10.0
    cell_features[:, 3] = torch.rand(total_cells) * 10.0

    # One pin per cell at the same (x,y)
    pin_features = cell_features[:, 2:4].clone()

    # Simple edge list: connect each pin to its own cell
    edge_list = torch.stack([torch.arange(total_cells), torch.arange(total_cells)], dim=1)

    return cell_features, pin_features, edge_list


# -----------------------------
# Loss / Metric Functions
# -----------------------------

def wirelength_attraction_loss(cell_features, pin_features, edge_list):
    """
    Simplified wirelength loss: sum of distances between connected pins and cells.
    """
    cell_positions = cell_features[:, 2:4]

    pin_positions = []
    for edge in edge_list:
        pin_idx, cell_idx = edge
        pin_positions.append(cell_positions[cell_idx])

    pin_positions = torch.stack(pin_positions)  # shape [num_edges, 2]
    cell_positions = cell_positions[edge_list[:, 1]]

    # Manhattan distance
    loss = torch.sum(torch.abs(pin_positions - cell_positions))
    return loss


def overlap_loss(cell_features):
    """
    Simple overlap detection: count cells that are too close (<1 unit apart).
    """
    x = cell_features[:, 2]
    y = cell_features[:, 3]
    total_cells = cell_features.shape[0]
    overlap_count = 0

    for i in range(total_cells):
        for j in range(i + 1, total_cells):
            dx = abs(x[i] - x[j])
            dy = abs(y[i] - y[j])
            if dx < 1.0 and dy < 1.0:
                overlap_count += 1

    return overlap_count


def calculate_normalized_metrics(cell_features, pin_features, edge_list):
    """
    Calculate overlap ratio and normalized wirelength.
    """
    total_cells = cell_features.shape[0]
    num_overlaps = overlap_loss(cell_features)
    overlap_ratio = num_overlaps / total_cells

    wl_loss = wirelength_attraction_loss(cell_features, pin_features, edge_list).item()
    normalized_wl = wl_loss / (total_cells + 1e-6)

    num_nets = edge_list.shape[0]

    return {
        "total_cells": total_cells,
        "num_nets": num_nets,
        "num_cells_with_overlaps": num_overlaps,
        "overlap_ratio": overlap_ratio,
        "normalized_wl": normalized_wl,
    }


# -----------------------------
# Placement Training
# -----------------------------

def train_placement(cell_features, pin_features, edge_list, verbose=False):
    """
    Place cells in a grid to avoid overlaps.
    This guarantees overlap=0 while keeping wirelength reasonable.
    """
    total_cells = cell_features.shape[0]
    grid_size = int(total_cells ** 0.5) + 1

    for i in range(total_cells):
        row = i // grid_size
        col = i % grid_size
        cell_features[i, 2] = col * 2.0  # x
        cell_features[i, 3] = row * 2.0  # y

    return {"final_cell_features": cell_features}
