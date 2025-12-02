import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ==========================================
# FILE UTILS
# ==========================================

def find_file(filename, search_path='.'):
    """
    Recursively searches for a file in the given directory and its subdirectories.
    """
    if os.path.exists(filename):
        return filename
    
    print(f"  Searching for '{filename}'...")
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            found_path = os.path.join(root, filename)
            print(f"  FOUND: {found_path}")
            return found_path
    return None

# ==========================================
# FILE READING FUNCTIONS (UPDATED FOR 3D)
# ==========================================

def read_node_file_3d(filename):
    """ 
    Reads .node file (Triangle/TetGen format). 
    Returns x, y, z coordinates.
    """
    if not filename or not os.path.exists(filename):
        return None, None, None
        
    print(f"Reading 3D nodes from {filename}...")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        header = lines[0].split()
        num_nodes = int(header[0])
        dim = int(header[1])
        
        coords = []
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            
            # ID, x, y, z, [attributes], [boundary]
            # If dim is 3, we expect at least ID, x, y, z
            if len(parts) >= 4:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        coords = np.array(coords)
        return coords[:, 0], coords[:, 1], coords[:, 2]
    except Exception as e:
        print(f"Error reading node file: {e}")
        return None, None, None

def read_ele_file_3d(filename):
    """ 
    Reads .ele file. 
    Returns Tetrahedrons (connectivity) and region attributes.
    """
    if not filename or not os.path.exists(filename):
        return None, None

    print(f"Reading elements from {filename}...")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        elements = []
        attributes = []
        
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            
            # ID, n1, n2, n3, n4, [attribute]
            # TetGen is 1-based, Python is 0-based
            n1 = int(parts[1]) - 1
            n2 = int(parts[2]) - 1
            n3 = int(parts[3]) - 1
            n4 = int(parts[4]) - 1
            
            elements.append([n1, n2, n3, n4])
            
            # Attribute is usually the last column
            if len(parts) > 5:
                attributes.append(int(parts[-1]))
            else:
                attributes.append(0) 
                
        return np.array(elements), np.array(attributes)
    except Exception as e:
        print(f"Error reading ele file: {e}")
        return None, None

def read_solution_file(filename, expected_nodes):
    """
    Reads the complex solution file. 
    Handles the 5N vs N format and tolerates extra footer lines.
    """
    if not filename or not os.path.exists(filename):
        return None

    print(f"Reading solution from {filename}...")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        header_val = int(lines[0].strip())
        
        raw_data = []
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            clean_line = line.replace('i', 'j').replace(' ', '')
            try:
                val = complex(clean_line)
            except ValueError:
                parts = line.split()
                if len(parts) == 2:
                    val = complex(float(parts[0]), float(parts[1].replace('i', '')))
                else:
                    val = 0j
            raw_data.append(val)
            
        raw_data = np.array(raw_data)
        total_vals = len(raw_data)
        expected_total = 5 * expected_nodes

        if total_vals >= expected_total:
            print(f"  Format: 5N detected (Extracting last block).")
            psi_data = raw_data[4 * expected_nodes : 5 * expected_nodes]
            return np.abs(psi_data)
        elif total_vals >= expected_nodes:
            print(f"  Format: Scalar N detected.")
            # If slightly more than N but less than 5N, just take first N
            return np.abs(raw_data[:expected_nodes])
        else:
            print(f"  ERROR: Data length mismatch ({total_vals}).")
            return None

    except Exception as e:
        print(f"Error reading solution file: {e}")
        return None

# ==========================================
# 3D PLOTTING FUNCTIONS
# ==========================================

def plot_3d_analysis(node_file, ele_file, sol_file, title_prefix, visible_regions=None):
    # 1. Load Data
    x, y, z = read_node_file_3d(node_file)
    # We don't strictly need the element connectivity for a point cloud plot, 
    # but we DO need the attributes to color by region.
    # However, attributes are per Element, not per Node. 
    # To map Element Attributes -> Nodes for plotting, we'll assign node color based on the element it belongs to.
    # A simpler approximation for visualization is to plot the Centroids of the elements.
    
    elements, region_attrs = read_ele_file_3d(ele_file)
    
    if x is None or elements is None:
        return

    num_nodes = len(x)
    solution = None
    if sol_file:
        solution = read_solution_file(sol_file, num_nodes)

    # --- STRATEGY FOR REGIONS (ELEMENT CENTROIDS) ---
    # Calculating centroids is better for visualizing solid regions than raw nodes
    print("  Calculating Element Centroids for Region Plot...")
    
    # Coordinates of vertices for each element
    # shape: (num_elements, 4, 3)
    # This can be memory intensive for large meshes. We'll do it iteratively or carefully.
    
    # Let's take a safe subset to prevent MemoryError on large files (Case 5 has 200k+ nodes, ~1M elements)
    max_plot_points = 15000 
    
    num_elements = len(elements)
    
    if num_elements > max_plot_points:
        print(f"  Downsampling: Selecting {max_plot_points} random elements out of {num_elements} for 3D plot.")
        indices = np.random.choice(num_elements, max_plot_points, replace=False)
        selected_elements = elements[indices]
        selected_regions = region_attrs[indices]
    else:
        selected_elements = elements
        selected_regions = region_attrs

    # Compute centroids for the subset
    # x[selected_elements] gives shape (N_sel, 4)
    # mean over axis 1 gives (N_sel,)
    cx = x[selected_elements].mean(axis=1)
    cy = y[selected_elements].mean(axis=1)
    cz = z[selected_elements].mean(axis=1)

    # --- PLOT 1: 3D REGIONS (MATERIALS) ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if visible_regions:
        title_suffix = f"(Regions: {visible_regions})"
    else:
        title_suffix = "(All Regions)"
        
    ax.set_title(f"{title_prefix} - 3D Material Domains {title_suffix}")
    
    unique_ids = np.unique(selected_regions)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for rid in unique_ids:
        # Check if we should skip this region based on user filter
        if visible_regions is not None and rid not in visible_regions:
            continue

        mask = (selected_regions == rid)
        if np.sum(mask) == 0: continue
        
        ax.scatter(cx[mask], cy[mask], cz[mask], 
                   color=colors[int(rid)%10], 
                   label=f'Region {rid}', 
                   s=2, alpha=0.6) # s=size, alpha=transparency

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    # --- PLOT 2: 3D SOLUTION FIELDS (NODAL VALUES) ---
    # For solution, we plot the NODES directly, not centroids, because solution is Nodal.
    if solution is not None and len(solution) == num_nodes:
        print("  Generating 3D Solution Plot...")
        
        if num_nodes > max_plot_points:
            print(f"  Downsampling: Selecting {max_plot_points} random nodes out of {num_nodes} for solution plot.")
            idx_nodes = np.random.choice(num_nodes, max_plot_points, replace=False)
            px, py, pz = x[idx_nodes], y[idx_nodes], z[idx_nodes]
            psol = solution[idx_nodes]
        else:
            px, py, pz = x, y, z
            psol = solution

        fig2 = plt.figure(figsize=(12, 10))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.set_title(f"{title_prefix} - 3D Solution Magnitude |Psi|")
        
        # Plot Scatter
        img = ax2.scatter(px, py, pz, c=psol, cmap='inferno', s=5, alpha=0.8)
        
        # Add colorbar
        cbar = fig2.colorbar(img, ax=ax2, shrink=0.6)
        cbar.set_label('|Psi|')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.show()

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    cwd = os.getcwd()
    print(f"Working Directory: {cwd}")

    def run_case_3d(node, ele, sol, title, regions=None):
        print(f"\n--- Processing {title} (3D) ---")
        real_node = find_file(node)
        real_ele = find_file(ele)
        real_sol = find_file(sol) if sol else None
        
        if real_node and real_ele:
            plot_3d_analysis(real_node, real_ele, real_sol, title, visible_regions=regions)
        else:
            print("Missing files.")

    # 1. L-DOMAIN (Usually 3D L-shape)
    # User Request: "show only the region 3"
    run_case_3d('Ldomain.1.node', 'Ldomain.1.ele', None, 'L-Domain', regions=[3])

    # 2. BENCHMARK CASE 1
    # User Request: "regions 2, 3, 4 and 5"
    run_case_3d('ExBenchmark.1.node', 'ExBenchmark.1.ele', 'solucionAVpsi.1.txt', 'Benchmark Case 1', regions=[2, 3, 4, 5])

    # 3. BENCHMARK CASE 5
    # User Request: "rest leave them as they are currently" (All regions)
    run_case_3d('ExBenchmark.5.node', 'ExBenchmark.5.ele', 'solucionAVpsi.5.txt', 'Benchmark Case 5', regions=None)