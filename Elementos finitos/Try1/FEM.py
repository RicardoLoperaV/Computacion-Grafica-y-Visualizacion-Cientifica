import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os

# ==========================================
# FILE UTILS
# ==========================================

def find_file(filename, search_path='.'):
    """
    Recursively searches for a file in the given directory and its subdirectories.
    Returns the full path if found, or None if not found.
    """
    # 1. Check direct path or current directory
    if os.path.exists(filename):
        return filename
    
    # 2. Walk through subdirectories
    print(f"  Searching for '{filename}'...")
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            found_path = os.path.join(root, filename)
            print(f"  FOUND: {found_path}")
            return found_path
            
    return None

# ==========================================
# FILE READING FUNCTIONS
# ==========================================

def read_node_file(filename):
    """
    Reads .node file (Triangle format).
    Returns x and y coordinates.
    """
    if not filename or not os.path.exists(filename):
        return None, None
        
    print(f"Reading nodes from {filename}...")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        header = lines[0].split()
        # header[0] is num_nodes
        
        coords = []
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
        
        coords = np.array(coords)
        return coords[:, 0], coords[:, 1]
    except Exception as e:
        print(f"Error reading node file: {e}")
        return None, None

def read_ele_file(filename):
    """
    Reads .ele file (Triangle format).
    Returns triangles (connectivity) and region attributes (material IDs).
    """
    if not filename or not os.path.exists(filename):
        return None, None

    print(f"Reading elements from {filename}...")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        triangles = []
        attributes = []
        
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            
            n1 = int(parts[1]) - 1
            n2 = int(parts[2]) - 1
            n3 = int(parts[3]) - 1
            
            triangles.append([n1, n2, n3])
            
            if len(parts) > 4:
                attributes.append(int(parts[-1]))
            else:
                attributes.append(0) 
                
        return np.array(triangles), np.array(attributes)
    except Exception as e:
        print(f"Error reading ele file: {e}")
        return None, None

def read_solution_file(filename, expected_nodes):
    """
    Reads the complex solution file with A-V-Psi format.
    """
    if not filename or not os.path.exists(filename):
        return None

    print(f"Reading solution from {filename}...")
    try:
        values = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        header_val = int(lines[0].strip())
        print(f"  File header says {header_val} nodes. Expected {expected_nodes}.")

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
                    real_part = float(parts[0])
                    imag_part = float(parts[1].replace('i', ''))
                    val = complex(real_part, imag_part)
                else:
                    val = 0j
            raw_data.append(val)
            
        raw_data = np.array(raw_data)
        total_lines = len(raw_data)
        
        if total_lines == 5 * expected_nodes:
            print("  Format detected: A (3N), V (N), Psi (N).")
            print("  Extracting Psi (last N values)...")
            psi_data = raw_data[4 * expected_nodes : 5 * expected_nodes]
            return np.abs(psi_data)
        elif total_lines == expected_nodes:
            print("  Format detected: Single scalar field (N). Using as is.")
            return np.abs(raw_data)
        else:
            print(f"  WARNING: File has {total_lines} values. Expected 5*{expected_nodes} or {expected_nodes}.")
            return np.abs(raw_data[:expected_nodes])

    except Exception as e:
        print(f"Error reading solution file: {e}")
        return None

def plot_domain_split(node_file, ele_file, sol_file, title_prefix):
    # Files are pre-located in the main block now
    x, y = read_node_file(node_file)
    triangles, regions = read_ele_file(ele_file)

    if x is None or triangles is None:
        return

    num_nodes = len(x)
    solution = None
    if sol_file:
        solution = read_solution_file(sol_file, num_nodes)

    unique_regions = np.unique(regions)
    print(f"Found Regions (Material IDs): {unique_regions}")

    # Plot 1: Grid Split
    plt.figure(figsize=(10, 8))
    plt.title(f"{title_prefix} - Mesh Grid (Split by Domain)")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, region_id in enumerate(unique_regions):
        mask = (regions == region_id)
        tris_in_region = triangles[mask]
        if len(tris_in_region) == 0: continue
        plt.triplot(x, y, tris_in_region, color=colors[int(region_id) % 10], lw=0.5, label=f'Region {region_id}')

    plt.legend()
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot 2: Solution
    if solution is not None and len(solution) == num_nodes:
        plt.figure(figsize=(10, 8))
        plt.title(f"{title_prefix} - Solution Magnitude |Psi|")
        full_tri = tri.Triangulation(x, y, triangles)
        cntr = plt.tricontourf(full_tri, solution, levels=50, cmap='viridis')
        plt.colorbar(cntr, label='Magnitude |Psi|')
        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

# ==========================================
# EXECUTION WITH AUTO-SEARCH
# ==========================================

if __name__ == "__main__":
    print("=== STARTING ANALYSIS ===")
    cwd = os.getcwd()
    print(f"Searching for files starting from: {cwd}")

    def run_case(node_name, ele_name, sol_name, title):
        print(f"\n--- Processing {title} ---")
        
        # 1. Locate files anywhere in subfolders
        real_node = find_file(node_name)
        real_ele = find_file(ele_name)
        real_sol = find_file(sol_name) if sol_name else None
        
        # 2. Check strict requirements
        if not real_node:
            print(f"ERROR: Could not find mesh node file: '{node_name}'")
            return
        if not real_ele:
            print(f"ERROR: Could not find mesh ele file: '{ele_name}'")
            return
        if sol_name and not real_sol:
            print(f"WARNING: Could not find solution file: '{sol_name}'")

        # 3. Run Plotter with resolved full paths
        plot_domain_split(real_node, real_ele, real_sol, title)

    # Run the cases
    run_case('Ldomain.1.node', 'Ldomain.1.ele', None, 'L-Domain')
    run_case('ExBenchmark.1.node', 'ExBenchmark.1.ele', 'solucionAVpsi.1.txt', 'Benchmark Case 1')
    run_case('ExBenchmark.5.node', 'ExBenchmark.5.ele', 'solucionAVpsi.5.txt', 'Benchmark Case 5')