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
# FILE READING FUNCTIONS
# ==========================================

def read_node_file(filename):
    """ Reads .node file (Triangle format). Returns x and y coordinates. """
    if not filename or not os.path.exists(filename):
        return None, None
        
    print(f"Reading nodes from {filename}...")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        header = lines[0].split()
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
    """ Reads .ele file. Returns triangles and region attributes. """
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
            
            # Triangle format is 1-based, Python is 0-based
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
    Reads the complex solution file. 
    Tolerates extra values (e.g., 5N + 1) by slicing the required data.
    """
    if not filename or not os.path.exists(filename):
        return None

    print(f"Reading solution from {filename}...")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        header_val = int(lines[0].strip())
        print(f"  File header says {header_val} nodes. Expected {expected_nodes}.")

        raw_data = []
        # Skip header, read all subsequent lines
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            
            # Clean 'i' to 'j' for python complex type
            clean_line = line.replace('i', 'j').replace(' ', '')
            try:
                val = complex(clean_line)
            except ValueError:
                # Manual fallback for formats like "1.23 - 4.56i"
                parts = line.split()
                if len(parts) == 2:
                    real_part = float(parts[0])
                    imag_part = float(parts[1].replace('i', ''))
                    val = complex(real_part, imag_part)
                else:
                    val = 0j
            raw_data.append(val)
            
        raw_data = np.array(raw_data)
        total_vals = len(raw_data)
        expected_total = 5 * expected_nodes

        # --- DATA EXTRACTION LOGIC ---
        # 1. Check for A(3N) + V(N) + Psi(N) structure
        if total_vals >= expected_total:
            if total_vals > expected_total:
                print(f"  Note: Found {total_vals} values (Expected {expected_total}). Truncating extra {total_vals - expected_total} values.")
            
            # Psi is the LAST block of N values in the 5N sequence
            # Indices: [0...3N-1] is A
            #          [3N...4N-1] is V
            #          [4N...5N-1] is Psi
            psi_start = 4 * expected_nodes
            psi_end = 5 * expected_nodes
            
            print("  Extracting Psi (Indices 4N to 5N)...")
            psi_data = raw_data[psi_start : psi_end]
            return np.abs(psi_data)

        # 2. Check for simple Scalar(N) structure
        elif total_vals == expected_nodes:
            print("  Format detected: Single scalar field (N).")
            return np.abs(raw_data)
            
        else:
            print(f"  ERROR: Data length mismatch. Found {total_vals}, needed {expected_total} or {expected_nodes}.")
            return None

    except Exception as e:
        print(f"Error reading solution file: {e}")
        return None

def plot_domain_split(node_file, ele_file, sol_file, title_prefix):
    # 1. Load Mesh
    x, y = read_node_file(node_file)
    triangles, regions = read_ele_file(ele_file)

    if x is None or triangles is None:
        return

    num_nodes = len(x)
    
    # 2. Load Solution
    solution = None
    if sol_file:
        solution = read_solution_file(sol_file, num_nodes)

    unique_regions = np.unique(regions)
    print(f"Found Regions (Material IDs): {unique_regions}")

    # --- PLOT 1: MESH SPLIT ---
    plt.figure(figsize=(10, 8))
    plt.title(f"{title_prefix} - Mesh Grid (Split by Domain)")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, region_id in enumerate(unique_regions):
        mask = (regions == region_id)
        tris_in_region = triangles[mask]
        
        if len(tris_in_region) == 0: continue
        
        # Plot only if we have data
        plt.triplot(x, y, tris_in_region, color=colors[int(region_id) % 10], lw=0.5, label=f'Region {region_id}')

    # Optimize Legend: Use fixed location to avoid 'best' calculation lag
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout() # Adjust layout to fit legend
    plt.show()

    # --- PLOT 2: SOLUTION ---
    if solution is not None and len(solution) == num_nodes:
        plt.figure(figsize=(10, 8))
        plt.title(f"{title_prefix} - Solution Magnitude |Psi|")
        
        full_tri = tri.Triangulation(x, y, triangles)
        
        # Use more levels for smoother gradient
        cntr = plt.tricontourf(full_tri, solution, levels=100, cmap='inferno') # 'inferno' or 'jet' or 'viridis'
        plt.colorbar(cntr, label='Magnitude |Psi|')
        
        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    cwd = os.getcwd()
    print(f"Working Directory: {cwd}")

    def run_case(node, ele, sol, title):
        print(f"\n--- Processing {title} ---")
        real_node = find_file(node)
        real_ele = find_file(ele)
        real_sol = find_file(sol) if sol else None
        
        if real_node and real_ele:
            plot_domain_split(real_node, real_ele, real_sol, title)
        else:
            print("Missing necessary mesh files.")

    # Run Cases
    run_case('Ldomain.1.node', 'Ldomain.1.ele', None, 'L-Domain')
    run_case('ExBenchmark.1.node', 'ExBenchmark.1.ele', 'solucionAVpsi.1.txt', 'Benchmark Case 1')
    run_case('ExBenchmark.5.node', 'ExBenchmark.5.ele', 'solucionAVpsi.5.txt', 'Benchmark Case 5')