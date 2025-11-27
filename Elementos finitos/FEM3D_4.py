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
# 3D GEOMETRY UTILS
# ==========================================

def get_boundary_faces(tets):
    """
    Input: Array of Tetrahedrons (N, 4)
    Output: Array of Faces (M, 3) that are on the boundary (appear exactly once).
    """
    # 1. Decompose Tets into 4 faces each
    # Faces are: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    print("    Extracting mesh boundary...")
    all_faces = np.vstack([
        tets[:, [0, 1, 2]],
        tets[:, [0, 1, 3]],
        tets[:, [0, 2, 3]],
        tets[:, [1, 2, 3]]
    ])
    
    # 2. Sort indices within each face so (1,2,3) == (3,2,1)
    all_faces.sort(axis=1)
    
    # 3. Find unique faces and their counts
    # Using numpy unique with axis=0 (requires modern numpy)
    unique_faces, counts = np.unique(all_faces, axis=0, return_counts=True)
    
    # 4. Boundary faces are those that appear exactly once
    boundary_faces = unique_faces[counts == 1]
    
    return boundary_faces

# ==========================================
# 3D PLOTTING FUNCTIONS
# ==========================================

def plot_3d_analysis(node_file, ele_file, sol_file, title_prefix, visible_regions=None):
    # 1. Load Data
    x, y, z = read_node_file_3d(node_file)
    elements, region_attrs = read_ele_file_3d(ele_file)
    
    if x is None or elements is None:
        return

    num_nodes = len(x)
    solution = None
    if sol_file:
        solution = read_solution_file(sol_file, num_nodes)

    # --- PLOT 1: 3D MATERIAL DOMAINS ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if visible_regions:
        title_suffix = f"(Surface: {visible_regions})"
    else:
        title_suffix = "(All Regions - Point Cloud)"
        
    ax.set_title(f"{title_prefix} - 3D Material Domains {title_suffix}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    unique_ids = np.unique(region_attrs)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # --- MODE SWITCH: SURFACE vs POINT CLOUD ---
    # If the user specified specific regions, we assume they want a high-quality surface plot.
    # If not (e.g. Case 5 global), we assume dataset is too large and stick to point clouds.
    if visible_regions is not None:
        print("  Generating Surface Plot (Boundary Extraction)...")
        # For selected regions, we extract the BOUNDARY FACES (skin) and plot trisurf
        for rid in unique_ids:
            if rid not in visible_regions:
                continue
            
            # Get elements for this region
            mask = (region_attrs == rid)
            region_tets = elements[mask]
            
            if len(region_tets) == 0: continue
            
            # Extract boundary skin
            boundary_tris = get_boundary_faces(region_tets)
            
            if len(boundary_tris) > 0:
                print(f"    Region {rid}: Plotting {len(boundary_tris)} boundary triangles.")
                # Plot Surface
                ax.plot_trisurf(x, y, z, triangles=boundary_tris,
                                color=colors[int(rid) % 10],
                                alpha=0.5,
                                edgecolor='none', # Remove edges for cleaner surface
                                label=f'Region {rid}')
    else:
        # --- FALLBACK: SCATTER PLOT FOR FULL DATASET ---
        print("  Generating Point Cloud Plot (Full Dataset)...")
        # Same centroid logic as before for performance
        max_plot_points = 15000 
        num_elements = len(elements)
        
        if num_elements > max_plot_points:
            indices = np.random.choice(num_elements, max_plot_points, replace=False)
            sel_elements = elements[indices]
            sel_regions = region_attrs[indices]
        else:
            sel_elements = elements
            sel_regions = region_attrs

        cx = x[sel_elements].mean(axis=1)
        cy = y[sel_elements].mean(axis=1)
        cz = z[sel_elements].mean(axis=1)
        
        for rid in unique_ids:
            mask = (sel_regions == rid)
            if np.sum(mask) == 0: continue
            ax.scatter(cx[mask], cy[mask], cz[mask], 
                       color=colors[int(rid)%10], 
                       label=f'Region {rid}', 
                       s=2, alpha=0.6)

    # Hack to make legend work for surfaces (matplotlib 3d legend is tricky)
    # We create fake proxies
    import matplotlib.lines as mlines
    proxies = []
    labels = []
    plot_ids = visible_regions if visible_regions else unique_ids
    for rid in plot_ids:
        if rid in unique_ids:
            proxies.append(mlines.Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=colors[int(rid)%10]))
            labels.append(f"Region {rid}")
    ax.legend(proxies, labels, loc='upper right')
    
    plt.show()

    # --- PLOT 2: 3D SOLUTION FIELDS (NODAL VALUES) ---
    # Keeping this as scatter for now as 'surface solution' is complex (requires mapped colors on trisurf)
    # and user specifically asked to modify the "Material Domains" plot.
    if solution is not None and len(solution) == num_nodes:
        print("  Generating 3D Solution Plot (Point Cloud)...")
        max_sol_points = 15000
        
        if num_nodes > max_sol_points:
            idx_nodes = np.random.choice(num_nodes, max_sol_points, replace=False)
            px, py, pz = x[idx_nodes], y[idx_nodes], z[idx_nodes]
            psol = solution[idx_nodes]
        else:
            px, py, pz = x, y, z
            psol = solution

        fig2 = plt.figure(figsize=(12, 10))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.set_title(f"{title_prefix} - 3D Solution Magnitude |Psi|")
        
        img = ax2.scatter(px, py, pz, c=psol, cmap='inferno', s=5, alpha=0.8)
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

    # 1. L-DOMAIN
    # Surface Plot for Region 3
    run_case_3d('Ldomain.1.node', 'Ldomain.1.ele', None, 'L-Domain', regions=[3])

    # 2. BENCHMARK CASE 1
    # Surface Plot for Regions 3 and 5 (Updated: removed 2 and 4)
    run_case_3d('ExBenchmark.1.node', 'ExBenchmark.1.ele', 'solucionAVpsi.1.txt', 'Benchmark Case 1', regions=[3, 5])

    # 3. BENCHMARK CASE 5
    # Point Cloud for All Regions (Performance Safe)
    run_case_3d('ExBenchmark.5.node', 'ExBenchmark.5.ele', 'solucionAVpsi.5.txt', 'Benchmark Case 5', regions=None)