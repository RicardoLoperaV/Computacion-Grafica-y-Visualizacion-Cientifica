import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os

def read_node_file(filename):
    """
    Reads .node file (Triangle format).
    Returns x and y coordinates.
    """
    print(f"Reading nodes from {filename}...")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        # First line is usually: <# of vertices> <dimension> <# of attributes> <# of boundary markers>
        header = lines[0].split()
        num_nodes = int(header[0])
        
        coords = []
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            # ID, x, y, [attributes], [boundary marker]
            # usually index 0 is ID, 1 is x, 2 is y
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
    print(f"Reading elements from {filename}...")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        # First line: <# of triangles> <nodes per triangle> <# of attributes>
        header = lines[0].split()
        # num_tris = int(header[0])
        
        triangles = []
        attributes = []
        
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            
            # Format: ID, node1, node2, node3, [attribute]
            # Triangle is 1-based index, Python is 0-based.
            n1 = int(parts[1]) - 1
            n2 = int(parts[2]) - 1
            n3 = int(parts[3]) - 1
            
            triangles.append([n1, n2, n3])
            
            # The region attribute is usually the last column
            if len(parts) > 4:
                attributes.append(int(parts[-1]))
            else:
                attributes.append(0) # Default if no attribute
                
        return np.array(triangles), np.array(attributes)
    except Exception as e:
        print(f"Error reading ele file: {e}")
        return None, None

def read_solution_file(filename):
    """
    Reads the complex solution file.
    Format assumes: Real Part (+/-) Imaginary Part i
    Returns magnitude of the solution.
    """
    print(f"Reading solution from {filename}...")
    try:
        values = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Skip header (node count)
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            
            # Replace 'i' with 'j' for Python complex parsing and remove whitespaces carefully
            # The file format seems to be: "-0.0000164552 -0.0000324275i"
            clean_line = line.replace('i', 'j').replace(' ', '')
            
            # Sometimes there is a space between the number and the sign, 
            # but Python's complex() is picky. 
            # Alternative manual parsing if complex() fails:
            try:
                val = complex(clean_line)
            except ValueError:
                # Fallback manual parser for formats like "-1.23 - 4.56i" with spaces
                parts = line.split()
                if len(parts) == 2:
                    real_part = float(parts[0])
                    # Handle the imaginary part which includes 'i'
                    imag_part_str = parts[1].replace('i', '')
                    imag_part = float(imag_part_str)
                    val = complex(real_part, imag_part)
                else:
                    val = 0j

            values.append(np.abs(val)) # Storing Magnitude
            
        return np.array(values)
    except Exception as e:
        print(f"Error reading solution file: {e}")
        return None

def plot_domain_split(node_file, ele_file, sol_file, title_prefix):
    # 1. Load Data
    x, y = read_node_file(node_file)
    triangles, regions = read_ele_file(ele_file)
    solution = read_solution_file(sol_file)

    if x is None or triangles is None:
        print("Failed to load mesh data.")
        return

    # Identify unique regions (materials)
    unique_regions = np.unique(regions)
    print(f"Found Regions (Material IDs): {unique_regions}")

    # --- PLOT 1: THE GRID (MESH) SPLIT BY DOMAIN ---
    plt.figure(figsize=(12, 10))
    plt.title(f"{title_prefix} - Mesh Grid (Split by Domain)")
    
    # Define colors for different regions
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regions)))

    for idx, region_id in enumerate(unique_regions):
        # Mask: Select only triangles belonging to this region
        mask = (regions == region_id)
        tris_in_region = triangles[mask]
        
        # Create a triangulation for just this region
        tri_obj = tri.Triangulation(x, y, tris_in_region)
        
        # Plot edges
        plt.triplot(tri_obj, color=colors[idx], lw=0.5, label=f'Region {region_id}')

    plt.legend()
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- PLOT 2: THE SOLUTION (MAGNITUDE) ---
    if solution is not None and len(solution) == len(x):
        plt.figure(figsize=(12, 10))
        plt.title(f"{title_prefix} - Solution Magnitude |Psi|")
        
        # We can plot the whole solution at once, but overlay the region boundaries
        full_tri = tri.Triangulation(x, y, triangles)
        
        # Plot filled contours
        cntr = plt.tricontourf(full_tri, solution, levels=20, cmap='viridis')
        plt.colorbar(cntr, label='Magnitude')
        
        # Overlay grid lines lightly (optional, remove if too messy)
        # plt.triplot(full_tri, color='white', lw=0.1, alpha=0.5)

        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    else:
        print(f"Solution size {len(solution) if solution is not None else 0} does not match nodes {len(x)}.")

# ==========================================
# EXECUTION BLOCKS
# ==========================================

# CASE 1: L-DOMAIN
# The L-domain usually has different regions for refinement or layers
print("\n--- Processing L-Domain ---")
plot_domain_split(
    node_file='Ldomain.1.node',
    ele_file='Ldomain.1.ele',
    # Note: You didn't upload a solution for Ldomain, only mesh. 
    # If you have one, put filename here. Otherwise passing None.
    sol_file='solucionAVpsi.1.txt', # Assuming this might correspond or is placeholder
    title_prefix='L-Domain'
)

# CASE 2: BENCHMARK (Team 7)
# This usually contains the Coil, the Plate, and the Air.
print("\n--- Processing Benchmark (Team 7) ---")
plot_domain_split(
    node_file='ExBenchmark.1.node',  # Or .5.node
    ele_file='ExBenchmark.1.ele',    # Or .5.ele
    sol_file='solucionAVpsi.1.txt',  # Or .5.txt
    title_prefix='Benchmark Team 7'
)