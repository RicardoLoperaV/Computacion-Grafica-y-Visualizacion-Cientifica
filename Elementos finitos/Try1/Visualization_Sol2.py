import numpy as np
import meshio
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_complex_solution(filepath):
    """
    Reads a solution file with format 'real imag i' or similar.
    Returns: Real part, Imaginary part, and Magnitude.
    """
    print(f"  Reading solution: {filepath}")
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header (line 1 contains the count)
    data_lines = lines[1:]
    
    real_vals = []
    imag_vals = []
    
    for line in data_lines:
        # Replace 'i' with 'j' for Python's complex parser, remove whitespace
        clean_line = line.strip().replace('i', 'j').replace(' ', '')
        # Handle cases like "-1.2 -3.4j" where spaces might be tricky. 
        # Better approach: string replace 'i' with 'j' and trust complex()
        # The file format seems to be: "-0.00001 -0.00003i" (space separated)
        
        # Robust parsing for "num1 num2i" format
        parts = line.strip().replace('i', '').split()
        if len(parts) >= 2:
            # Assuming format: RealPart ImagPart (with sign attached)
            # We construct a complex number string
            r = float(parts[0])
            i = float(parts[1])
            real_vals.append(r)
            imag_vals.append(i)
        else:
            # Fallback for other formats
            try:
                c = complex(line.strip().replace('i', 'j'))
                real_vals.append(c.real)
                imag_vals.append(c.imag)
            except ValueError:
                continue

    real_arr = np.array(real_vals)
    imag_arr = np.array(imag_vals)
    mag_arr = np.sqrt(real_arr**2 + imag_arr**2)
    
    return real_arr, imag_arr, mag_arr

def read_triangle_mesh(node_file, ele_file):
    """
    Reads Triangle .node and .ele files.
    Returns: points (3D), cells, and region_ids.
    """
    print(f"  Reading geometry: {node_file}")
    
    # --- 1. Nodes ---
    # .node file: <index> <x> <y> [z]
    # We force Z=0 for 2D meshes to make them compatible with 3D VTK
    nodes = np.loadtxt(node_file, skiprows=1, usecols=(1, 2))
    num_nodes = nodes.shape[0]
    
    # Create 3D points (x, y, 0.0)
    points_3d = np.column_stack((nodes, np.zeros(num_nodes)))
    
    # --- 2. Elements & Regions ---
    print(f"  Reading connectivity: {ele_file}")
    # .ele file: <index> <n1> <n2> <n3> <region_attribute>
    # We read columns 1,2,3 for triangles and 4 for Region ID
    try:
        # Try reading 4 columns (connectivity) + 1 column (region)
        data = np.loadtxt(ele_file, skiprows=1, usecols=(1, 2, 3, 4), dtype=int)
        elements = data[:, 0:3]
        region_ids = data[:, 3]
    except:
        # Fallback if no region column exists
        data = np.loadtxt(ele_file, skiprows=1, usecols=(1, 2, 3), dtype=int)
        elements = data
        region_ids = np.zeros(data.shape[0], dtype=int)

    # Convert 1-based indexing (Triangle) to 0-based (Python/VTK)
    cells = elements - 1
    
    return points_3d, cells, region_ids

def create_vtk(node_path, ele_path, sol_path, output_name):
    # 1. Read Mesh
    points, cells, region_ids = read_triangle_mesh(node_path, ele_path)
    
    # 2. Read Solution (if provided)
    point_data = {}
    if sol_path and os.path.exists(sol_path):
        re, im, mag = parse_complex_solution(sol_path)
        
        # Check consistency
        if len(re) != len(points):
            print(f"  [Warning] Mismatch: Mesh has {len(points)} nodes, Solution has {len(re)} values.")
            print(f"  Skipping solution data for {output_name}.")
        else:
            point_data["Solution_Real"] = re
            point_data["Solution_Imag"] = im
            point_data["Solution_Magnitude"] = mag
    
    # 3. Create Cell Data (for splitting domains)
    cell_data = {"RegionID": [region_ids]}
    
    # 4. Write VTU
    print(f"  Writing {output_name}...")
    mesh = meshio.Mesh(
        points,
        [("triangle", cells)],
        point_data=point_data,
        cell_data=cell_data
    )
    mesh.write(output_name)
    print("  Done.\n")

# ==========================================
# Processing List
# ==========================================
# Update this list with your specific file pairs
tasks = [
    {
        "node": "ExBenchmark.1.node",
        "ele": "ExBenchmark.1.ele",
        "sol": "solucionAVpsi.1.txt",
        "out": "Benchmark_1_Result.vtu"
    },
    {
        "node": "ExBenchmark.5.node",
        "ele": "ExBenchmark.5.ele",
        "sol": "solucionAVpsi.5.txt",
        "out": "Benchmark_5_Result.vtu"
    },
    # Note: Ldomain.1.node was uploaded, but no matching solution file was found 
    # with the same node count (1239). We generate the mesh geometry anyway.
    {
        "node": "Ldomain.1.node",
        "ele": "Ldomain.1.ele",
        "sol": None, # No matching solution provided
        "out": "Ldomain_Geometry.vtu"
    }
]

if __name__ == "__main__":
    print("Starting VTK Conversion...")
    for task in tasks:
        # Build full paths relative to script directory
        node_path = os.path.join(SCRIPT_DIR, task["node"])
        ele_path = os.path.join(SCRIPT_DIR, task["ele"])
        sol_path = os.path.join(SCRIPT_DIR, task["sol"]) if task["sol"] else None
        out_path = os.path.join(SCRIPT_DIR, task["out"])
        
        if os.path.exists(node_path) and os.path.exists(ele_path):
            create_vtk(node_path, ele_path, sol_path, out_path)
        else:
            print(f"  [Error] Missing files for {task['out']}")
            if not os.path.exists(node_path):
                print(f"    Missing: {node_path}")
            if not os.path.exists(ele_path):
                print(f"    Missing: {ele_path}")