import numpy as np
import meshio
import os

def read_triangle_files(node_file, ele_file):
    """
    Reads .node and .ele files (Triangle format) and returns points and cells
    adjusted for 0-based indexing and 3D VTK format.
    """
    print(f"Reading mesh geometry: {node_file}, {ele_file}...")

    # --- 1. Read Nodes ---
    # Read header to get number of nodes (optional validation, but good practice)
    with open(node_file, 'r') as f:
        header = f.readline().split()
        num_nodes = int(header[0])
    
    # Load node data, skipping header. Assuming standard format: <index> <x> <y> [attributes]
    # We only need columns 1 and 2 (X and Y coordinates)
    try:
        raw_nodes = np.loadtxt(node_file, skiprows=1, usecols=(1, 2))
    except Exception as e:
        print(f"Error reading {node_file}. Ensure it matches standard Triangle format.")
        raise e

    if raw_nodes.shape[0] != num_nodes:
        print(f"Warning: Header said {num_nodes} nodes, but found {raw_nodes.shape[0]}.")
        
    # VTK requires 3D points. Add a Z coordinate of 0 column.
    points_3d = np.column_stack((raw_nodes, np.zeros(raw_nodes.shape[0])))


    # --- 2. Read Elements (Connectivity) ---
    # Read header to get number of triangles
    with open(ele_file, 'r') as f:
        header = f.readline().split()
        # num_triangles = int(header[0]) # Not strictly needed by numpy loadtxt

    # Load element data, skipping header. Assuming format: <index> <n1> <n2> <n3> [attributes]
    # We need columns 1, 2, and 3.
    try:
        raw_elements = np.loadtxt(ele_file, skiprows=1, usecols=(1, 2, 3), dtype=int)
    except Exception as e:
        print(f"Error reading {ele_file}.")
        raise e

    # CRITICAL STEP: Convert 1-based indexing (Triangle standard) to 0-based indexing (Python standard)
    triangles_0_based = raw_elements - 1

    return points_3d, triangles_0_based


def read_complex_solution(solution_path, num_nodes):
    """
    Reads solution file with complex numbers in format: real imag (e.g., -0.0001 +0.0002i)
    Returns the magnitude of the complex values for the first num_nodes values.
    The file may contain multiple solution fields, so we only read the first set.
    """
    complex_values = []
    with open(solution_path, 'r') as f:
        lines = f.readlines()
    
    # First line is typically the count header, skip it
    # Only read up to num_nodes values (first solution field)
    for line in lines[1:num_nodes + 1]:
        line = line.strip()
        if not line:
            continue
        # Parse complex number format: "real imag" where imag ends with 'i'
        # Example: "-0.0000164552 -0.0000324275i"
        parts = line.split()
        if len(parts) == 2:
            real = float(parts[0])
            imag_str = parts[1].replace('i', '')  # Remove 'i' suffix
            imag = float(imag_str)
            complex_values.append(complex(real, imag))
        else:
            # Single real value
            complex_values.append(complex(float(parts[0]), 0))
    
    # Return magnitude of complex numbers for visualization
    return np.abs(np.array(complex_values))


def create_vtk(node_path, ele_path, solution_path, output_filename):
    """
    Combines mesh geometry and solution data into a VTU file.
    """
    # 1. Get geometry
    points, triangles = read_triangle_files(node_path, ele_path)

    # 2. Read Solution Data (complex numbers)
    print(f"Reading solution data: {solution_path}...")
    try:
        solution_data = read_complex_solution(solution_path, len(points))
    except Exception as e:
        print(f"Error reading solution file {solution_path}.")
        raise e

    # Validation: Ensure number of solution values matches number of nodes
    if len(solution_data) != len(points):
        raise ValueError(f"Mismatch! Mesh has {len(points)} nodes, but solution file has {len(solution_data)} values.")

    # 3. Create Meshio Object
    # Define cells format for meshio
    cells = [("triangle", triangles)]
    
    # Create the mesh object, associating the solution data with the points (nodes)
    # We call the data field "PSI_Solution" based on your file names.
    mesh = meshio.Mesh(
        points,
        cells,
        point_data={"PSI_Solution": solution_data}
    )

    # 4. Write to VTK (.vtu format is recommended for unstructured grids)
    print(f"Writing output to: {output_filename}")
    mesh.write(output_filename)
    print("Success.\n")


# ==========================================
# Main Processing Script
# ==========================================
if __name__ == "__main__":
    # Change to the script's directory so relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Define the sets of files to process based on your upload list.
    # Based on the naming convention, it seems solution.1 goes with mesh.1 files, etc.
    
    tasks = [
        # Task 1: The L-Domain (matches your target image)
        {
            "node": "Ldomain.1.node",
            "ele": "Ldomain.1.ele",
            "sol": "solucionAVpsi.1.txt",
            "out": "Ldomain_Result.vtu"
        },
        # Task 2: Benchmark 1
        {
            "node": "ExBenchmark.1.node",
            "ele": "ExBenchmark.1.ele",
            "sol": "solucionAVpsi.1.txt",
            # Note: I am assuming solucionAVpsi.1.txt applies to both Ldomain.1 and ExBenchmark.1
            # based on the file list provided. If this is incorrect, adjust accordingly.
            "out": "Benchmark1_Result.vtu"
        },
        # Task 3: Benchmark 5
        {
            "node": "ExBenchmark.5.node",
            "ele": "ExBenchmark.5.ele",
            "sol": "solucionAVpsi.5.txt",
            "out": "Benchmark5_Result.vtu"
        }
    ]

    print("Starting VTK conversion process...")
    print("="*30 + "\n")

    for task in tasks:
        # Check if files exist before attempting processing
        if not (os.path.exists(task["node"]) and os.path.exists(task["ele"]) and os.path.exists(task["sol"])):
            print(f"Skipping {task['out']}: One or more input files not found.")
            continue
            
        try:
            create_vtk(task["node"], task["ele"], task["sol"], task["out"])
        except Exception as e:
            print(f"Failed to create {task['out']}. Error: {e}")
            print("-" * 20)

    print("Conversion process finished.")