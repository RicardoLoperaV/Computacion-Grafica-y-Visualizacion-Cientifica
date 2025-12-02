#!/usr/bin/env python3
"""
FEM to VTK Converter for 3D Visualization in ParaView
Converts .node, .ele, and .txt solution files to .vtu format
Handles 3D tetrahedral meshes with complex-valued vector field solutions
"""

import numpy as np
import meshio
import os
import sys
from pathlib import Path


def read_node_file(node_path):
    """
    Read .node file containing 3D point coordinates.
    
    Format:
        Header: <#nodes> <#dim> <#attr> <#bound>
        Data:   <Index> <X> <Y> <Z> <Attribute>
    
    Returns:
        points: numpy array of shape (n_nodes, 3) with [X, Y, Z] coordinates
    """
    print(f"Reading node file: {node_path}")
    
    with open(node_path, 'r') as f:
        # Read header
        header = f.readline().strip().split()
        n_nodes = int(header[0])
        n_dim = int(header[1])
        n_attr = int(header[2])
        n_bound = int(header[3])
        
        print(f"  Nodes: {n_nodes}, Dimensions: {n_dim}, Attributes: {n_attr}, Boundary: {n_bound}")
        
        if n_dim != 3:
            raise ValueError(f"Expected 3D data (dim=3), but got dim={n_dim}")
        
        # Read node coordinates
        points = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            # Index, X, Y, Z, [Attribute]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            points.append([x, y, z])
        
        points = np.array(points)
        print(f"  Read {len(points)} points successfully")
        
        if len(points) != n_nodes:
            print(f"  Warning: Expected {n_nodes} nodes but read {len(points)}")
        
        return points


def read_ele_file(ele_path):
    """
    Read .ele file containing tetrahedral element connectivity.
    
    Format:
        Header: <#elements> <#nodes_per_element> <#attr>
        Data:   <Index> <Node1> <Node2> <Node3> <Node4> <RegionAttribute>
    
    Returns:
        connectivity: numpy array of shape (n_elements, 4) with 0-based node indices
        region_ids: numpy array of shape (n_elements,) with region attributes
    """
    print(f"Reading element file: {ele_path}")
    
    with open(ele_path, 'r') as f:
        # Read header
        header = f.readline().strip().split()
        n_elements = int(header[0])
        nodes_per_elem = int(header[1])
        n_attr = int(header[2])
        
        print(f"  Elements: {n_elements}, Nodes per element: {nodes_per_elem}, Attributes: {n_attr}")
        
        if nodes_per_elem != 4:
            raise ValueError(f"Expected tetrahedral elements (4 nodes), but got {nodes_per_elem} nodes")
        
        # Read element connectivity
        connectivity = []
        region_ids = []
        
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            # Index, Node1, Node2, Node3, Node4, [RegionAttribute]
            # Convert from 1-based to 0-based indexing
            nodes = [int(parts[i]) - 1 for i in range(1, 5)]
            connectivity.append(nodes)
            
            # Store region attribute if present
            if len(parts) > 5:
                region_ids.append(int(parts[5]))
            else:
                region_ids.append(0)
        
        connectivity = np.array(connectivity, dtype=np.int32)
        region_ids = np.array(region_ids, dtype=np.int32)
        
        print(f"  Read {len(connectivity)} tetrahedral elements successfully")
        
        if len(connectivity) != n_elements:
            print(f"  Warning: Expected {n_elements} elements but read {len(connectivity)}")
        
        return connectivity, region_ids


def read_solution_file(sol_path, n_expected_nodes):
    """
    Read .txt file containing complex-valued solution (one per node).
    
    Format:
        First line: number of nodes
        Following lines: complex numbers like "-0.0000164552 -0.0000324275i"
        File may contain multiple frequency solutions (5 blocks of n_nodes values)
    
    Returns:
        list of solution_vectors: List of numpy arrays, each of shape (n_nodes, 3) 
                                  where each row is [Real, Imag, 0.0]
    """
    print(f"Reading solution file: {sol_path}")
    
    solution_complex = []
    
    with open(sol_path, 'r') as f:
        lines = f.readlines()
        
        # First line is the count
        if len(lines) > 0:
            first_line = lines[0].strip()
            try:
                n_solutions = int(first_line)
                print(f"  Expected {n_solutions} solution values from header")
                start_idx = 1
            except ValueError:
                # No header, start from first line
                start_idx = 0
        else:
            start_idx = 0
        
        # Parse complex numbers from remaining lines
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse complex number with format: "real +/-imag i"
            # Remove spaces and replace 'i' with 'j' for Python's complex()
            try:
                # Remove all spaces and replace 'i' with 'j'
                clean_line = line.replace(' ', '').replace('i', 'j')
                value = complex(clean_line)
                solution_complex.append(value)
            except ValueError:
                print(f"  Warning: Could not parse line: {line}")
                solution_complex.append(0.0 + 0.0j)
    
    print(f"  Read {len(solution_complex)} solution values")
    
    # Determine number of frequency points
    n_frequencies = len(solution_complex) // n_expected_nodes
    
    if len(solution_complex) % n_expected_nodes != 0:
        print(f"  Warning: Total values ({len(solution_complex)}) not evenly divisible by nodes ({n_expected_nodes})")
        n_frequencies = len(solution_complex) // n_expected_nodes
    
    print(f"  Detected {n_frequencies} frequency point(s)")
    
    # Split into frequency blocks and convert each to vector field
    solution_vectors = []
    for freq_idx in range(n_frequencies):
        start = freq_idx * n_expected_nodes
        end = start + n_expected_nodes
        freq_data = solution_complex[start:end]
        
        # Convert to vector field: [Real, Imaginary, 0.0]
        solution_vector = np.zeros((len(freq_data), 3), dtype=np.float64)
        for i, val in enumerate(freq_data):
            solution_vector[i, 0] = val.real
            solution_vector[i, 1] = val.imag
            solution_vector[i, 2] = 0.0
        
        solution_vectors.append(solution_vector)
        print(f"    Frequency {freq_idx + 1}: Converted to 3D vector field (shape: {solution_vector.shape})")
    
    return solution_vectors


def create_vtu_file(points, connectivity, region_ids, solution_vector, output_path):
    """
    Create VTU file using meshio for ParaView visualization.
    
    Args:
        points: Node coordinates (n_nodes, 3)
        connectivity: Element connectivity (n_elements, 4)
        region_ids: Region attributes per element (n_elements,)
        solution_vector: Solution as vector field (n_nodes, 3) or None
        output_path: Output .vtu file path
    """
    print(f"Creating VTU file: {output_path}")
    
    # Create cells dictionary for meshio (tetrahedral elements)
    cells = [("tetra", connectivity)]
    
    # Prepare cell data (per-element data)
    cell_data = {
        "RegionID": [region_ids]
    }
    
    # Prepare point data (per-node data)
    point_data = {}
    if solution_vector is not None:
        point_data["SolutionVector"] = solution_vector
        print(f"  Added SolutionVector field to point data")
    
    # Create mesh
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data=cell_data,
        point_data=point_data
    )
    
    # Write to VTU file
    meshio.write(output_path, mesh, file_format="vtu")
    print(f"  Successfully written to {output_path}")
    print()


def process_case(node_file, ele_file, sol_file, output_base):
    """
    Process a single case: read input files and generate VTU output(s).
    
    Args:
        node_file: Path to .node file
        ele_file: Path to .ele file
        sol_file: Path to .txt solution file (or None for geometry-only)
        output_base: Base name for output .vtu file(s) (without extension)
    """
    print("=" * 70)
    print(f"Processing: {output_base}")
    print("=" * 70)
    
    try:
        # Check if required files exist
        if not os.path.exists(node_file):
            print(f"Skipping: {node_file} not found\n")
            return 0
        
        if not os.path.exists(ele_file):
            print(f"Skipping: {ele_file} not found\n")
            return 0
        
        # Read geometry
        points = read_node_file(node_file)
        connectivity, region_ids = read_ele_file(ele_file)
        
        # Read solution if file is provided
        solution_vectors = None
        if sol_file is not None:
            if os.path.exists(sol_file):
                solution_vectors = read_solution_file(sol_file, len(points))
            else:
                print(f"Warning: Solution file {sol_file} not found. Creating geometry-only VTU.\n")
        
        # Create VTU file(s)
        files_created = 0
        if solution_vectors is not None and len(solution_vectors) > 1:
            # Multiple frequency points - create separate files
            for freq_idx, solution_vector in enumerate(solution_vectors):
                output_file = f"{output_base}_freq{freq_idx + 1}.vtu"
                create_vtu_file(points, connectivity, region_ids, solution_vector, output_file)
                files_created += 1
        else:
            # Single frequency or geometry only
            solution_vector = solution_vectors[0] if solution_vectors else None
            output_file = f"{output_base}.vtu"
            create_vtu_file(points, connectivity, region_ids, solution_vector, output_file)
            files_created += 1
        
        return files_created
    
    except Exception as e:
        print(f"ERROR processing {output_base}: {e}")
        import traceback
        traceback.print_exc()
        print()
        return 0


def main():
    """
    Main execution: Process all specified file combinations.
    """
    print("\n" + "=" * 70)
    print("FEM to VTK Converter - 3D Tetrahedral Meshes")
    print("=" * 70 + "\n")
    
    # Get script's directory (where all input files are located)
    work_dir = Path(__file__).parent.resolve()
    print(f"Working directory: {work_dir}\n")
    
    # Change to script directory to ensure relative paths work
    os.chdir(work_dir)
    
    # Define all cases to process
    cases = [
        {
            "name": "Benchmark 1",
            "node": "ExBenchmark.1.node",
            "ele": "ExBenchmark.1.ele",
            "sol": "solucionAVpsi.1.txt",
            "output": "Benchmark1_3D_Sol"
        },
        {
            "name": "Benchmark 5",
            "node": "ExBenchmark.5.node",
            "ele": "ExBenchmark.5.ele",
            "sol": "solucionAVpsi.5.txt",
            "output": "Benchmark5_3D_Sol"
        },
        {
            "name": "L-Domain 1",
            "node": "Ldomain.1.node",
            "ele": "Ldomain.1.ele",
            "sol": "LsolucionAVpsi.1.txt",
            "output": "Ldomain1_3D_Sol"
        },
        {
            "name": "L-Domain 10",
            "node": "Ldomain.10.node",
            "ele": "Ldomain.10.ele",
            "sol": "LsolucionAVpsi.10.txt",
            "output": "Ldomain10_3D_Sol"
        },
        {
            "name": "L-Domain 22",
            "node": "Ldomain.22.node",
            "ele": "Ldomain.22.ele",
            "sol": "LsolucionAVpsi.22.txt",
            "output": "Ldomain22_3D_Sol"
        }
    ]
    
    # Process each case
    files_created = 0
    for case in cases:
        n_files = process_case(
            node_file=case["node"],
            ele_file=case["ele"],
            sol_file=case["sol"],
            output_base=case["output"]
        )
        files_created += n_files
    
    # Summary
    print("=" * 70)
    print(f"Conversion complete: {files_created} VTU file(s) created successfully")
    print("=" * 70 + "\n")
    
    if files_created > 0:
        print("You can now open the .vtu files in ParaView.")
        print("Tips for ParaView:")
        print("  - Multiple frequency files were created (e.g., *_freq1.vtu, *_freq2.vtu, etc.)")
        print("  - Use 'Threshold' filter with 'RegionID' to split domains")
        print("  - Use 'Glyph' filter to visualize 'SolutionVector' field")
        print("  - Use 'Warp by Vector' with 'SolutionVector' for deformation")
        print("  - Compare different frequencies by loading multiple files")
    

if __name__ == "__main__":
    main()
