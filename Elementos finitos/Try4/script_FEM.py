#!/usr/bin/env python3
"""
FEM to VTK Converter for 3D Visualization in ParaView
Converts .node, .ele, and .txt solution files to .vtu format
Handles 3D tetrahedral meshes with complex-valued vector field solutions
Separates domains by region ID for improved ParaView visualization
"""

import numpy as np
import meshio
import os
import sys
from pathlib import Path
from collections import defaultdict


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


def extract_nodes_for_elements(connectivity, n_total_nodes):
    """
    Extract unique nodes used by a set of elements.
    
    Args:
        connectivity: Element connectivity array (n_elements, 4)
        n_total_nodes: Total number of nodes in mesh
    
    Returns:
        node_mask: Boolean array indicating which nodes are used
        node_map: Mapping from old node indices to new (compacted) indices
    """
    unique_nodes = np.unique(connectivity.flatten())
    node_mask = np.zeros(n_total_nodes, dtype=bool)
    node_mask[unique_nodes] = True
    
    # Create mapping from old to new node indices
    node_map = np.full(n_total_nodes, -1, dtype=np.int32)
    node_map[unique_nodes] = np.arange(len(unique_nodes))
    
    return node_mask, node_map


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
    print(f"  Creating VTU: {output_path}")
    
    # Create cells dictionary for meshio (tetrahedral elements)
    cells = [("tetra", connectivity)]
    
    # Prepare cell data (per-element data)
    cell_data = {
        "RegionID": [region_ids]
    }
    
    # Prepare point data (per-node data)
    point_data = {}
    if solution_vector is not None:
        # Calculate derived fields from complex solution
        real_part = solution_vector[:, 0]
        imag_part = solution_vector[:, 1]
        
        # Magnitude: |z| = sqrt(real² + imag²)
        magnitude = np.sqrt(real_part**2 + imag_part**2)
        
        # Phase: arg(z) = atan2(imag, real)
        phase = np.arctan2(imag_part, real_part)
        
        # Add all fields to point data
        point_data["SolutionVector"] = solution_vector  # Full 3D vector [Real, Imag, 0]
        point_data["Magnitude"] = magnitude              # Scalar: field intensity
        point_data["RealPart"] = real_part              # Scalar: real component
        point_data["ImaginaryPart"] = imag_part         # Scalar: imaginary component
        point_data["Phase"] = phase                      # Scalar: phase angle in radians
        
        print(f"    Added 5 solution fields: SolutionVector, Magnitude, RealPart, ImaginaryPart, Phase")
    
    # Create mesh
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data=cell_data,
        point_data=point_data
    )
    
    # Write to VTU file
    meshio.write(output_path, mesh, file_format="vtu")
    print(f"    Elements: {len(connectivity)}, Nodes: {len(points)}")


def separate_domains_by_region(points, connectivity, region_ids, solution_vector, output_base, region_names):
    """
    Separate mesh into multiple VTU files based on region IDs.
    
    Args:
        points: Node coordinates (n_nodes, 3)
        connectivity: Element connectivity (n_elements, 4)
        region_ids: Region attributes per element (n_elements,)
        solution_vector: Solution as vector field (n_nodes, 3) or None
        output_base: Base name for output files
        region_names: Dictionary mapping region IDs to descriptive names
    
    Returns:
        Number of files created
    """
    files_created = 0
    
    # Group elements by region
    for region_id, region_name in region_names.items():
        # Find elements belonging to this region
        region_mask = region_ids == region_id
        
        if not np.any(region_mask):
            continue
        
        # Extract connectivity for this region
        region_connectivity = connectivity[region_mask]
        region_region_ids = region_ids[region_mask]
        
        # Extract only nodes used by this region
        node_mask, node_map = extract_nodes_for_elements(region_connectivity, len(points))
        region_points = points[node_mask]
        
        # Remap connectivity to new node indices
        remapped_connectivity = node_map[region_connectivity]
        
        # Extract solution for used nodes
        region_solution = None
        if solution_vector is not None:
            region_solution = solution_vector[node_mask]
        
        # Create output filename
        output_file = f"{output_base}_{region_name}.vtu"
        
        # Create VTU file
        create_vtu_file(region_points, remapped_connectivity, region_region_ids, 
                       region_solution, output_file)
        files_created += 1
    
    return files_created


def process_case(node_file, ele_file, sol_file, output_base, region_names):
    """
    Process a single case: read input files and generate VTU output(s).
    
    Args:
        node_file: Path to .node file
        ele_file: Path to .ele file
        sol_file: Path to .txt solution file (or None for geometry-only)
        output_base: Base name for output .vtu file(s) (without extension)
        region_names: Dictionary mapping region IDs to descriptive names
    """
    print("\n" + "=" * 70)
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
        
        # Analyze region distribution
        unique_regions, counts = np.unique(region_ids, return_counts=True)
        print(f"\nRegion distribution:")
        for region_id, count in sorted(zip(unique_regions, counts), key=lambda x: x[1], reverse=True)[:10]:
            region_name = region_names.get(region_id, f"Unknown_{region_id}")
            print(f"  Region {region_id} ({region_name}): {count} elements")
        
        # Read solution if file is provided
        solution_vectors = None
        if sol_file is not None:
            if os.path.exists(sol_file):
                solution_vectors = read_solution_file(sol_file, len(points))
            else:
                print(f"Warning: Solution file {sol_file} not found. Creating geometry-only VTU.\n")
        
        # Create VTU file(s) separated by domain
        files_created = 0
        print(f"\nGenerating domain-separated VTU files:")
        
        if solution_vectors is not None and len(solution_vectors) > 1:
            # Multiple frequency points - create separate files for each frequency
            for freq_idx, solution_vector in enumerate(solution_vectors):
                print(f"\n  Frequency {freq_idx + 1}:")
                output_freq_base = f"{output_base}_freq{freq_idx + 1}"
                n_files = separate_domains_by_region(
                    points, connectivity, region_ids, solution_vector, 
                    output_freq_base, region_names
                )
                files_created += n_files
        else:
            # Single frequency or geometry only
            solution_vector = solution_vectors[0] if solution_vectors else None
            n_files = separate_domains_by_region(
                points, connectivity, region_ids, solution_vector, 
                output_base, region_names
            )
            files_created += n_files
        
        print(f"\n  Total files created: {files_created}")
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
    print("FEM to VTK Converter - 3D Tetrahedral Meshes with Domain Separation")
    print("=" * 70 + "\n")
    
    # Get script's directory (where all input files are located)
    work_dir = Path(__file__).parent.resolve()
    print(f"Working directory: {work_dir}\n")
    
    # Change to script directory to ensure relative paths work
    os.chdir(work_dir)
    
    # Define region names for Benchmark cases (Coil + Plate)
    # Based on analysis: Region 1=Air, 2=Coil, 3=?, 4=?, 5=Plate
    benchmark_regions = {
        1: "Air",
        2: "Coil", 
        3: "Domain3",
        4: "Domain4",
        5: "Plate"
    }
    
    # Define region names for L-Domain cases
    # These have many small regions - we'll focus on the largest ones
    # You can customize this based on your specific needs
    ldomain_regions = {
        1: "LDomain_Main",
        2: "LDomain_Region2",
        3: "LDomain_Region3"
    }
    
    # Define all cases to process
    cases = [
        {
            "name": "Benchmark 1",
            "node": "ExBenchmark.1.node",
            "ele": "ExBenchmark.1.ele",
            "sol": "solucionAVpsi.1.txt",
            "output": "Benchmark1_3D",
            "regions": benchmark_regions
        },
        {
            "name": "Benchmark 5",
            "node": "ExBenchmark.5.node",
            "ele": "ExBenchmark.5.ele",
            "sol": "solucionAVpsi.5.txt",
            "output": "Benchmark5_3D",
            "regions": benchmark_regions
        },
        {
            "name": "L-Domain 1",
            "node": "Ldomain.1.node",
            "ele": "Ldomain.1.ele",
            "sol": "LsolucionAVpsi.1.txt",
            "output": "Ldomain1_3D",
            "regions": ldomain_regions
        },
        {
            "name": "L-Domain 10",
            "node": "Ldomain.10.node",
            "ele": "Ldomain.10.ele",
            "sol": "LsolucionAVpsi.10.txt",
            "output": "Ldomain10_3D",
            "regions": ldomain_regions
        },
        {
            "name": "L-Domain 22",
            "node": "Ldomain.22.node",
            "ele": "Ldomain.22.ele",
            "sol": "LsolucionAVpsi.22.txt",
            "output": "Ldomain22_3D",
            "regions": ldomain_regions
        }
    ]
    
    # Process each case
    files_created = 0
    for case in cases:
        n_files = process_case(
            node_file=case["node"],
            ele_file=case["ele"],
            sol_file=case["sol"],
            output_base=case["output"],
            region_names=case["regions"]
        )
        files_created += n_files
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Conversion complete: {files_created} VTU file(s) created successfully")
    print("=" * 70 + "\n")
    
    if files_created > 0:
        print("Generated separate VTU files for each domain/region.")
        print("\nParaView Visualization Tips:")
        print("  1. Load all domain files together to see the complete system")
        print("  2. For Benchmark cases:")
        print("     - *_Coil.vtu: The coil domain")
        print("     - *_Plate.vtu: The plate domain")
        print("     - *_Air.vtu: The surrounding air/medium")
        print("  3. For L-Domain cases:")
        print("     - *_LDomain_*.vtu: Different regions of the L-shaped domain")
        print("  4. Use 'SolutionVector' to visualize the field")
        print("  5. Use 'Magnitude' to see the field strength")
        print("  6. Apply 'Glyph' filter to show vector directions")
        print("  7. Apply 'Warp by Vector' to visualize deformations")
        print("  8. Compare different frequencies by loading *_freq1, *_freq2, etc.")
        print("\nEach domain is now a separate file for cleaner visualization!")
    

if __name__ == "__main__":
    main()
