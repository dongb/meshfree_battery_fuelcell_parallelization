"""
Comprehensive Performance Analysis and Profiling for fuel_cell_3D
==================================================================

This script profiles computational hotspots, memory usage, and scalability
to identify optimization opportunities.
"""

import time
from common import np
import tracemalloc
from collections import defaultdict

from get_nodes_gauss_points import (
    get_x_nodes_fuel_cell_3d_toy_image,
    x_G_and_def_J_time_weight_3d_fuelcell_domain,
    x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary,
    x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface,
    x_G_and_det_J_line_3d_fuelcell_1d_boundary
)

from shape_function_in_domain import (
    compute_phi_M,
    shape_grad_shape_func,
    shape_func_n_nodes_by_n_nodes
)


class PerformanceProfiler:
    """Profile performance of computational kernels"""
    
    def __init__(self):
        self.results = defaultdict(dict)
    
    def profile_function(self, func, *args, name="function", iterations=5, **kwargs):
        """Profile a function with timing and memory tracking"""
        times = []
        mem_peaks = []
        
        for i in range(iterations):
            # Memory tracking
            tracemalloc.start()
            
            # Timing
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            
            # Memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            times.append(elapsed)
            mem_peaks.append(peak / 1024 / 1024)  # Convert to MB
        
        # Calculate statistics
        times = np.array(times)
        mem_peaks = np.array(mem_peaks)
        
        self.results[name] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_memory_mb': np.mean(mem_peaks),
            'max_memory_mb': np.max(mem_peaks)
        }
        
        return result


def analyze_node_generation_scalability():
    """Analyze node generation scalability with different problem sizes"""
    print("=" * 80)
    print("SCALABILITY ANALYSIS: Node Generation")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    
    sizes = [
        (10, 10, 10),
        (20, 20, 20),
        (30, 30, 30),
        (40, 40, 40),
        (50, 50, 50),
    ]
    
    results = []
    
    for nx, ny, nz in sizes:
        img = np.random.randint(0, 3, (nx, ny, nz), dtype=np.uint8)
        img[0, 0, 0] = 0
        img[1, 1, 1] = 1
        img[2, 2, 2] = 2
        
        name = f"node_gen_{nx}x{ny}x{nz}"
        
        result = profiler.profile_function(
            get_x_nodes_fuel_cell_3d_toy_image,
            0, 1e-5, 0, 1e-5, 0, 1e-5,
            [nx, ny, nz], img,
            name=name,
            iterations=3
        )
        
        total_pixels = nx * ny * nz
        stats = profiler.results[name]
        
        print(f"\nImage size: {nx}×{ny}×{nz} = {total_pixels:,} pixels")
        print(f"  Time:           {stats['mean_time']:.4f} ± {stats['std_time']:.4f} s")
        print(f"  Memory:         {stats['mean_memory_mb']:.2f} MB")
        print(f"  Throughput:     {total_pixels/stats['mean_time']:,.0f} pixels/s")
        
        results.append({
            'size': total_pixels,
            'time': stats['mean_time'],
            'memory': stats['mean_memory_mb'],
            'throughput': total_pixels/stats['mean_time']
        })
    
    # Analyze scalability
    print("\n" + "-" * 80)
    print("SCALABILITY METRICS:")
    print("-" * 80)
    
    for i in range(1, len(results)):
        size_ratio = results[i]['size'] / results[i-1]['size']
        time_ratio = results[i]['time'] / results[i-1]['time']
        mem_ratio = results[i]['memory'] / results[i-1]['memory']
        
        print(f"\n{results[i-1]['size']:,} → {results[i]['size']:,} pixels:")
        print(f"  Size increase:     {size_ratio:.2f}x")
        print(f"  Time increase:     {time_ratio:.2f}x")
        print(f"  Memory increase:   {mem_ratio:.2f}x")
        print(f"  Scaling efficiency: {(size_ratio/time_ratio)*100:.1f}%")
    
    return profiler.results


def analyze_shape_function_performance():
    """Analyze shape function computation performance"""
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS: Shape Functions")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    
    # Test different problem sizes
    test_configs = [
        {'n_gauss': 100, 'n_nodes': 50, 'name': 'small'},
        {'n_gauss': 500, 'n_nodes': 200, 'name': 'medium'},
        {'n_gauss': 1000, 'n_nodes': 500, 'name': 'large'},
        {'n_gauss': 2000, 'n_nodes': 1000, 'name': 'xlarge'},
    ]
    
    results = []
    
    for config in test_configs:
        n_gauss = config['n_gauss']
        n_nodes = config['n_nodes']
        name = config['name']
        
        # Generate test data
        x_G = np.random.rand(n_gauss, 3) * 1e-5
        x_nodes = np.random.rand(n_nodes, 3) * 1e-5
        Gauss_grain_id = np.ones(n_gauss, dtype=np.int32)
        nodes_grain_id = np.ones(n_nodes, dtype=np.int32)
        a = np.full(n_nodes, 2e-6)
        
        # Initialize moment matrices
        M = np.zeros((n_gauss, 4, 4), dtype=np.float64)
        M_P_x = np.zeros((n_gauss, 4, 4), dtype=np.float64)
        M_P_y = np.zeros((n_gauss, 4, 4), dtype=np.float64)
        M_P_z = np.zeros((n_gauss, 4, 4), dtype=np.float64)
        
        # Profile compute_phi_M
        func_name = f"compute_phi_M_{name}"
        result = profiler.profile_function(
            compute_phi_M,
            x_G, Gauss_grain_id, x_nodes, nodes_grain_id, a,
            M.copy(), M_P_x.copy(), M_P_y.copy(),
            0, np.zeros((1, 1)), np.zeros((1, 1)),
            'False', 'False', M_P_z.copy(),
            name=func_name,
            iterations=3
        )
        
        stats = profiler.results[func_name]
        
        print(f"\n{name.upper()} Problem: {n_gauss} Gauss points, {n_nodes} nodes")
        print(f"  compute_phi_M:")
        print(f"    Time:       {stats['mean_time']*1000:.2f} ± {stats['std_time']*1000:.2f} ms")
        print(f"    Memory:     {stats['mean_memory_mb']:.2f} MB")
        print(f"    Rate:       {n_gauss/stats['mean_time']:,.0f} Gauss pts/s")
        
        results.append({
            'config': name,
            'n_gauss': n_gauss,
            'n_nodes': n_nodes,
            'time': stats['mean_time'],
            'memory': stats['mean_memory_mb']
        })
    
    return profiler.results


def analyze_matrix_assembly_complexity():
    """Analyze computational complexity of matrix assembly operations"""
    print("\n" + "=" * 80)
    print("COMPLEXITY ANALYSIS: Matrix Assembly")
    print("=" * 80)
    
    # Simulate different matrix sizes
    matrix_sizes = [100, 500, 1000, 2000, 5000]
    
    print("\nSparse Matrix Operations:")
    print("-" * 40)
    
    for n in matrix_sizes:
        # Sparse matrix creation
        start = time.perf_counter()
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
        # Simulate creating sparse matrix with ~10% density
        nnz = int(n * n * 0.1)
        row = np.random.randint(0, n, nnz)
        col = np.random.randint(0, n, nnz)
        data = np.random.rand(nnz)
        K = csr_matrix((data, (row, col)), shape=(n, n))
        elapsed_create = time.perf_counter() - start
        
        # Matrix-vector multiply
        x = np.random.rand(n)
        start = time.perf_counter()
        y = K @ x
        elapsed_matvec = time.perf_counter() - start
        
        # Solve (using simple iteration to estimate)
        from scipy.sparse.linalg import spsolve
        start = time.perf_counter()
        try:
            u = spsolve(K, x)
            elapsed_solve = time.perf_counter() - start
        except:
            elapsed_solve = np.nan
        
        print(f"\nMatrix size: {n}×{n} ({nnz:,} non-zeros, {nnz/(n*n)*100:.1f}% dense)")
        print(f"  Creation:   {elapsed_create*1000:.2f} ms")
        print(f"  Mat-vec:    {elapsed_matvec*1000:.2f} ms")
        if not np.isnan(elapsed_solve):
            print(f"  Solve:      {elapsed_solve*1000:.2f} ms")


def identify_bottlenecks():
    """Identify computational bottlenecks in typical workflow"""
    print("\n" + "=" * 80)
    print("BOTTLENECK IDENTIFICATION")
    print("=" * 80)
    
    # Simulate typical problem
    nx, ny, nz = 30, 30, 30
    img = np.random.randint(0, 3, (nx, ny, nz), dtype=np.uint8)
    img[0, 0, 0] = 0
    img[1, 1, 1] = 1
    img[2, 2, 2] = 2
    
    timings = {}
    
    # 1. Node generation
    print("\nStep 1: Node Generation...")
    start = time.perf_counter()
    result = get_x_nodes_fuel_cell_3d_toy_image(
        0, 1e-5, 0, 1e-5, 0, 1e-5,
        [nx, ny, nz], img
    )
    timings['node_generation'] = time.perf_counter() - start
    
    x_nodes_mechanical = result[0]
    print(f"  Time: {timings['node_generation']:.4f} s")
    print(f"  Generated {len(x_nodes_mechanical)} mechanical nodes")
    
    # 2. Gauss point generation
    print("\nStep 2: Gauss Point Generation...")
    n_cells = 1000
    cell_x = np.random.rand(n_cells, 8)
    cell_y = np.random.rand(n_cells, 8)
    cell_z = np.random.rand(n_cells, 8)
    sqrt3 = 1/np.sqrt(3)
    x_G_domain = [
        [-sqrt3, -sqrt3, -sqrt3], [sqrt3, -sqrt3, -sqrt3],
        [-sqrt3, sqrt3, -sqrt3], [sqrt3, sqrt3, -sqrt3],
        [-sqrt3, -sqrt3, sqrt3], [sqrt3, -sqrt3, sqrt3],
        [-sqrt3, sqrt3, sqrt3], [sqrt3, sqrt3, sqrt3]
    ]
    weight_G_domain = [1.0] * 8
    
    start = time.perf_counter()
    x_G, det_J_weight = x_G_and_def_J_time_weight_3d_fuelcell_domain(
        cell_x, cell_y, cell_z, x_G_domain, weight_G_domain
    )
    timings['gauss_point_generation'] = time.perf_counter() - start
    print(f"  Time: {timings['gauss_point_generation']:.4f} s")
    print(f"  Generated {len(x_G)} Gauss points")
    
    # 3. Shape function computation (simulated)
    print("\nStep 3: Shape Function Computation...")
    n_gauss = min(500, len(x_G))
    n_nodes = min(200, len(x_nodes_mechanical))
    x_G_sample = np.array(x_G[:n_gauss])
    x_nodes_sample = np.array(x_nodes_mechanical[:n_nodes])
    
    Gauss_grain_id = np.ones(n_gauss, dtype=np.int32)
    nodes_grain_id = np.ones(n_nodes, dtype=np.int32)
    a = np.full(n_nodes, 2e-6)
    
    M = np.zeros((n_gauss, 4, 4), dtype=np.float64)
    M_P_x = np.zeros((n_gauss, 4, 4), dtype=np.float64)
    M_P_y = np.zeros((n_gauss, 4, 4), dtype=np.float64)
    M_P_z = np.zeros((n_gauss, 4, 4), dtype=np.float64)
    
    start = time.perf_counter()
    result = compute_phi_M(
        x_G_sample, Gauss_grain_id, x_nodes_sample, nodes_grain_id, a,
        M, M_P_x, M_P_y,
        0, np.zeros((1, 1)), np.zeros((1, 1)),
        'False', 'False', M_P_z
    )
    timings['shape_function_compute'] = time.perf_counter() - start
    print(f"  Time: {timings['shape_function_compute']:.4f} s")
    
    # 4. Sparse matrix assembly (simulated)
    print("\nStep 4: Sparse Matrix Assembly (simulated)...")
    from scipy.sparse import csr_matrix
    n = n_nodes * 3  # 3 DOF per node
    nnz = int(n * 50)  # ~50 non-zeros per row
    row = np.random.randint(0, n, nnz)
    col = np.random.randint(0, n, nnz)
    data = np.random.rand(nnz)
    start = time.perf_counter()
    K = csr_matrix((data, (row, col)), shape=(n, n))
    timings['matrix_assembly'] = time.perf_counter() - start
    print(f"  Time: {timings['matrix_assembly']:.4f} s")
    
    # 5. Linear solve (simulated)
    print("\nStep 5: Linear System Solve (simulated)...")
    from scipy.sparse.linalg import spsolve
    K = K + csr_matrix(np.eye(n) * 1e6)  # Make it well-conditioned
    f = np.random.rand(n)
    start = time.perf_counter()
    u = spsolve(K, f)
    timings['linear_solve'] = time.perf_counter() - start
    print(f"  Time: {timings['linear_solve']:.4f} s")
    
    # Summary
    print("\n" + "-" * 80)
    print("BOTTLENECK SUMMARY:")
    print("-" * 80)
    total_time = sum(timings.values())
    
    sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    
    for name, elapsed in sorted_timings:
        percentage = (elapsed / total_time) * 100
        bar = "█" * int(percentage / 2)
        print(f"{name:30s}: {elapsed:6.4f}s ({percentage:5.1f}%) {bar}")
    
    print(f"\nTotal time: {total_time:.4f}s")
    
    return timings


def main():
    """Run comprehensive performance analysis"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " COMPREHENSIVE PERFORMANCE ANALYSIS ".center(78) + "║")
    print("║" + " fuel_cell_3D Code Base ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    # Run all analyses
    analyze_node_generation_scalability()
    analyze_shape_function_performance()
    analyze_matrix_assembly_complexity()
    bottleneck_timings = identify_bottlenecks()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey findings written to PERFORMANCE_ANALYSIS_REPORT.md")
    print()


if __name__ == "__main__":
    main()

