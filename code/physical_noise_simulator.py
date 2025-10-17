import numpy as np
from scipy.linalg import svd
import math

# --- 1. CORE UTILITY FUNCTIONS ---

def von_neumann_entropy(rho):
    """Calculates the Von Neumann Entropy S(rho) = -Tr(rho log2(rho))."""
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filter out eigenvalues close to zero to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    S = -np.sum(eigenvalues * np.log2(eigenvalues))
    return S

def calculate_R(rho, p_error_total):
    """Calculates the Robustness Metric R = (1 - gamma) * exp(-S(rho))."""
    S = von_neumann_entropy(rho)
    # Total error accumulation over the Bell circuit (H + CNOT + 2 decoherence windows)
    one_minus_gamma = 1.0 - p_error_total 
    R = one_minus_gamma * math.exp(-S)
    return R

def calculate_C_Bell_max(rho):
    """
    Calculates the maximum Bell correlation C_Bell^max, 
    normalized to the range [0, 1].
    """
    # Standard Pauli matrices
    sig_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sig_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sig_z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_ops = [sig_x, sig_y, sig_z]

    # Calculate the 3x3 correlation matrix T (T_ij = Tr(rho * sigma_i @ sigma_j))
    T = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            op = np.kron(pauli_ops[i], pauli_ops[j])
            T[i, j] = np.real(np.trace(rho @ op))

    # Calculate the singular values of T
    singular_values = svd(T, compute_uv=False)
    s1 = singular_values[0]
    s2 = singular_values[1]
    
    # Normalized Bell Correlation (C_Bell^max) is based on the largest two singular values
    # C_Bell^max_normalized = (2 * sqrt(s1^2 + s2^2)) / (2 * sqrt(2)) 
    C_Bell_max_normalized = np.sqrt(s1**2 + s2**2) / np.sqrt(2)

    return C_Bell_max_normalized


# --- 2. NOISE CHANNEL IMPLEMENTATION ---

def apply_channel(rho, kraus_operators):
    """Applies a generic Kraus channel E(rho) = sum_k E_k rho E_k^dagger."""
    rho_out = np.zeros_like(rho, dtype=complex)
    for E_k in kraus_operators:
        rho_out += E_k @ rho @ E_k.conj().T
    # Simple renormalization to ensure trace is 1.0
    rho_out /= np.trace(rho_out)
    return rho_out

def get_t1_channel(t, T1, num_qubits=1):
    """Returns Kraus operators for Amplitude Damping (T1 noise)."""
    if T1 == np.inf:
        return [np.eye(2**num_qubits, dtype=complex)]
    
    p_T1 = 1.0 - math.exp(-t / T1)
    E0 = np.array([[1, 0], [0, np.sqrt(1 - p_T1)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(p_T1)], [0, 0]], dtype=complex)
    return [E0, E1]

def get_t2_channel(t, T2, T1, num_qubits=1):
    """Returns Kraus operators for Phase Damping (effective T2 noise)."""
    if T2 == np.inf:
        return [np.eye(2**num_qubits, dtype=complex)]

    # Calculate T_phi (pure dephasing time) from T1 and T2
    if T1 == np.inf: 
        T_phi = T2
    else:
        # 1/T2 = 1/T_phi + 1/(2*T1)
        try:
            inv_T_phi = 1/T2 - 1/(2*T1)
            T_phi = 1 / inv_T_phi
        except (ZeroDivisionError, OverflowError):
            T_phi = np.inf
        
        # Handle cases where T2 > 2*T1 due to measurement or estimation errors
        if inv_T_phi < 1e-9:
             T_phi = np.inf
    
    if T_phi == np.inf:
        p_T2 = 0.0
    else:
        p_T2 = 1.0 - math.exp(-t / T_phi)

    # Effective T2 channel (Phase Damping component)
    E0 = np.array([[1, 0], [0, np.sqrt(1 - p_T2)]], dtype=complex)
    E1 = np.array([[0, 0], [0, np.sqrt(p_T2)]], dtype=complex)
    return [E0, E1]

# --- 3. SIMULATION CORE ---

def simulate_bell_circuit(T1, T2, tau_gate, p_gate_error_CNOT):
    """Simulates a noisy Bell State preparation circuit |Phi+> = CNOT(H |00>)."""
    
    # Initial state |00><00|
    rho = np.zeros((4, 4), dtype=complex); rho[0, 0] = 1.0
    
    # Gates
    H_1 = np.kron(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), np.eye(2))
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    
    # ----------------------------------------------------
    # Step 0: Prepare Kraus operators for single-qubit T1/T2 noise application
    
    # Qubit 0 Noise
    E_T1_Q0 = get_t1_channel(tau_gate, T1, 1)
    E_T2_Q0 = get_t2_channel(tau_gate, T2, T1, 1)
    kraus_Q0 = [np.kron(E_T1_Q0[i] @ E_T2_Q0[j], np.eye(2)) 
                for i in range(len(E_T1_Q0)) for j in range(len(E_T2_Q0))]
    
    # Qubit 1 Noise
    E_T1_Q1 = get_t1_channel(tau_gate, T1, 1)
    E_T2_Q1 = get_t2_channel(tau_gate, T2, T1, 1)
    kraus_Q1 = [np.kron(np.eye(2), E_T1_Q1[i] @ E_T2_Q1[j])
                for i in range(len(E_T1_Q1)) for j in range(len(E_T2_Q1))]
    # ----------------------------------------------------

    # 1. Hadamard Gate (H_1 on Q0)
    rho = H_1 @ rho @ H_1.conj().T
    
    # 2. Decoherence Window 1 (After H) - Apply T1/T2 noise
    rho = apply_channel(rho, kraus_Q0)
    rho = apply_channel(rho, kraus_Q1)

    # 3. CNOT Gate
    rho = CNOT @ rho @ CNOT.conj().T
    # Apply generalized gate depolarization error (p_gate_error_CNOT)
    p_depolarization = p_gate_error_CNOT
    rho = (1.0 - p_depolarization) * rho + p_depolarization * (np.eye(4) / 4.0)

    # 4. Decoherence Window 2 (After CNOT) - Apply T1/T2 noise
    rho = apply_channel(rho, kraus_Q0)
    rho = apply_channel(rho, kraus_Q1)
    
    # For R calculation, gamma is approximated by the accumulated gate error
    total_p_error = p_gate_error_CNOT 
    
    return rho, total_p_error


# --- 4. COMPLETE DATA STRUCTURE AND MAIN EXECUTION ---

# List of hardware systems to simulate. T1, T2, and tau_gate are in seconds.
# CNOT Error is a fraction (e.g., 0.005 for 0.5%).
SYSTEMS_DATA = [
    # T1 (s), T2 (s), tau_gate (s), CNOT Error (frac), System Name
    
    # --- 1-7: Benchmark Systems ---
    (300.0e-6, 200.0e-6, 0.25e-6, 0.005, "System 1: High-End Superconducting Qubit"),
    (100.0e-6, 75.0e-6, 0.50e-6, 0.015, "System 2: Mid-Range Superconducting Qubit"),
    ( 50.0e-6, 40.0e-6, 1.00e-6, 0.030, "System 3: Low-End SC Qubit (Pre-Advantage)"),
    (5000.0e-6, 3000.0e-6, 25.0e-6, 0.010, "System 4: Trapped Ion Qubit (Slow Gate)"),
    ( 20.0e-6, 15.0e-6, 0.05e-6, 0.050, "System 5: Short-Coherence Qubit (High Error)"),
    ( 1.0e-6, 1.0e-6, 1.0e-6, 0.010, "System 6: High Decoherence SC Qubit"),
    ( 1.0e-6, 1.0e-6, 1.0e-6, 0.010, "System 7: High Decoherence Photonic Qubit"),

    # --- 8-23: Diverse Architectures and Noise Profiles (Full Dataset) ---
    (500.0e-6, 400.0e-6, 0.40e-6, 0.008, "System 8: SC: Google Sycamore Equivalent"),
    (1000.0e-6, 800.0e-6, 10.0e-6, 0.002, "System 9: Neutral Atom: Rydberg Qubit"),
    (10000.0e-6, 6000.0e-6, 50.0e-6, 0.001, "System 10: Trapped Ion: High-Coherence Flagship"),
    (200.0e-6, 150.0e-6, 0.30e-6, 0.004, "System 11: SC: IBM Falcon Architecture"),
    ( 10.0e-6, 8.0e-6, 0.10e-6, 0.080, "System 12: Photonic: High Loss System (No Adv)"),
    (400.0e-6, 300.0e-6, 0.50e-6, 0.020, "System 13: Silicon Spin: Research Grade Qubit"),
    ( 80.0e-6, 60.0e-6, 1.50e-6, 0.040, "System 14: SC: Older Transmon System (No Adv)"),
    (250.0e-6, 200.0e-6, 0.20e-6, 0.002, "System 15: SC: Low-Noise Research System"),
    (  5.0e-6, 4.0e-6, 0.05e-6, 0.100, "System 16: SC: Very Noisy Testbed (No Adv)"),
    (150.0e-6, 120.0e-6, 0.80e-6, 0.010, "System 17: SC: System at the Advantage Boundary"),
    ( 70.0e-6, 50.0e-6, 0.90e-6, 0.018, "System 18: SC: System at the Pre-Advantage Boundary"),
    (6000.0e-6, 5000.0e-6, 40.0e-6, 0.005, "System 19: Trapped Ion: Mid-Coherence System"),
    (  1.0e-6, 0.8e-6, 0.01e-6, 0.150, "System 20: Photonic: Extreme High Loss (No Adv)"),
    (150.0e-6, 100.0e-6, 0.40e-6, 0.012, "System 21: SC: Early Flagship Qubit"),
    (200.0e-6, 150.0e-6, 0.40e-6, 0.025, "System 22: Silicon Spin: Highly Coupled Qubit (No Adv)"),
    ( 90.0e-6, 80.0e-6, 0.60e-6, 0.035, "System 23: SC: Highly Error-Prone Qubit (No Adv)"),
]

if __name__ == '__main__':
    
    print("-" * 75)
    print("| Quantum Noise Simulation: R vs. C_Bell^max (23 System Analysis) |")
    print("-" * 75)
    
    print(f"| {'System Name':<40} | {'T1 (us)':>8} | {'T2 (us)':>8} | {'R':>8} | {'C_Bell^max':>10} |")
    print("|" + "-" * 73 + "|")

    for T1_sim, T2_sim, tau_gate, p_CNOT_error, system_name in SYSTEMS_DATA:
        
        # 1. Run the simulation
        # Note: T1, T2, and tau_gate are in seconds
        rho_final, total_p_error = simulate_bell_circuit(T1_sim, T2_sim, tau_gate, p_CNOT_error)
        
        # 2. Calculate the metrics
        R_val = calculate_R(rho_final, total_p_error)
        C_Bell_val = calculate_C_Bell_max(rho_final)
        
        # 3. Print the results (T1/T2 are converted to microseconds for the table)
        print(f"| {system_name:<40} | {T1_sim*1e6:>8.1f} | {T2_sim*1e6:>8.1f} | {R_val:>8.4f} | {C_Bell_val:>10.4f} |")
        
    print("-" * 75)
    print("\n" + "=" * 75)
    print("Simulation Complete. The table above contains the R and C_Bell^max metrics for all 23 systems.")
    print("The final step is to analyze and plot this data.")
    print("=" * 75)
