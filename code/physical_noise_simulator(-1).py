import numpy as np
from scipy.linalg import svd
import math

# --- 1. CORE UTILITY FUNCTIONS (UNCHANGED) ---

def von_neumann_entropy(rho):
    """Calculates the Von Neumann Entropy S(rho) = -Tr(rho log2(rho))."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    S = -np.sum(eigenvalues * np.log2(eigenvalues))
    return S

def calculate_R(rho, p_error_total):
    """Calculates the Robustness Metric R = (1 - gamma) * exp(-S(rho))."""
    S = von_neumann_entropy(rho)
    one_minus_gamma = 1.0 - p_error_total
    R = one_minus_gamma * math.exp(-S)
    return R

def bell_correlation(rho):
    """
    Calculates the maximum Bell correlation C_Bell^max, 
    normalized to the range [0, 1].
    """
    sig_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sig_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sig_z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_ops = [sig_x, sig_y, sig_z]

    T = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            op = np.kron(pauli_ops[i], pauli_ops[j])
            T[i, j] = np.real(np.trace(rho @ op))

    singular_values = svd(T, compute_uv=False)
    s1 = singular_values[0]
    s2 = singular_values[1]
    
    S_value = 2 * np.sqrt(s1**2 + s2**2)
    
    # Normalized to the Tsirelson bound (2*sqrt(2))
    C_Bell_max_normalized = S_value / (2 * np.sqrt(2))

    return S_value, C_Bell_max_normalized

# --- 2. NOISE CHANNEL IMPLEMENTATION (UNCHANGED) ---

def apply_channel(rho, kraus_operators):
    """Applies a generic Kraus channel E(rho) = sum_k E_k rho E_k^dagger."""
    rho_out = np.zeros_like(rho, dtype=complex)
    for E_k in kraus_operators:
        rho_out += E_k @ rho @ E_k.conj().T
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

    if T1 == np.inf: 
        T_phi = T2
    else:
        if 1/T2 < 1/(2*T1): 
             T_phi = np.inf
        else:
            T_phi = 1 / (1/T2 - 1/(2*T1))
    
    if T_phi == np.inf:
        p_T2 = 0.0
    else:
        p_T2 = 1.0 - math.exp(-t / T_phi)

    E0 = np.array([[1, 0], [0, np.sqrt(1 - p_T2)]], dtype=complex)
    E1 = np.array([[0, 0], [0, np.sqrt(p_T2)]], dtype=complex)
    return [E0, E1]

# --- 3. SIMULATION CORE (UNCHANGED) ---

def simulate_bell_circuit(T1, T2, tau_gate, p_gate_error_CNOT):
    """Simulates a noisy Bell State preparation circuit."""
    rho = np.zeros((4, 4), dtype=complex); rho[0, 0] = 1.0
    
    H_1 = np.kron(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), np.eye(2))
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    
    # Prepare Kraus operators for sequential T1/T2 noise application
    E_T1_Q0 = get_t1_channel(tau_gate, T1, 1)
    E_T2_Q0 = get_t2_channel(tau_gate, T2, T1, 1)
    kraus_Q0 = [np.kron(E_T1_Q0[i] @ E_T2_Q0[j], np.eye(2)) 
                for i in range(len(E_T1_Q0)) for j in range(len(E_T2_Q0))]
    
    E_T1_Q1 = get_t1_channel(tau_gate, T1, 1)
    E_T2_Q1 = get_t2_channel(tau_gate, T2, T1, 1)
    kraus_Q1 = [np.kron(np.eye(2), E_T1_Q1[i] @ E_T2_Q1[j])
                for i in range(len(E_T1_Q1)) for j in range(len(E_T2_Q1))]
                
    # 1. Hadamard Gate (H_1 on Q0)
    rho = H_1 @ rho @ H_1.conj().T
    rho = apply_channel(rho, kraus_Q0)
    rho = apply_channel(rho, kraus_Q1)

    # 2. CNOT Gate
    rho = CNOT @ rho @ CNOT.conj().T
    p = p_gate_error_CNOT
    rho = (1.0 - p) * rho + p * (np.eye(4) / 4.0)

    # Add post-CNOT decoherence
    rho = apply_channel(rho, kraus_Q0)
    rho = apply_channel(rho, kraus_Q1)

    total_p_error = p_gate_error_CNOT * 2
    
    return rho, total_p_error


# --- 4. DATA STRUCTURE AND MAIN EXECUTION ---

# List of hardware systems to simulate. T1, T2, and tau_gate are in seconds.
# CNOT Error is a fraction (e.g., 0.005 for 0.5%).
# NOTE: Replace the remaining placeholder rows (e.g., 'System 6: Placeholder...') 
# with your 23 actual systems and their parameters.
SYSTEMS_DATA = [
    # T1 (s), T2 (s), tau_gate (s), CNOT Error (frac), System Name
    (300.0e-6, 200.0e-6, 0.25e-6, 0.005, "System 1: High-End Superconducting Qubit"),
    (100.0e-6, 75.0e-6, 0.50e-6, 0.015, "System 2: Mid-Range Superconducting Qubit"),
    ( 50.0e-6, 40.0e-6, 1.00e-6, 0.030, "System 3: Low-End SC Qubit (Pre-Advantage)"),
    (5000.0e-6, 3000.0e-6, 25.0e-6, 0.010, "System 4: Trapped Ion Qubit (Slow Gate)"),
    ( 20.0e-6, 15.0e-6, 0.05e-6, 0.050, "System 5: Short-Coherence Qubit (High Error)"),
    ( 1.0e-6, 1.0e-6, 1.0e-6, 0.010, "System 6: Placeholder Data System"), # Placeholder
    ( 1.0e-6, 1.0e-6, 1.0e-6, 0.010, "System 7: Placeholder Data System"), # Placeholder
    # ... Add up to 23 systems here ...
]

if __name__ == '__main__':
    
    print("-" * 75)
    print("| Quantum Noise Simulation: R vs. C_Bell^max (23 System Analysis) |")
    print("-" * 75)
    
    print(f"| {'System Name':<40} | {'T1 (us)':>8} | {'T2 (us)':>8} | {'R':>8} | {'C_Bell^max':>10} |")
    print("|" + "-" * 73 + "|")

    for T1_sim, T2_sim, tau_gate, p_CNOT_error, system_name in SYSTEMS_DATA:
        
        # 1. Run the simulation
        rho_final, total_p_error = simulate_bell_circuit(T1_sim, T2_sim, tau_gate, p_CNOT_error)
        
        # 2. Calculate the metrics
        R_val = calculate_R(rho_final, total_p_error)
        S_val_unnormalized, C_Bell_val = bell_correlation(rho_final)
        
        # 3. Print the results
        print(f"| {system_name:<40} | {T1_sim*1e6:>8.1f} | {T2_sim*1e6:>8.1f} | {R_val:>8.4f} | {C_Bell_val:>10.4f} |")
        
    print("-" * 75)
    print("\n" + "=" * 75)
    print("Next Steps: Populate the SYSTEMS_DATA list with your 23 systems' actual T1, T2, tau_gate, and CNOT error values, then run the script again.")
    print("=" * 75)