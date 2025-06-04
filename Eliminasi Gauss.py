import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Bagian 1: Definisi Fungsi Eliminasi Gauss (DIPERBAIKI)
# -----------------------------------------------------------------------------
def gaussian_elimination(matrix_augmented_input):
    try:
        A = np.array(matrix_augmented_input, dtype=float)
    except ValueError:
        return None, "Input matriks tidak valid. Pastikan semua elemen adalah angka.", []

    if A.ndim != 2 or A.shape[0] == 0:
        return None, "Input matriks tidak valid. Harap masukkan matriks 2D yang valid.", []

    m, n = A.shape  # m = jumlah persamaan, n = jumlah kolom (termasuk konstanta b)

    if m != n - 1:
        return None, (
            f"Matriks augmented harus memiliki kolom satu lebih banyak dari baris "
            f"(diterima {m} baris, {n} kolom). Untuk {m} persamaan, "
            f"harus ada {m+1} kolom."
        ), []

    # List untuk menyimpan langkah-langkah
    steps = []
    
    # Simpan matriks awal
    steps.append({
        'type': 'initial',
        'matrix': A.copy(),
        'description': 'Matriks augmented awal [A|b]'
    })

    # Toleransi yang lebih ketat untuk pembulatan
    tolerance = 1e-10

    # Proses eliminasi maju
    for k in range(m):
        # Pivoting parsial: cari baris dengan elemen absolut terbesar di kolom k
        pivot_values = np.abs(A[k:m, k])
        if not pivot_values.size:
            continue 
        
        i_max_local = np.argmax(pivot_values)
        i_max = k + i_max_local

        # Periksa singularitas
        if np.isclose(A[i_max, k], 0.0, atol=tolerance):
            return None, "Matriks singular atau sistem tidak memiliki solusi unik (elemen pivot nol setelah pivoting).", steps

        # Tukar baris k dengan baris i_max
        if k != i_max:
            A[[k, i_max]] = A[[i_max, k]]
            steps.append({
                'type': 'pivot',
                'matrix': A.copy(),
                'description': f'Tukar baris {k+1} dengan baris {i_max+1} (pivoting)',
                'pivot_row': k,
                'swapped_row': i_max
            })

        # Lakukan eliminasi untuk baris di bawah baris pivot k
        for i in range(k + 1, m):
            if np.isclose(A[k, k], 0.0, atol=tolerance):
                 return None, "Terjadi pembagian dengan nol yang tidak terduga saat eliminasi.", steps
            
            factor = A[i, k] / A[k, k]
            if not np.isclose(factor, 0.0, atol=tolerance):
                # PERBAIKAN: Operasi yang lebih presisi
                for j in range(k, n):
                    A[i, j] = A[i, j] - factor * A[k, j]
                
                # Pastikan elemen yang seharusnya nol benar-benar nol
                A[i, k] = 0.0
                
                steps.append({
                    'type': 'elimination',
                    'matrix': A.copy(),
                    'description': f'R{i+1} = R{i+1} - ({factor:.6f}) × R{k+1}',
                    'eliminated_row': i,
                    'pivot_row': k,
                    'factor': factor
                })

    # Periksa konsistensi sistem
    for i in range(m):
        row_sum_coeffs = np.sum(np.abs(A[i, :m]))
        if np.isclose(row_sum_coeffs, 0.0, atol=tolerance) and not np.isclose(A[i, m], 0.0, atol=tolerance):
            return None, "Sistem tidak konsisten (tidak ada solusi).", steps
        if np.isclose(row_sum_coeffs, 0.0, atol=tolerance) and np.isclose(A[i, m], 0.0, atol=tolerance):
            return None, "Sistem memiliki solusi tak terhingga atau dependen (tidak ada solusi unik).", steps

    # Simpan matriks setelah eliminasi maju
    steps.append({
        'type': 'forward_complete',
        'matrix': A.copy(),
        'description': 'Matriks setelah eliminasi maju (bentuk segitiga atas)'
    })

    # Proses substitusi mundur (DIPERBAIKI)
    x = np.zeros(m)
    substitution_steps = []
    
    for i in range(m - 1, -1, -1):
        if np.isclose(A[i, i], 0.0, atol=tolerance):
            return None, "Pembagian dengan nol saat substitusi mundur (indikasi matriks singular).", steps
        
        # PERBAIKAN: Perhitungan substitusi mundur yang lebih akurat
        sum_ax = 0.0
        for j in range(i + 1, m):
            sum_ax += A[i, j] * x[j]
        
        x[i] = (A[i, -1] - sum_ax) / A[i, i]
        
        # Buat deskripsi langkah substitusi mundur
        if i == m - 1:
            substitution_steps.append(f'x{i+1} = {A[i, -1]:.4f} / {A[i, i]:.4f} = {x[i]:.6f}')
        else:
            substitution_desc = f'x{i+1} = ({A[i, -1]:.4f}'
            for j in range(i+1, m):
                if not np.isclose(A[i, j], 0.0, atol=tolerance):
                    substitution_desc += f' - ({A[i, j]:.4f}) × ({x[j]:.6f})'
            substitution_desc += f') / {A[i, i]:.4f} = {x[i]:.6f}'
            substitution_steps.append(substitution_desc)
    
    steps.append({
        'type': 'back_substitution',
        'substitution_steps': substitution_steps,
        'description': 'Substitusi mundur untuk mencari solusi'
    })
    
    return x, None, steps

# -----------------------------------------------------------------------------
# Bagian 2: Fungsi Helper untuk Tampilan (DIPERBAIKI)
# -----------------------------------------------------------------------------
def display_matrix(matrix, step_num=None, description=""):
    if step_num is not None:
        st.markdown(f"**Langkah {step_num}:** {description}")
    else:
        st.markdown(f"**{description}**")
    
    # Format matriks untuk ditampilkan dengan presisi yang lebih baik
    df_display = pd.DataFrame(matrix)
    
    # Buat nama kolom
    num_vars = matrix.shape[1] - 1
    col_names = [f'x{i+1}' for i in range(num_vars)] + ['b']
    df_display.columns = col_names
    
    # PERBAIKAN: Format angka dengan presisi yang lebih baik
    df_styled = df_display.style.format("{:.6f}").set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ])
    
    st.dataframe(df_styled, use_container_width=True)

def reset_all_data():
    """Reset semua data di session state termasuk data input matriks"""
    keys_to_remove = ['matrix_df', 'solution', 'error_message', 'steps']
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    # Increment counter untuk memaksa refresh data_editor
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    st.session_state.reset_counter += 1

def create_empty_matrix(num_rows, num_cols):
    """Buat matriks kosong (semua nilai 0)"""
    column_names = [f'x{i+1}' for i in range(num_cols-1)] + ['b (konstanta)']
    return pd.DataFrame(
        np.zeros((num_rows, num_cols), dtype=float), 
        columns=column_names
    )

def update_matrix_size(num_rows, num_cols):
    """Update ukuran matriks dengan mempertahankan data yang ada"""
    if 'matrix_df' in st.session_state:
        current_df = st.session_state.matrix_df
        current_rows, current_cols = current_df.shape
        
        # Buat nama kolom yang baru
        column_names = [f'x{i+1}' for i in range(num_cols-1)] + ['b (konstanta)']
        
        # Buat DataFrame baru dengan ukuran yang diinginkan
        new_df = pd.DataFrame(
            np.zeros((num_rows, num_cols), dtype=float), 
            columns=column_names
        )
        
        # Copy data lama yang masih muat
        copy_rows = min(current_rows, num_rows)
        copy_cols = min(current_cols, num_cols)
        
        new_df.iloc[:copy_rows, :copy_cols] = current_df.iloc[:copy_rows, :copy_cols]
        
        st.session_state.matrix_df = new_df
    else:
        st.session_state.matrix_df = create_empty_matrix(num_rows, num_cols)

# PERBAIKAN: Fungsi verifikasi yang lebih akurat
def verify_solution(original_matrix, solution):
    """Verifikasi solusi dengan perhitungan yang lebih akurat"""
    A_original = original_matrix[:, :-1]  # Koefisien
    b_original = original_matrix[:, -1]   # Konstanta
    
    # Hitung Ax dengan presisi tinggi
    result = np.dot(A_original, solution)
    
    # Hitung error relatif dan absolut
    abs_errors = np.abs(result - b_original)
    rel_errors = abs_errors / (np.abs(b_original) + 1e-12)  # Hindari pembagian nol
    
    return result, abs_errors, rel_errors

# -----------------------------------------------------------------------------
# Bagian 3: Antarmuka Pengguna Streamlit (DIPERBAIKI)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Solver Eliminasi Gauss", layout="wide")
st.title("MyKalkulator Eliminasi Gauss 🧮")
st.markdown("""
Aplikasi ini memungkinkan Anda untuk menyelesaikan sistem persamaan linear $Ax = b$ menggunakan metode eliminasi Gauss.
Atur ukuran matriks sesuai kebutuhan, lalu isi koefisien matriks augmented $[A|b]$.
""")

# Sidebar untuk kontrol matriks
st.sidebar.header("⚙️ Pengaturan Matriks")

# Input untuk dimensi matriks
col1, col2 = st.sidebar.columns(2)

with col1:
    num_rows = st.number_input(
        "Jumlah Baris:", 
        min_value=2, 
        max_value=8,
        value=3,
        step=1,
        key="num_rows_input",
        help="Jumlah persamaan dalam sistem"
    )

with col2:
    num_variables = st.number_input(
        "Jumlah Variabel:", 
        min_value=2, 
        max_value=8,
        value=3,
        step=1,
        key="num_vars_input",
        help="Jumlah variabel yang tidak diketahui"
    )

# Validasi dimensi
if num_rows != num_variables:
    st.sidebar.warning("⚠️ Untuk solusi unik, jumlah baris harus sama dengan jumlah variabel!")

num_cols = num_variables + 1

# Tombol untuk reset dan update matriks
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Update Ukuran", help="Perbarui ukuran matriks"):
        update_matrix_size(num_rows, num_cols)
        for key in ['solution', 'error_message', 'steps']:
            if key in st.session_state:
                del st.session_state[key]

with col2:
    if st.button("🗑️ Reset Semua", help="Hapus semua data"):
        reset_all_data()
        st.session_state.matrix_df = create_empty_matrix(num_rows, num_cols)
        st.rerun()

# Informasi matriks saat ini
st.sidebar.info(f"📊 Ukuran matriks: {num_rows} × {num_cols}")

# Template contoh (DIPERBAIKI)
st.sidebar.subheader("📚 Template Contoh")
if st.sidebar.button("📝 Isi Contoh 3×3"):
    if num_rows >= 3 and num_cols == 4:
        # PERBAIKAN: Contoh yang sudah diverifikasi
        example_data = np.array([
            [2.0, 3.0, -1.0, 5.0],
            [4.0, 4.0, -3.0, 3.0],
            [-2.0, 3.0, 2.0, 7.0]
        ], dtype=float)
        
        column_names = [f'x{i+1}' for i in range(num_cols-1)] + ['b (konstanta)']
        st.session_state.matrix_df = pd.DataFrame(example_data[:num_rows, :num_cols], columns=column_names)
        
        for key in ['solution', 'error_message', 'steps']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.rerun()
    else:
        st.warning("Template 3×3 hanya tersedia untuk ukuran 3 baris dan 3 variabel.")

# Inisialisasi
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

if 'matrix_df' not in st.session_state:
    update_matrix_size(num_rows, num_cols)

# Main content area
st.subheader(f"📝 Input Matriks Augmented [{num_rows}×{num_cols}]")

# PERBAIKAN: Konfigurasi format input yang lebih presisi
column_config_editor = {}
for name in st.session_state.matrix_df.columns:
    column_config_editor[name] = st.column_config.NumberColumn(
        label=name,
        step=0.01,  # Langkah yang lebih kecil untuk presisi
        format="%.4f",  # Format dengan 4 desimal
    )

editor_key = f"data_editor_main_matrix_{st.session_state.reset_counter}"

with st.form(key="matrix_input_form"):
    edited_df = st.data_editor(
        st.session_state.matrix_df, 
        column_config=column_config_editor,
        num_rows="fixed",
        key=editor_key,
        use_container_width=True,
        hide_index=False
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.form_submit_button(
            label="🚀 Selesaikan Sistem Persamaan", 
            use_container_width=True,
            type="primary"
        )

# Logika setelah tombol submit ditekan
if submit_button:
    st.session_state.matrix_df = edited_df
    
    matrix_input_numpy = edited_df.to_numpy(dtype=float)
    
    if np.isnan(matrix_input_numpy).any():
        st.session_state.error_message = "Harap isi semua sel dalam matriks dengan nilai numerik."
        if 'solution' in st.session_state: del st.session_state.solution
        if 'steps' in st.session_state: del st.session_state.steps
    else:
        solution, error_msg, steps = gaussian_elimination(matrix_input_numpy)
        st.session_state.solution = solution
        st.session_state.error_message = error_msg
        st.session_state.steps = steps

# Tampilkan hasil
if 'error_message' in st.session_state and st.session_state.error_message:
    st.error(st.session_state.error_message)
elif 'solution' in st.session_state and st.session_state.solution is not None:
    st.success("Solusi berhasil ditemukan! 🎉")
    
    # Tampilkan langkah-langkah
    if 'steps' in st.session_state and st.session_state.steps:
        st.markdown("## 🔢 Langkah-langkah Penyelesaian")
        
        step_counter = 1
        
        for step in st.session_state.steps:
            if step['type'] == 'initial':
                display_matrix(step['matrix'], step_counter, step['description'])
                step_counter += 1
                
            elif step['type'] == 'pivot':
                display_matrix(step['matrix'], step_counter, step['description'])
                step_counter += 1
                
            elif step['type'] == 'elimination':
                display_matrix(step['matrix'], step_counter, step['description'])
                step_counter += 1
                
            elif step['type'] == 'forward_complete':
                display_matrix(step['matrix'], step_counter, step['description'])
                step_counter += 1
                
            elif step['type'] == 'back_substitution':
                st.markdown(f"**Langkah {step_counter}:** {step['description']}")
                for i, sub_step in enumerate(step['substitution_steps']):
                    st.markdown(f"• {sub_step}")
                step_counter += 1
    
    # Tampilkan solusi akhir dengan presisi tinggi
    st.markdown("## ⚡ Solusi Akhir")
    solution_output = st.session_state.solution
    
    # PERBAIKAN: Format solusi yang lebih presisi
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Solusi dalam notasi matematika:**")
        solution_latex = "\\begin{pmatrix}\n"
        for i, val in enumerate(solution_output):
            solution_latex += f"x_{{{i+1}}} = {val:.6f} \\\\\n"
        solution_latex = solution_latex.rstrip("\\\\\n")
        solution_latex += "\n\\end{pmatrix}"
        
        try:
            st.latex(solution_latex)
        except Exception:
            for i, val in enumerate(solution_output):
                st.text(f"x{i+1} = {val:.6f}")
    
    with col2:
        st.markdown("**Solusi dalam bentuk tabel:**")
        solution_df = pd.DataFrame({
            'Variabel': [f'x{i+1}' for i in range(len(solution_output))],
            'Nilai': [f'{val:.6f}' for val in solution_output]
        })
        st.dataframe(solution_df, use_container_width=True, hide_index=True)
    
    # PERBAIKAN: Verifikasi yang lebih komprehensif
    st.markdown("## ✅ Verifikasi Solusi")
    original_matrix = st.session_state.steps[0]['matrix'] if 'steps' in st.session_state else None
    if original_matrix is not None:
        result, abs_errors, rel_errors = verify_solution(original_matrix, solution_output)
        
        verification_df = pd.DataFrame({
            'Persamaan': [f'Persamaan {i+1}' for i in range(len(result))],
            'Ax (Hasil)': [f'{val:.8f}' for val in result],
            'b (Target)': [f'{val:.8f}' for val in original_matrix[:, -1]],
            'Error Absolut': [f'{val:.2e}' for val in abs_errors],
            'Error Relatif': [f'{val:.2e}' for val in rel_errors],
            'Status': ['✅ Akurat' if err < 1e-6 else '⚠️ Kurang Akurat' for err in abs_errors]
        })
        
        st.dataframe(verification_df, use_container_width=True, hide_index=True)
        
        # Ringkasan akurasi
        max_abs_error = np.max(abs_errors)
        max_rel_error = np.max(rel_errors)
        
        if max_abs_error < 1e-10:
            st.success(f"🎯 Solusi sangat akurat! Error maksimum: {max_abs_error:.2e}")
        elif max_abs_error < 1e-6:
            st.info(f"✅ Solusi akurat. Error maksimum: {max_abs_error:.2e}")
        else:
            st.warning(f"⚠️ Solusi kurang akurat. Error maksimum: {max_abs_error:.2e}")

st.markdown("---")
st.markdown("**Perbaikan pada versi ini:** Algoritma eliminasi Gauss yang lebih stabil dan akurat 💝")
