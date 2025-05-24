import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Bagian 1: Definisi Fungsi Eliminasi Gauss
# -----------------------------------------------------------------------------
def gaussian_elimination(matrix_augmented_input):
    try:
        A = np.array(matrix_augmented_input, dtype=float)
    except ValueError:
        return None, "Input matriks tidak valid. Pastikan semua elemen adalah angka.", []

    if A.ndim != 2 or A.shape[0] == 0:  # Periksa apakah matriks 2D dan tidak kosong
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

    # Proses eliminasi maju
    for k in range(m):
        # Pivoting parsial: cari baris dengan elemen absolut terbesar di kolom k (mulai dari baris k)
        pivot_values = np.abs(A[k:m, k])
        if not pivot_values.size: # Jika tidak ada elemen pivot (misalnya, k >= m)
            continue 
        
        i_max_local = np.argmax(pivot_values)
        i_max = k + i_max_local # Indeks absolut di matriks A

        # Periksa singularitas atau pembagian dengan nol
        # Toleransi kecil untuk membandingkan float dengan nol
        if np.isclose(A[i_max, k], 0.0, atol=1e-12):
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
            if np.isclose(A[k, k], 0.0, atol=1e-12): # Seharusnya tidak terjadi karena pivoting
                 return None, "Terjadi pembagian dengan nol yang tidak terduga saat eliminasi.", steps
            
            factor = A[i, k] / A[k, k]
            if not np.isclose(factor, 0.0, atol=1e-12):  # Hanya lakukan eliminasi jika factor != 0
                A[i, k:n] = A[i, k:n] - factor * A[k, k:n]
                A[i, k] = 0.0 # Pastikan nol untuk presisi numerik
                
                steps.append({
                    'type': 'elimination',
                    'matrix': A.copy(),
                    'description': f'R{i+1} = R{i+1} - ({factor:.4f}) × R{k+1}',
                    'eliminated_row': i,
                    'pivot_row': k,
                    'factor': factor
                })

    # Periksa kembali singularitas setelah eliminasi
    # Jika ada baris [0 0... 0 | c] di mana c!= 0, maka tidak ada solusi
    # Jika ada baris [0 0... 0 | 0], maka ada tak hingga solusi (jika konsisten)
    for i in range(m):
        row_sum_coeffs = np.sum(np.abs(A[i, :m]))
        if np.isclose(row_sum_coeffs, 0.0, atol=1e-12) and not np.isclose(A[i, m], 0.0, atol=1e-12):
            return None, "Sistem tidak konsisten (tidak ada solusi).", steps
        if np.isclose(row_sum_coeffs, 0.0, atol=1e-12) and np.isclose(A[i, m], 0.0, atol=1e-12):
            # Ini bisa menjadi indikasi solusi tak terhingga, perlu penanganan lebih lanjut
            # Untuk implementasi ini, kita bisa menganggapnya sebagai tidak ada solusi *unik*
            return None, "Sistem memiliki solusi tak terhingga atau dependen (tidak ada solusi unik).", steps

    # Simpan matriks setelah eliminasi maju
    steps.append({
        'type': 'forward_complete',
        'matrix': A.copy(),
        'description': 'Matriks setelah eliminasi maju (bentuk segitiga atas)'
    })

    # Proses substitusi mundur
    x = np.zeros(m)
    substitution_steps = []
    
    for i in range(m - 1, -1, -1):
        if np.isclose(A[i, i], 0.0, atol=1e-12):
            # Ini seharusnya sudah ditangkap oleh pemeriksaan singularitas sebelumnya
            return None, "Pembagian dengan nol saat substitusi mundur (indikasi matriks singular).", steps
        
        sum_ax = np.dot(A[i, i+1:n-1], x[i+1:n-1])
        x[i] = (A[i, -1] - sum_ax) / A[i, i]
        
        # Buat deskripsi langkah substitusi mundur
        if i == m - 1:
            # Persamaan terakhir (paling sederhana)
            substitution_steps.append(f'x{i+1} = {A[i, -1]:.4f} / {A[i, i]:.4f} = {x[i]:.4f}')
        else:
            # Persamaan dengan substitusi
            substitution_desc = f'x{i+1} = ({A[i, -1]:.4f}'
            for j in range(i+1, m):
                if not np.isclose(A[i, j], 0.0, atol=1e-12):
                    substitution_desc += f' - ({A[i, j]:.4f}) × ({x[j]:.4f})'
            substitution_desc += f') / {A[i, i]:.4f} = {x[i]:.4f}'
            substitution_steps.append(substitution_desc)
    
    steps.append({
        'type': 'back_substitution',
        'substitution_steps': substitution_steps,
        'description': 'Substitusi mundur untuk mencari solusi'
    })
    
    return x, None, steps # Solusi ditemukan, tidak ada error

# -----------------------------------------------------------------------------
# Bagian 2: Antarmuka Pengguna Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Solver Eliminasi Gauss", layout="wide")
st.title("MyKalkulator Eliminasi Gauss")
st.markdown("""
Aplikasi ini memungkinkan Anda untuk menyelesaikan sistem persamaan linear $Ax = b$ menggunakan metode eliminasi Gauss.
Masukkan jumlah persamaan, lalu isi koefisien matriks augmented $[A|b]$ di bawah ini.
""")

# Input untuk jumlah persamaan
num_equations = st.number_input(
    "Masukkan jumlah persamaan (n):", 
    min_value=1, 
    max_value=10,
    value=3,
    step=1,
    key="num_equations_selector",
    help="Jumlah persamaan akan menentukan ukuran matriks n x (n+1)."
)

# Membuat nama kolom yang dinamis berdasarkan jumlah persamaan
num_variables = num_equations
num_cols_augmented = num_variables + 1
column_names = [f'x{i+1}' for i in range(num_variables)] + ['b (konstanta)']

# Inisialisasi atau update DataFrame di session state
# Ini penting agar data editor tidak reset saat num_equations berubah
if 'matrix_df' not in st.session_state or \
   st.session_state.matrix_df.shape[0] != num_equations or \
   st.session_state.matrix_df.shape[1] != num_cols_augmented:
    
    st.session_state.matrix_df = pd.DataFrame(
        np.zeros((num_equations, num_cols_augmented), dtype=int), 
        columns=column_names
    )
    # Reset juga solusi jika matriks diubah ukurannya
    if 'solution' in st.session_state:
        del st.session_state.solution
    if 'error_message' in st.session_state:
        del st.session_state.error_message


st.subheader("Masukkan Matriks Augmented $[A|b]$:")

# Konfigurasi kolom untuk st.data_editor - TANPA DESIMAL
column_config_editor = {}
for name in column_names:
    column_config_editor[name] = st.column_config.NumberColumn(
        label=name,
        step=1,  # Menentukan langkah increment/decrement
        format="%d",  # Format tampilan sebagai integer
        # help=f"Koefisien untuk variabel {name}" if 'x' in name else "Nilai konstanta sisi kanan"
    )

# Gunakan form untuk mengelompokkan input data editor dan tombol submit
with st.form(key="matrix_input_form"):
    edited_df = st.data_editor(
        st.session_state.matrix_df, 
        column_config=column_config_editor,
        num_rows="fixed", # Jumlah baris tetap sesuai num_equations
        key="data_editor_main_matrix",
        use_container_width=True
    )
    submit_button = st.form_submit_button(label="Selesaikan Sistem Persamaan")

# Logika setelah tombol submit ditekan
if submit_button:
    st.session_state.matrix_df = edited_df # Simpan data terbaru dari editor
    
    # Validasi input dari DataFrame
    matrix_input_numpy = edited_df.to_numpy(dtype=float)
    
    if np.isnan(matrix_input_numpy).any():
        st.session_state.error_message = "Harap isi semua sel dalam matriks dengan nilai numerik."
        if 'solution' in st.session_state: del st.session_state.solution # Hapus solusi lama
        if 'steps' in st.session_state: del st.session_state.steps # Hapus langkah lama
    else:
        # Panggil fungsi eliminasi Gauss
        solution, error_msg, steps = gaussian_elimination(matrix_input_numpy)
        st.session_state.solution = solution
        st.session_state.error_message = error_msg
        st.session_state.steps = steps

# Fungsi helper untuk menampilkan matriks dalam format yang rapi
def display_matrix(matrix, step_num=None, description=""):
    if step_num is not None:
        st.markdown(f"**Langkah {step_num}:** {description}")
    else:
        st.markdown(f"**{description}**")
    
    # Format matriks untuk ditampilkan
    df_display = pd.DataFrame(matrix)
    
    # Buat nama kolom
    num_vars = matrix.shape[1] - 1
    col_names = [f'x{i+1}' for i in range(num_vars)] + ['b']
    df_display.columns = col_names
    
    # Format angka untuk tampilan yang lebih rapi
    df_styled = df_display.style.format("{:.4f}").set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ])
    
    st.dataframe(df_styled, use_container_width=True)

# Tampilkan hasil atau error di luar form, berdasarkan session state
if 'error_message' in st.session_state and st.session_state.error_message:
    st.error(st.session_state.error_message)
elif 'solution' in st.session_state and st.session_state.solution is not None:
    st.success("Solusi berhasil ditemukan!")
    
    # Tampilkan langkah-langkah penyelesaian
    if 'steps' in st.session_state and st.session_state.steps:
        st.markdown("## 📋 Langkah-langkah Penyelesaian")
        
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
    
    # Tampilkan solusi akhir
    st.markdown("## 🎯 Solusi Akhir")
    solution_output = st.session_state.solution
    
    # Membuat string LaTeX untuk solusi
    solution_latex = "\\begin{pmatrix}\n"
    for i, val in enumerate(solution_output):
        solution_latex += f"x_{{{i+1}}} = {val:.4f} \\\\\n"
    solution_latex = solution_latex.rstrip("\\\\\n") # Hapus \\ dan newline terakhir
    solution_latex += "\n\\end{pmatrix}"
    
    try:
        st.latex(solution_latex)
    except Exception as e:
        st.warning(f"Gagal menampilkan solusi dalam format LaTeX. Menampilkan sebagai teks biasa. Error: {e}")
        for i, val in enumerate(solution_output):
            st.text(f"x{i+1} = {val:.4f}")
    
    # Verifikasi solusi
    st.markdown("## ✅ Verifikasi Solusi")
    original_matrix = st.session_state.steps[0]['matrix'] if 'steps' in st.session_state else None
    if original_matrix is not None:
        A_original = original_matrix[:, :-1]  # Koefisien
        b_original = original_matrix[:, -1]   # Konstanta
        
        # Hitung Ax
        result = np.dot(A_original, solution_output)
        
        verification_df = pd.DataFrame({
            'Persamaan': [f'Persamaan {i+1}' for i in range(len(b_original))],
            'Ax (Hasil)': [f'{val:.4f}' for val in result],
            'b (Seharusnya)': [f'{val:.4f}' for val in b_original],
            'Selisih': [f'{abs(result[i] - b_original[i]):.6f}' for i in range(len(b_original))]
        })
        
        st.dataframe(verification_df, use_container_width=True)

st.markdown("---")
st.markdown("Dibuat dengan cinta dan kasih sayang.")
